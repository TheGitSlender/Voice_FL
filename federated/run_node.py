"""
Docker node entry point.

Reads SPEAKER_HASH from environment (set in docker-compose.yml),
connects to the server, and starts a Flower client.

Usage (Docker):
    docker run ... -e SPEAKER_HASH=abc123 voicefl-node

Usage (direct):
    SPEAKER_HASH=abc123 python federated/run_node.py --config configs/poc.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--server_address", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--nodes_dir", type=str, default=None,
                   help="Local nodes directory override (e.g. data/nodes for local runs)")
    return p.parse_args()

def load_config(path: str | None) -> dict:
    from maml import _load_yaml
    defaults = {
        "server_address": "server:8080",
        "device": "cpu",
        "inner_steps": 3,
        "inner_lr": 1e-4,
        "support_size": 8,
        "query_size": 8,
        "max_grad_norm": 10.0,
        "nodes_dir": "/data",
    }
    if path is None:
        return defaults
    cfg = _load_yaml(path)
    defaults.update(cfg.get("node", {}))
    defaults.update(cfg.get("maml", {}))
    return defaults

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.server_address is not None:
        cfg["server_address"] = args.server_address
    if args.device is not None:
        cfg["device"] = args.device
    if args.nodes_dir is not None:
        cfg["nodes_dir"] = args.nodes_dir

    speaker_hash = os.environ.get("SPEAKER_HASH", "")
    if not speaker_hash:
        print("ERROR: SPEAKER_HASH environment variable not set")
        sys.exit(1)

    nodes_dir = cfg["nodes_dir"]
    if nodes_dir == "/data":
        node_dir = Path("/data")
    else:
        node_dir = Path(nodes_dir) / speaker_hash

    if not node_dir.exists():
        print(f"ERROR: Node data directory not found: {node_dir}")
        sys.exit(1)

    print(f"[node] Speaker hash: {speaker_hash[:8]}...")
    print(f"[node] Data dir: {node_dir}")
    print(f"[node] Server: {cfg['server_address']}")
    print(f"[node] Device: {cfg['device']}")

    import time
    import flwr as fl
    from federated.client_maml import MAMLClient

    client = MAMLClient(
        node_dir=node_dir,
        device=cfg["device"],
        inner_steps=cfg["inner_steps"],
        inner_lr=cfg["inner_lr"],
        support_size=cfg["support_size"],
        query_size=cfg["query_size"],
        max_grad_norm=cfg["max_grad_norm"],
    )

    import socket
    host, port_str = cfg["server_address"].rsplit(":", 1)
    port = int(port_str)
    max_wait = 120           
    poll_interval = 3
    elapsed = 0
    print(f"[node] Waiting for server at {host}:{port}...")
    while elapsed < max_wait:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[node] Server is up (waited {elapsed}s)")
                break
        except OSError:
            time.sleep(poll_interval)
            elapsed += poll_interval
    else:
        print(f"[node] ERROR: server not reachable after {max_wait}s")
        sys.exit(1)

    fl.client.start_client(
        server_address=cfg["server_address"],
        client=client,
        grpc_max_message_length=512 * 1024 * 1024,
    )

if __name__ == "__main__":
    main()
