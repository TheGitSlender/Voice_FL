"""
Docker node entry point for FedLoRA-MAML (Phase 2 — VCTK).

Reads SPEAKER_ID from environment (set in docker-compose.vctk.yml).
The speaker's feature directory is mounted at /data by the compose file:
  data/vctk_nodes/{SPEAKER_ID} → /data

Usage (Docker):
    docker run -e SPEAKER_ID=p225 voicefl-lora-node

Usage (direct, for debugging):
    SPEAKER_ID=p225 python federated/run_node_lora.py \\
        --config configs/vctk_lora_poc.yaml \\
        --nodes_dir data/vctk_nodes --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--server_address", type=str,
                   default=os.environ.get("FL_SERVER_ADDRESS"),
                   help="FL server address (host:port). "
                        "Also reads FL_SERVER_ADDRESS env var. "
                        "Defaults to 'server:8080' if neither is set.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--speaker_id", type=str, default=None,
                   help="Speaker ID (e.g. RRBI). "
                        "Also reads SPEAKER_ID env var.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Direct path to speaker data directory "
                        "(must contain features.pt + labels.txt).")
    p.add_argument("--nodes_dir", type=str, default=None,
                   help="Local nodes dir for direct runs (e.g. data/l2arctic_nodes). "
                        "In Docker the compose mounts the speaker dir at /data directly.")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    from maml import _load_yaml

    defaults = {
        "server_address": "server:8080",
        "device": "cuda",
        "inner_steps": 5,
        "inner_lr": 1e-4,
        "support_size": 20,
        "query_size": 30,
        "tasks_per_node": 4,
        "max_audio_samples": None,
        "max_grad_norm": 50.0,
    }
    if path is None:
        return defaults
    cfg = _load_yaml(path)

    maml_cfg = cfg.get("maml", {})
    if "k" in maml_cfg and "inner_steps" not in maml_cfg:
        maml_cfg["inner_steps"] = maml_cfg.pop("k")
    defaults.update(maml_cfg)

    if "hardware" in cfg:
        defaults["device"] = cfg["hardware"].get("device", defaults["device"])

    return defaults


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.server_address is not None:
        cfg["server_address"] = args.server_address
    if args.device is not None:
        cfg["device"] = args.device

    # Speaker ID: CLI arg > env var
    speaker_id = args.speaker_id or os.environ.get("SPEAKER_ID", "")
    if not speaker_id:
        print("ERROR: Speaker ID not set. Use --speaker_id or SPEAKER_ID env var.", flush=True)
        sys.exit(1)

    # In Docker: compose mounts data/vctk_nodes/{speaker_id} → /data
    # Data directory: --data_dir (direct path) > --nodes_dir/speaker_id > /data (Docker)
    if args.data_dir is not None:
        node_dir = Path(args.data_dir)
    elif args.nodes_dir is not None:
        node_dir = Path(args.nodes_dir) / speaker_id
    else:
        node_dir = Path("/data")

    if not node_dir.exists():
        print(f"ERROR: Node data directory not found: {node_dir}", flush=True)
        sys.exit(1)
    if not (node_dir / "features.pt").exists():
        print(f"ERROR: features.pt missing in {node_dir}", flush=True)
        sys.exit(1)

    print(f"[node] Speaker: {speaker_id}", flush=True)
    print(f"[node] Data dir: {node_dir}", flush=True)
    print(f"[node] Server: {cfg['server_address']}", flush=True)
    print(f"[node] Device: {cfg['device']}", flush=True)
    print(
        f"[node] inner_steps={cfg['inner_steps']} inner_lr={cfg['inner_lr']} "
        f"tasks/round={cfg['tasks_per_node']} max_samples={cfg['max_audio_samples']}",
        flush=True,
    )

    import flwr as fl
    from federated.client_lora import MAMLClientLora

    client = MAMLClientLora(
        node_dir=node_dir,
        device=cfg["device"],
        inner_steps=cfg["inner_steps"],
        inner_lr=cfg["inner_lr"],
        support_size=cfg["support_size"],
        query_size=cfg["query_size"],
        tasks_per_node=cfg["tasks_per_node"],
        max_audio_samples=cfg.get("max_audio_samples"),
        speaker_id=speaker_id,
        max_grad_norm=cfg["max_grad_norm"],
    )

    # Poll until the Flower server is reachable
    import socket
    host, port_str = cfg["server_address"].rsplit(":", 1)
    port = int(port_str)
    max_wait = 300
    poll_interval = 3
    elapsed = 0
    print(f"[node] Waiting for server at {host}:{port}...", flush=True)
    while elapsed < max_wait:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[node] Server is up (waited {elapsed}s)", flush=True)
                break
        except OSError:
            time.sleep(poll_interval)
            elapsed += poll_interval
    else:
        print(f"[node] ERROR: server not reachable after {max_wait}s", flush=True)
        sys.exit(1)

    fl.client.start_client(
        server_address=cfg["server_address"],
        client=client,
        grpc_max_message_length=64 * 1024 * 1024,  # 64 MB — matches server
    )


if __name__ == "__main__":
    main()
