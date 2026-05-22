#!/usr/bin/env bash
# add_nodes.sh — Add more FL worker nodes to a running training session.
#
# Assumes the GPU instance already has Docker, NVIDIA toolkit, and the
# node Docker image built (i.e., launch_node.sh was run before with full setup).
# This script is a thin wrapper that skips setup and just starts new containers.
#
# Usage:
#   bash scripts/oci/add_nodes.sh --server 10.0.0.5:8080 --speakers BWC,LXC
#   bash scripts/oci/add_nodes.sh --server 10.0.0.5:8080 --speakers YKWK,HJK,ERMS,EBVS
#
# The new nodes will connect to the server and join the next training round.

set -euo pipefail

# Delegate to launch_node.sh with --skip_setup flag
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/launch_node.sh" --skip_setup "$@"
