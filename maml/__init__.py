from .engine import MAMLEngine, compute_wer_k0, compute_wer_k3

__all__ = ["MAMLEngine", "compute_wer_k0", "compute_wer_k3"]


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}
