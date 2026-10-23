"""
Quick checkpoint inspection script.

Stubs out habitat's Config class to avoid importing habitat-lab
(which has OpenCV/NumPy compatibility issues). This lets us inspect
the checkpoint's state_dict without a working habitat installation.

Usage:
    python offline/inspect_checkpoint.py
    python offline/inspect_checkpoint.py --path /path/to/checkpoint.pth
"""
import argparse
import io
import pickle
import sys
import os
import types

import torch


class _StubConfig:
    """Dummy stand-in for habitat.config.default.Config."""
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, '_data', {})
        for k, v in kwargs.items():
            self._data[k] = v
    def __repr__(self):
        return "StubConfig(...)"
    def __getattr__(self, name):
        try:
            data = object.__getattribute__(self, '_data')
        except AttributeError:
            return _StubConfig()
        return data.get(name, _StubConfig())
    def __setattr__(self, name, value):
        if name == '_data':
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
    def __getitem__(self, key):
        return self._data.get(key, _StubConfig())
    def __setitem__(self, key, value):
        self._data[key] = value
    def __contains__(self, key):
        return key in self._data
    def __iter__(self):
        return iter(self._data)
    def __len__(self):
        return len(self._data)
    def __bool__(self):
        return True
    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def items(self):
        return self._data.items()
    def get(self, key, default=None):
        return self._data.get(key, default)
    def __getstate__(self):
        return self._data
    def __setstate__(self, state):
        object.__setattr__(self, '_data',
                           state if isinstance(state, dict) else {})


def _install_stubs():
    """
    Create fake habitat modules so torch.load can unpickle
    habitat.config.default.Config without actually importing habitat.
    """
    stub_modules = [
        "habitat",
        "habitat.config",
        "habitat.config.default",
        "habitat.config.default_structured_configs",
    ]
    for mod_name in stub_modules:
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod

    # The checkpoint pickled habitat.config.default.Config
    sys.modules["habitat.config.default"].Config = _StubConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None,
                       help="Checkpoint path (default: checkpoint/checkpoint.pth)")
    args = parser.parse_args()

    ckpt_path = args.path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoint", "checkpoint.pth"
    )

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"  File size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB")

    # Install stubs BEFORE torch.load so it can unpickle habitat Config
    _install_stubs()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\n=== Top-Level Keys ===")
    for k in sorted(ckpt.keys()):
        v = ckpt[k]
        if isinstance(v, dict):
            print(f"  {k}: dict ({len(v)} keys)")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {list(v.shape)}")
        elif isinstance(v, int):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # Find the state dict
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))

    print(f"\n=== State Dict: {len(sd)} keys ===")

    # Group keys by module prefix
    modules = {}
    for k in sorted(sd.keys()):
        # Use first two dotted components as prefix
        parts = k.split(".")
        prefix = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        if prefix not in modules:
            modules[prefix] = {"keys": [], "params": 0}
        modules[prefix]["keys"].append(k)
        if isinstance(sd[k], torch.Tensor):
            modules[prefix]["params"] += sd[k].numel()

    print(f"\n--- By module ---")
    for prefix, info in sorted(modules.items()):
        print(f"  {prefix}: {len(info['keys'])} keys, {info['params']:,} params")

    # Print all keys with shapes
    print(f"\n=== All Keys ===")
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    total_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
    print(f"\nTotal parameters: {total_params:,}")

    print("\n=== Checkpoint OK ===")


if __name__ == "__main__":
    main()
