"""
Quick checkpoint inspection script.
Run this in the RSRNav conda env to verify the checkpoint loads correctly.

Usage:
    conda activate rsrnav
    python offline/inspect_checkpoint.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoint", "checkpoint.pth"
    )

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"  File size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB")

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
        prefix = k.split(".")[0] if "." in k else k
        if prefix not in modules:
            modules[prefix] = {"keys": [], "params": 0}
        modules[prefix]["keys"].append(k)
        if isinstance(sd[k], torch.Tensor):
            modules[prefix]["params"] += sd[k].numel()

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

    # Try to determine what config was used
    if "config" in ckpt:
        config = ckpt["config"]
        print(f"\n=== Config (type: {type(config).__name__}) ===")
        if hasattr(config, "habitat_baselines"):
            hb = config.habitat_baselines
            if hasattr(hb, "rl") and hasattr(hb.rl, "ppo"):
                ppo = hb.rl.ppo
                for attr in ["backbone", "goal_backbone", "rnn_type",
                             "num_recurrent_layers", "hidden_size",
                             "input_size", "visual_encoder_embedding_size",
                             "goal_embedding_size"]:
                    if hasattr(ppo, attr):
                        print(f"  ppo.{attr}: {getattr(ppo, attr)}")

    print("\n=== Checkpoint OK ===")


if __name__ == "__main__":
    main()
