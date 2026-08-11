"""kev03 weight exporter — PyTorch checkpoint -> C++ inference blob.

Walks a GoldRushNet state_dict and writes two files the dependency-free C++
engine (ml/csrc/nn_infer.cpp) can load:

  <out>/weights.bin       every tensor's data, little-endian float32,
                          concatenated in manifest order, no padding
  <out>/manifest.json     {"format": "kev03-weights-v1",
                           "dtype": "float32", "endianness": "little",
                           "total_bytes": N,
                           "config": {...model dims...},
                           "tensors": [{"name", "shape", "offset",
                                        "numel"}, ...]}

`offset` is a BYTE offset into weights.bin.  Tensors keep their PyTorch
memory layout (row-major / C order): Conv2d weight [out_c, in_c, kh, kw],
Linear weight [out_features, in_features].  The critic is EXCLUDED by
default — it is privileged, training-only, and must never ship
(--include-critic exists only for training-side debugging).

Usage:
  python3 ml/export_weights.py --ckpt ml/ckpt/imitation_0020000.pt --out ml/export
  python3 ml/export_weights.py --smoke     # random net round-trip self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _require_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        sys.stderr.write(
            "[kev03] PyTorch is not installed — export_weights.py cannot run.\n"
            "        Install with:  pip install -r ml/requirements.txt\n"
            "        (The C++ side only needs the already-exported blob.)\n")
        return False


if not _require_torch():
    raise SystemExit(2)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

FORMAT = "kev03-weights-v1"
SIZE_BUDGET_BYTES = 16 * 1024 * 1024   # contest cap for the whole .so


def export_model(net: torch.nn.Module, out_dir: str,
                 include_critic: bool = False) -> dict:
    """Write weights.bin + manifest.json; return the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    offset = 0
    blob = bytearray()
    for name, tensor in net.state_dict().items():
        if not include_critic and name.startswith("critic."):
            continue
        arr = tensor.detach().cpu().numpy().astype("<f4")   # LE float32
        blob += arr.tobytes(order="C")
        entries.append({"name": name, "shape": list(arr.shape),
                        "offset": offset, "numel": int(arr.size)})
        offset += arr.nbytes
    manifest = {
        "format": FORMAT,
        "dtype": "float32",
        "endianness": "little",
        "total_bytes": offset,
        "config": {
            "GRID": M.GRID, "C_OBS": M.C_OBS, "C_ENC": M.C_ENC,
            "C_INTERP": M.C_INTERP, "C_LATENT": M.C_LATENT,
            "C_BELIEF": M.C_BELIEF, "D_TOKEN": M.D_TOKEN,
            "interpretable_channels": M.get_interpretable_channels(),
        },
        "tensors": entries,
    }
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_roundtrip(net: torch.nn.Module, out_dir: str,
                     include_critic: bool = False) -> None:
    """Re-read the blob and check byte-exact equality with the state_dict."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["format"] == FORMAT
    blob = np.fromfile(os.path.join(out_dir, "weights.bin"), dtype="<f4")
    assert blob.nbytes == manifest["total_bytes"], \
        f"blob {blob.nbytes} B != manifest total_bytes {manifest['total_bytes']}"
    sd = net.state_dict()
    seen = set()
    for e in manifest["tensors"]:
        name, shape, off, numel = e["name"], e["shape"], e["offset"], e["numel"]
        assert off % 4 == 0, f"{name}: unaligned offset {off}"
        got = blob[off // 4: off // 4 + numel].reshape(shape)
        want = sd[name].detach().cpu().numpy().astype("<f4")
        assert got.shape == tuple(want.shape), f"{name}: shape mismatch"
        assert np.array_equal(got, want), f"{name}: data round-trip mismatch"
        seen.add(name)
    expect = {n for n in sd if include_critic or not n.startswith("critic.")}
    assert seen == expect, f"tensor set mismatch: {seen ^ expect}"


def run_smoke() -> int:
    import tempfile
    torch.manual_seed(0)
    net = M.GoldRushNet(with_critic=True)   # critic present, must be skipped
    with tempfile.TemporaryDirectory(prefix="kev03_export_") as td:
        manifest = export_model(net, td, include_critic=False)
        verify_roundtrip(net, td, include_critic=False)
        n_tensors = len(manifest["tensors"])
        n_bytes = manifest["total_bytes"]
        assert not any(e["name"].startswith("critic.")
                       for e in manifest["tensors"]), "critic leaked into export"
        print(f"[kev03][smoke] exported {n_tensors} tensors, "
              f"{n_bytes / 1e6:.2f} MB fp32 "
              f"({100.0 * n_bytes / SIZE_BUDGET_BYTES:.1f}% of the 16 MB .so cap; "
              "remember the code + tables share that cap)")
        print("[kev03][smoke] export round-trip PASSED (byte-exact)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="", help=".pt checkpoint (train scripts')")
    ap.add_argument("--out", default="ml/export", help="output directory")
    ap.add_argument("--include-critic", action="store_true",
                    help="debug only — the critic must never ship")
    ap.add_argument("--smoke", action="store_true",
                    help="random-net export + byte-exact re-read, exit 0")
    args = ap.parse_args(argv)
    if args.smoke:
        return run_smoke()
    if not args.ckpt:
        ap.error("--ckpt is required unless --smoke")
    net = M.GoldRushNet(with_critic=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck.get("model", ck)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[kev03] note: {len(missing)} params missing from ckpt "
              "(fresh init used)")
    manifest = export_model(net, args.out, include_critic=args.include_critic)
    verify_roundtrip(net, args.out, include_critic=args.include_critic)
    print(f"[kev03] wrote {args.out}/weights.bin "
          f"({manifest['total_bytes'] / 1e6:.2f} MB, "
          f"{len(manifest['tensors'])} tensors) + manifest.json; "
          "round-trip verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
