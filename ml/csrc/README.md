# kev03 C++ inference engine (skeleton)

This directory is the deployment path for the kev03 neural belief-map agent:
the trained PyTorch `GoldRushNet` is exported by `ml/export_weights.py` into
a flat float32 blob, and this dependency-free C++ code runs it inside the
contest `player.so`. The judge machine has **no GPU** and the `.so` is capped
at **16 MB**, so there is no torch, no ONNX runtime, no BLAS — only C++17
stdlib loops.

## Build & run

```bash
g++ -O2 -std=c++17 -o nn_infer nn_infer.cpp
./nn_infer                          # random tiny net + timing
./nn_infer manifest.json weights.bin  # also exercise the blob loader
```

`main()` builds a random-weight net at deployment scale (18-plane input,
16-channel conv, 16-channel ConvGRU) and times one forward pass; that number
is the belief-side cost of one round's decision and must stay well under
~200 µs (the latency race: the faster bot moves before the NPCs and the
opponent — see `docs/KEV03.md` section 7).

## Blob + manifest format (`kev03-weights-v1`)

Written by `ml/export_weights.py`:

- `weights.bin` — every exported tensor's data, **little-endian float32**,
  concatenated with no padding. Tensors keep PyTorch memory layout
  (row-major / C order): `Conv2d.weight` is `[out_c, in_c, kh, kw]`,
  `Linear.weight` is `[out_features, in_features]`.
- `manifest.json` — `format`, `dtype`, `endianness`, `total_bytes`, a
  `config` block with the model dimensions, and `tensors`: a list of
  `{name, shape, offset, numel}` where `offset` is a **byte** offset into
  `weights.bin` and `name` is the PyTorch `state_dict()` key
  (e.g. `core.cell.conv_ih.weight`).

The critic (`critic.*`) is excluded from exports by default — it is the
privileged training-only value head and must never ship.

The loader in `nn_infer.cpp` is a tiny scanner tailored to this
machine-generated manifest, not a general JSON parser. It assumes a
little-endian host (the x86_64 judge machine is).

## Contracts that must match `ml/model.py`

- **GRU gate order is (r, z, n)** in the stacked `3*hidden` dimension, with
  `n = tanh(i_n + r * h_n)` and `h' = (1-z)*n + z*h` — identical to
  `nn.GRUCell` / `ConvGRUCell` in `ml/model.py`. Change both sides or
  neither.
- Convs are stride-1, same-padding (`pad = k/2`), NCHW.
- Observation plane layout (18 planes) is documented in `ml/model.py`'s
  module docstring and built by `ml/dataset.py:build_observation_planes`;
  the C++ side of the shipped bot must rebuild those planes from
  `GameInput` byte-for-byte the same way.

## Status / future work

- [x] conv (any k, same padding), dense GRU, ConvGRU, linear, softmax
- [x] blob + manifest loader, byte-exact against `export_weights.py --smoke`
- [ ] full GoldRushNet forward graph (encoder -> ConvGRU -> aux reads ->
      unit tokens -> cross-attention -> autoregressive heads) wired to the
      manifest names
- [ ] plane construction from `GameInput` (mirror `ml/dataset.py`)
- [ ] legality masking + argmax decode of order/k/moves/vp
- [ ] **AVX-512** kernels (judge is a 32-core AMD EPYC; zen4 AVX-512 gives
      ~8-16x on these convs) and/or **int8** weights — correctness first,
      speed later; measure win-rate-vs-latency in the simulator before
      spending here.
