// kev03 nn_infer.cpp — dependency-free float32 inference skeleton.
//
// This is the path to embedding the trained kev03 policy inside the contest
// player.so: NO torch, NO BLAS, NO external headers — just C++17 stdlib.
// It provides the primitives the exported GoldRushNet needs:
//
//   * conv2d, 3x3 (any k) same-padding, NCHW row-major
//   * ConvGRU cell  (gate order r, z, n — matches ml/model.py ConvGRUCell
//     and PyTorch nn.GRUCell exactly; change both sides or neither)
//   * dense GRU cell, linear, relu/sigmoid/tanh, softmax
//   * loader for the weights.bin + manifest.json pair written by
//     ml/export_weights.py (little-endian float32, byte offsets)
//
// main() builds a RANDOM-weight tiny net at deployment scale (18 -> 16ch
// conv, 16ch ConvGRU, small heads) and times one forward pass.  If given
// argv[1]=manifest.json argv[2]=weights.bin it also exercises the loader.
//
// Correctness over speed: everything is straight loops.  Future work:
// AVX-512 (the judge is an AMD EPYC; zen4 has AVX-512) and/or int8 — see
// ml/csrc/README.md and docs/KEV03.md section 7 for the latency budget.
//
// Build:  g++ -O2 -std=c++17 -o nn_infer nn_infer.cpp
// Run:    ./nn_infer [manifest.json weights.bin]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tensor: shape + flat float storage, NCHW / row-major (PyTorch layout)
// ---------------------------------------------------------------------------

struct Tensor {
    std::vector<int> shape;   // e.g. {C,H,W} or {Out,In} or {N}
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int> s) : shape(std::move(s)) {
        data.assign(numel(), 0.0f);
    }
    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= (size_t)d;
        return n;
    }
};

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void relu_(Tensor& t) {
    for (float& v : t.data) v = v > 0.0f ? v : 0.0f;
}

// conv2d, same padding (pad = k/2), stride 1.
//   in  [C_in, H, W]
//   w   [C_out, C_in, k, k]   (PyTorch Conv2d.weight layout)
//   b   [C_out]               (may be empty -> no bias)
//   out [C_out, H, W]
static void conv2d_same(const Tensor& in, const Tensor& w, const Tensor& b,
                        Tensor& out) {
    const int Cin = in.shape[0], H = in.shape[1], W = in.shape[2];
    const int Cout = w.shape[0], K = w.shape[2], P = K / 2;
    out.shape = {Cout, H, W};
    out.data.assign((size_t)Cout * H * W, 0.0f);
    for (int oc = 0; oc < Cout; ++oc) {
        const float bias = b.data.empty() ? 0.0f : b.data[oc];
        float* op = &out.data[(size_t)oc * H * W];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) op[y * W + x] = bias;
        for (int ic = 0; ic < Cin; ++ic) {
            const float* ip = &in.data[(size_t)ic * H * W];
            const float* wp = &w.data[(((size_t)oc * Cin + ic) * K) * K];
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float wv = wp[ky * K + kx];
                    if (wv == 0.0f) continue;
                    const int dy = ky - P, dx = kx - P;
                    const int y0 = std::max(0, -dy), y1 = std::min(H, H - dy);
                    const int x0 = std::max(0, -dx), x1 = std::min(W, W - dx);
                    for (int y = y0; y < y1; ++y) {
                        const float* irow = ip + (y + dy) * W + dx;
                        float* orow = op + y * W;
                        for (int x = x0; x < x1; ++x)
                            orow[x] += wv * irow[x];
                    }
                }
            }
        }
    }
}

// linear:  out[M] = w[M,N] * in[N] + b[M]      (PyTorch Linear layout)
static void linear(const float* in, int N, const Tensor& w, const Tensor& b,
                   float* out) {
    const int M = w.shape[0];
    for (int m = 0; m < M; ++m) {
        float acc = b.data.empty() ? 0.0f : b.data[m];
        const float* wr = &w.data[(size_t)m * N];
        for (int n = 0; n < N; ++n) acc += wr[n] * in[n];
        out[m] = acc;
    }
}

static void softmax(float* v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, v[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { v[i] = std::exp(v[i] - mx); sum += v[i]; }
    for (int i = 0; i < n; ++i) v[i] /= sum;
}

// Dense GRU cell (PyTorch nn.GRUCell layouts and gate order r,z,n):
//   w_ih [3M, N]  b_ih [3M]  (rows: r, z, n)
//   w_hh [3M, M]  b_hh [3M]
//   h [M] updated in place, x [N]
[[maybe_unused]] static void gru_cell(const float* x, int N, float* h, int M,
                     const Tensor& w_ih, const Tensor& b_ih,
                     const Tensor& w_hh, const Tensor& b_hh) {
    std::vector<float> gi(3 * M), gh(3 * M);
    linear(x, N, w_ih, b_ih, gi.data());
    linear(h, M, w_hh, b_hh, gh.data());
    for (int m = 0; m < M; ++m) {
        const float r = sigmoidf(gi[m] + gh[m]);
        const float z = sigmoidf(gi[M + m] + gh[M + m]);
        const float n = std::tanh(gi[2 * M + m] + r * gh[2 * M + m]);
        h[m] = (1.0f - z) * n + z * h[m];
    }
}

// ConvGRU cell, mirrors ml/model.py ConvGRUCell:
//   conv_ih.weight [3H, C_in, k, k]   conv_hh.weight [3H, H, k, k]
//   x [C_in, 17, 17], h [H, 17, 17] updated in place.  Gate order r, z, n.
static void convgru_cell(const Tensor& x, Tensor& h,
                         const Tensor& w_ih, const Tensor& b_ih,
                         const Tensor& w_hh, const Tensor& b_hh,
                         Tensor& scratch_gi, Tensor& scratch_gh) {
    const int Hc = h.shape[0], H = h.shape[1], W = h.shape[2];
    conv2d_same(x, w_ih, b_ih, scratch_gi);   // [3Hc, H, W]
    conv2d_same(h, w_hh, b_hh, scratch_gh);   // [3Hc, H, W]
    const size_t plane = (size_t)H * W;
    for (int c = 0; c < Hc; ++c) {
        const float* ir = &scratch_gi.data[(size_t)c * plane];
        const float* iz = &scratch_gi.data[((size_t)Hc + c) * plane];
        const float* in_ = &scratch_gi.data[((size_t)2 * Hc + c) * plane];
        const float* hr = &scratch_gh.data[(size_t)c * plane];
        const float* hz = &scratch_gh.data[((size_t)Hc + c) * plane];
        const float* hn = &scratch_gh.data[((size_t)2 * Hc + c) * plane];
        float* hp = &h.data[(size_t)c * plane];
        for (size_t i = 0; i < plane; ++i) {
            const float r = sigmoidf(ir[i] + hr[i]);
            const float z = sigmoidf(iz[i] + hz[i]);
            const float n = std::tanh(in_[i] + r * hn[i]);
            hp[i] = (1.0f - z) * n + z * hp[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Blob + manifest loader (format: ml/export_weights.py, kev03-weights-v1)
// ---------------------------------------------------------------------------
// The manifest is machine-generated JSON with a fixed key order; a tiny
// scanner is enough — no JSON library dependency.  Not a general parser.

struct WeightStore {
    std::vector<float> blob;                    // whole weights.bin
    std::map<std::string, Tensor> tensors;      // name -> copy with shape

    const Tensor& get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::fprintf(stderr, "[nn_infer] missing tensor '%s'\n", name.c_str());
            std::exit(1);
        }
        return it->second;
    }
};

static bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// scan helpers ---------------------------------------------------------------
static size_t find_key(const std::string& s, const std::string& key, size_t from) {
    return s.find("\"" + key + "\"", from);
}

static std::string parse_string_after(const std::string& s, size_t pos) {
    size_t q1 = s.find('"', s.find(':', pos) + 1);
    size_t q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}

static long parse_long_after(const std::string& s, size_t pos) {
    size_t c = s.find(':', pos) + 1;
    return std::strtol(s.c_str() + c, nullptr, 10);
}

static std::vector<int> parse_int_array_after(const std::string& s, size_t pos) {
    std::vector<int> out;
    size_t b0 = s.find('[', pos), b1 = s.find(']', b0);
    std::string body = s.substr(b0 + 1, b1 - b0 - 1);
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos)
            out.push_back(std::atoi(tok.c_str()));
    return out;
}

static bool load_weights(const std::string& manifest_path,
                         const std::string& blob_path, WeightStore& ws) {
    std::string mj, bb;
    if (!read_file(manifest_path, mj) || !read_file(blob_path, bb)) {
        std::fprintf(stderr, "[nn_infer] cannot read %s / %s\n",
                     manifest_path.c_str(), blob_path.c_str());
        return false;
    }
    if (mj.find("kev03-weights-v1") == std::string::npos) {
        std::fprintf(stderr, "[nn_infer] unexpected manifest format\n");
        return false;
    }
    // NOTE: assumes a little-endian host (x86_64 judge machine is).
    ws.blob.resize(bb.size() / sizeof(float));
    std::memcpy(ws.blob.data(), bb.data(), ws.blob.size() * sizeof(float));

    size_t pos = mj.find("\"tensors\"");
    if (pos == std::string::npos) return false;
    while ((pos = find_key(mj, "name", pos)) != std::string::npos) {
        std::string name = parse_string_after(mj, pos);
        size_t sp = find_key(mj, "shape", pos);
        size_t op = find_key(mj, "offset", pos);
        size_t np = find_key(mj, "numel", pos);
        if (sp == std::string::npos || op == std::string::npos ||
            np == std::string::npos) return false;
        Tensor t;
        t.shape = parse_int_array_after(mj, sp);
        if (t.shape.empty()) t.shape = {1};      // scalar params (aux scales)
        long off = parse_long_after(mj, op);
        long numel = parse_long_after(mj, np);
        if (off % 4 != 0 || (size_t)(off / 4 + numel) > ws.blob.size()) {
            std::fprintf(stderr, "[nn_infer] tensor '%s' out of range\n",
                         name.c_str());
            return false;
        }
        t.data.assign(ws.blob.begin() + off / 4,
                      ws.blob.begin() + off / 4 + numel);
        if (t.numel() != (size_t)numel) {
            std::fprintf(stderr, "[nn_infer] tensor '%s' shape/numel mismatch\n",
                         name.c_str());
            return false;
        }
        ws.tensors.emplace(std::move(name), std::move(t));
        pos = np;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Timing harness: random tiny net at DEPLOYMENT scale
// ---------------------------------------------------------------------------
// 18 input planes -> conv3x3 -> 16ch -> ConvGRU(16) -> 1x1 head -> 8ch
// -> global average pool -> linear 8 -> 5 -> softmax.
// This is roughly the belief-side cost of the shipped net; the token/actor
// side is tiny by comparison (a few k MACs).  See docs/KEV03.md sec. 7.

static void fill_random(Tensor& t, std::mt19937& rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    for (float& v : t.data) v = d(rng);
}

int main(int argc, char** argv) {
    if (argc == 3) {
        WeightStore ws;
        if (!load_weights(argv[1], argv[2], ws)) return 1;
        size_t total = 0;
        for (auto& kv : ws.tensors) total += kv.second.numel();
        std::printf("[nn_infer] loaded %zu tensors, %zu floats (%.2f MB) "
                    "from %s\n", ws.tensors.size(), total,
                    total * 4.0 / 1e6, argv[2]);
        // spot-check one known tensor if present
        auto it = ws.tensors.find("encoder.conv1.weight");
        if (it != ws.tensors.end())
            std::printf("[nn_infer] encoder.conv1.weight shape [%d,%d,%d,%d]\n",
                        it->second.shape[0], it->second.shape[1],
                        it->second.shape[2], it->second.shape[3]);
    }

    const int H = 17, W = 17;
    const int C_OBS = 18, C_ENC = 16, C_GRU = 16, C_HEAD = 8, N_OUT = 5;
    std::mt19937 rng(1234);

    Tensor w_enc({C_ENC, C_OBS, 3, 3}), b_enc({C_ENC});
    Tensor w_ih({3 * C_GRU, C_ENC, 3, 3}), b_ih({3 * C_GRU});
    Tensor w_hh({3 * C_GRU, C_GRU, 3, 3}), b_hh({3 * C_GRU});
    Tensor w_head({C_HEAD, C_GRU, 1, 1}), b_head({C_HEAD});
    Tensor w_fc({N_OUT, C_HEAD}), b_fc({N_OUT});
    for (Tensor* t : {&w_enc, &w_ih, &w_hh, &w_head, &w_fc})
        fill_random(*t, rng, 0.08f);

    Tensor obs({C_OBS, H, W});
    fill_random(obs, rng, 1.0f);
    Tensor hidden({C_GRU, H, W});           // persistent belief across rounds
    Tensor enc, head, gi, gh;
    std::vector<float> pooled(C_HEAD), logits(N_OUT);

    // rough MAC count of this belief-side stack, per forward:
    const double macs =
        (double)C_OBS * C_ENC * 9 * H * W +          // encoder conv
        (double)C_ENC * 3 * C_GRU * 9 * H * W +      // conv_ih
        (double)C_GRU * 3 * C_GRU * 9 * H * W +      // conv_hh
        (double)C_GRU * C_HEAD * H * W +             // 1x1 head
        (double)C_HEAD * N_OUT;                      // fc

    auto forward = [&]() {
        conv2d_same(obs, w_enc, b_enc, enc);
        relu_(enc);
        convgru_cell(enc, hidden, w_ih, b_ih, w_hh, b_hh, gi, gh);
        conv2d_same(hidden, w_head, b_head, head);
        relu_(head);
        const size_t plane = (size_t)H * W;
        for (int c = 0; c < C_HEAD; ++c) {
            float acc = 0.0f;
            const float* p = &head.data[(size_t)c * plane];
            for (size_t i = 0; i < plane; ++i) acc += p[i];
            pooled[c] = acc / (float)plane;
        }
        linear(pooled.data(), C_HEAD, w_fc, b_fc, logits.data());
        softmax(logits.data(), N_OUT);
    };

    forward();  // warm-up (also touches all memory once)
    const int iters = 500;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) forward();
    auto t1 = std::chrono::steady_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    float checksum = 0.0f;
    for (int i = 0; i < N_OUT; ++i) checksum += logits[i] * (float)(i + 1);
    std::printf("[nn_infer] tiny net (%d->%dch conv, %dch ConvGRU): "
                "%.1f us / forward, ~%.2fM MACs (%.2f GMAC/s), checksum %.4f\n",
                C_OBS, C_ENC, C_GRU, us, macs / 1e6, macs / us / 1e3, checksum);
    std::printf("[nn_infer] budget: <= ~200 us on the judge EPYC before the "
                "latency race is hurt (see docs/KEV03.md sec. 7)\n");
    return 0;
}
