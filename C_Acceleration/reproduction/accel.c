/*
 * accel.c -- portable C99 acceleration kernels for the numpy-only
 * inference path used by SynthesizeHandwriting.py's SynthesizeJointBestOf().
 *
 * Portability constraints (see project README for why): this file must
 * compile unmodified with a plain `gcc -O3` on both an x86_64 Windows
 * laptop (MinGW-w64) and an ARM64 Linux single-board computer (ODROID
 * N2+). It therefore uses ONLY <stdint.h>, <stdlib.h>, <string.h>,
 * <math.h> -- no platform-specific headers, no x86 SIMD intrinsics
 * (SSE/AVX), no ARM NEON intrinsics. Just standard C99 loops; -O3 lets
 * gcc auto-vectorize for whatever target it's built on.
 *
 * Every function here is a pure, allocation-free (caller supplies all
 * buffers) re-implementation of the matching function in
 * np_inference/layers.py. Nothing in this directory is imported by any
 * existing production file -- see accel_py.py / README.md for how a
 * caller could opt in.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#define ACCEL_EXPORT __declspec(dllexport)
#else
#define ACCEL_EXPORT
#endif

/*
 * conv2d_forward
 * ---------------
 * Matches np_inference.layers.conv2d(x, weight, bias, stride, padding):
 *   x:      (B, C, H, W)            flat, row-major, length B*C*H*W
 *   weight: (OutC, InC, kh, kw)     flat, row-major, length OutC*InC*kh*kw
 *   bias:   (OutC,) or NULL
 *   out:    (B, OutC, outH, outW)   flat, row-major, caller-allocated
 *
 * outH = (H + 2*padding - kh) / stride + 1   (integer division, floor)
 * outW = (W + 2*padding - kw) / stride + 1
 *
 * Zero-padding, same semantics as np.pad with mode "constant" (fill 0).
 * Direct nested-loop convolution (no im2col needed in C -- see README):
 * out-channel x out-row x out-col x in-channel x kernel-row x kernel-col,
 * with the innermost loops using plain pointer/array indexing.
 */
ACCEL_EXPORT void conv2d_forward(
    const double *restrict x, int B, int C, int H, int W,
    const double *restrict weight, int OutC, int InC, int kh, int kw,
    const double *restrict bias,
    int stride, int padding,
    double *restrict out, int outH, int outW)
{
    /* InC must equal C (same contract as the Python version). */
    const int x_b_stride = C * H * W;
    const int x_c_stride = H * W;
    const int w_oc_stride = InC * kh * kw;
    const int w_ic_stride = kh * kw;
    const int out_b_stride = OutC * outH * outW;
    const int out_oc_stride = outH * outW;

    /* Weight-stationary accumulation: for each (oc, ic, kernel-row,
     * kernel-col) we add wval * (a shifted row of x) into a row of out.
     * The valid output-column range for a given kernel-col is computed
     * up front so the innermost loop has NO bounds check at all -- just
     * `out_row[ow] += wval * xrow[ow*stride + shift]`, which -O3 can
     * auto-vectorize on both x86_64 (SSE2 baseline, 2 doubles/op) and
     * ARM64 (NEON is mandatory in the base ISA, also 2 doubles/op) with
     * no special flags and no intrinsics. This matters because a naive
     * accumulate-per-output-pixel loop with a branch in the kw loop
     * measured far slower than numpy's BLAS-backed im2col+matmul; this
     * form is the portable-C way to get close to it. */
    for (int b = 0; b < B; b++) {
        const double *xb = x + (size_t)b * x_b_stride;
        double *outb = out + (size_t)b * out_b_stride;

        for (int oc = 0; oc < OutC; oc++) {
            double bval = (bias != NULL) ? bias[oc] : 0.0;
            double *out_plane = outb + (size_t)oc * out_oc_stride;

            /* Initialize with bias (or zero). */
            for (int i = 0; i < outH * outW; i++) out_plane[i] = bval;

            const double *wo = weight + (size_t)oc * w_oc_stride;

            for (int ic = 0; ic < InC; ic++) {
                const double *xc = xb + (size_t)ic * x_c_stride;
                const double *wc = wo + (size_t)ic * w_ic_stride;

                for (int r = 0; r < kh; r++) {
                    const double *wrow = wc + (size_t)r * kw;

                    for (int cidx = 0; cidx < kw; cidx++) {
                        double wval = wrow[cidx];

                        /* Valid oh range: 0 <= oh*stride - padding + r < H */
                        int num_lo_h = padding - r;
                        int oh_lo = (num_lo_h <= 0) ? 0 : (num_lo_h + stride - 1) / stride;
                        int num_hi_h = H - 1 + padding - r;
                        int oh_hi = (num_hi_h < 0) ? -1 : num_hi_h / stride;
                        if (oh_hi >= outH) oh_hi = outH - 1;
                        if (oh_lo > oh_hi) continue;

                        /* Valid ow range: 0 <= ow*stride - padding + cidx < W */
                        int num_lo_w = padding - cidx;
                        int ow_lo = (num_lo_w <= 0) ? 0 : (num_lo_w + stride - 1) / stride;
                        int num_hi_w = W - 1 + padding - cidx;
                        int ow_hi = (num_hi_w < 0) ? -1 : num_hi_w / stride;
                        if (ow_hi >= outW) ow_hi = outW - 1;
                        if (ow_lo > ow_hi) continue;

                        for (int oh = oh_lo; oh <= oh_hi; oh++) {
                            int ih = oh * stride - padding + r;
                            const double *xrow = xc + (size_t)ih * W;
                            double *out_row = out_plane + (size_t)oh * outW;

                            if (stride == 1) {
                                int shift = cidx - padding; /* iw = ow + shift */
                                for (int ow = ow_lo; ow <= ow_hi; ow++) {
                                    out_row[ow] += wval * xrow[ow + shift];
                                }
                            } else {
                                for (int ow = ow_lo; ow <= ow_hi; ow++) {
                                    int iw = ow * stride - padding + cidx;
                                    out_row[ow] += wval * xrow[iw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * lstm_forward
 * ------------
 * Matches np_inference.layers.lstm_forward(x, weight_ih, weight_hh,
 * bias_ih, bias_hh, hidden_size) for ONE direction, ONE layer.
 *
 *   x:         (T, B, input_size)       flat, row-major
 *   weight_ih: (4*H, input_size)        flat, row-major (PyTorch layout)
 *   weight_hh: (4*H, H)                 flat, row-major
 *   bias_ih:   (4*H,)
 *   bias_hh:   (4*H,)
 *   out:       (T, B, H)                flat, row-major, caller-allocated
 *   h_out, c_out: (B, H) final hidden/cell state, caller-allocated
 *
 * Gate row-blocks within the 4*H rows are ordered [input, forget,
 * cell/g, output] -- PyTorch's convention, confirmed against layers.py.
 * This is inherently sequential over T (each timestep depends on the
 * previous h/c), which is exactly why it's slow in pure Python/numpy
 * and a reasonable C target -- but it must be bit-exact with the
 * reference equations, so it's kept straightforward (no attempt at
 * clever blocking) to minimize the chance of a transcription bug.
 */
ACCEL_EXPORT void lstm_forward(
    const double *x, int T, int B, int input_size,
    const double *weight_ih, const double *weight_hh,
    const double *bias_ih, const double *bias_hh,
    int hidden_size,
    double *out, double *h_out, double *c_out)
{
    const int H = hidden_size;
    const int G = 4 * H;

    double *h = (double *)calloc((size_t)B * H, sizeof(double));
    double *c = (double *)calloc((size_t)B * H, sizeof(double));
    double *gates = (double *)malloc((size_t)B * G * sizeof(double));
    if (h == NULL || c == NULL || gates == NULL) {
        free(h); free(c); free(gates);
        return; /* caller-visible failure: out buffers left untouched */
    }

    for (int t = 0; t < T; t++) {
        const double *xt = x + (size_t)t * B * input_size;

        for (int b = 0; b < B; b++) {
            const double *xb = xt + (size_t)b * input_size;
            const double *hb = h + (size_t)b * H;
            double *gb = gates + (size_t)b * G;

            for (int g = 0; g < G; g++) {
                double acc = bias_ih[g] + bias_hh[g];
                const double *wih_row = weight_ih + (size_t)g * input_size;
                for (int k = 0; k < input_size; k++) {
                    acc += xb[k] * wih_row[k];
                }
                const double *whh_row = weight_hh + (size_t)g * H;
                for (int k = 0; k < H; k++) {
                    acc += hb[k] * whh_row[k];
                }
                gb[g] = acc;
            }
        }

        for (int b = 0; b < B; b++) {
            double *gb = gates + (size_t)b * G;
            double *hb = h + (size_t)b * H;
            double *cb = c + (size_t)b * H;
            double *outb = out + ((size_t)t * B + b) * H;

            for (int j = 0; j < H; j++) {
                double i_gate = 1.0 / (1.0 + exp(-gb[0 * H + j]));
                double f_gate = 1.0 / (1.0 + exp(-gb[1 * H + j]));
                double g_gate = tanh(gb[2 * H + j]);
                double o_gate = 1.0 / (1.0 + exp(-gb[3 * H + j]));

                double new_c = f_gate * cb[j] + i_gate * g_gate;
                double new_h = o_gate * tanh(new_c);

                cb[j] = new_c;
                hb[j] = new_h;
                outb[j] = new_h;
            }
        }
    }

    memcpy(h_out, h, (size_t)B * H * sizeof(double));
    memcpy(c_out, c, (size_t)B * H * sizeof(double));

    free(h);
    free(c);
    free(gates);
}
