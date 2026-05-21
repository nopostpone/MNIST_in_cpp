#include "layers/conv2d.h"

namespace mnist {

Conv2D::Conv2D(int in_channels, int out_channels,
               int kernel_size, int stride, int padding,
               int input_h, int input_w)
    : in_c_(in_channels), out_c_(out_channels),
      h_(input_h), w_(input_w),
      k_(kernel_size), s_(stride), p_(padding)
{
    h_out_ = (h_ - k_ + 2 * p_) / s_ + 1;
    w_out_ = (w_ - k_ + 2 * p_) / s_ + 1;

    int fan_in = in_c_ * k_ * k_;
    W_ = Parameter(out_c_, fan_in);
    b_ = Parameter(out_c_, 1);

    static std::mt19937 rng(42);
    he_init(W_.data, fan_in, rng);
    b_.data.setZero();
}

// ── Fast-path im2col for padding=0 (no bounds checks, unrolled indexing) ─────

static void im2col_pad0(const float* __restrict src,
                         int C, int H, int W, int K,
                         int H_out, int W_out,
                         float* __restrict dst) {
    int patch_sz = C * K * K;
    int n_patches = H_out * W_out;

    for (int ci = 0; ci < C; ++ci) {
        const float* ch = src + ci * H * W;
        float* dch = dst + ci * K * K;   // first channel's portion
        for (int oh = 0; oh < H_out; ++oh) {
            int src_row_off = oh * W;
            for (int ow = 0; ow < W_out; ++ow) {
                int src_off = src_row_off + ow;
                float* pd = dch + (oh * W_out + ow) * patch_sz;
                for (int kh = 0; kh < K; ++kh) {
                    const float* srow = ch + src_off + kh * W;
                    for (int kw = 0; kw < K; ++kw) {
                        pd[kh * K + kw] = srow[kw];
                    }
                }
            }
        }
    }
}

static void col2im_pad0(const float* __restrict dcols,
                         int C, int H, int W, int K,
                         int H_out, int W_out,
                         float* __restrict dsrc) {
    int patch_sz = C * K * K;
    int n_patches = H_out * W_out;

    // Zero output first (accumulate)
    for (int i = 0; i < C * H * W; ++i) dsrc[i] = 0.0f;

    for (int ci = 0; ci < C; ++ci) {
        float* dch = dsrc + ci * H * W;
        const float* dcols_ch = dcols + ci * K * K;
        for (int oh = 0; oh < H_out; ++oh) {
            int dst_row_off = oh * W;
            for (int ow = 0; ow < W_out; ++ow) {
                const float* pc = dcols_ch + (oh * W_out + ow) * patch_sz;
                int dst_off = dst_row_off + ow;
                for (int kh = 0; kh < K; ++kh) {
                    float* drow = dch + dst_off + kh * W;
                    for (int kw = 0; kw < K; ++kw) {
                        drow[kw] += pc[kh * K + kw];
                    }
                }
            }
        }
    }
}

// ── General (slow) im2col for padding != 0 ───────────────────────────────────

Eigen::MatrixXf Conv2D::im2col(const Eigen::MatrixXf& x) {
    int patch_sz = in_c_ * k_ * k_;
    int n_patches = h_out_ * w_out_;
    Eigen::MatrixXf cols = Eigen::MatrixXf::Zero(patch_sz, n_patches);

    for (int oh = 0; oh < h_out_; ++oh) {
        for (int ow = 0; ow < w_out_; ++ow) {
            int col = oh * w_out_ + ow;
            int row = 0;
            for (int ci = 0; ci < in_c_; ++ci) {
                for (int kh = 0; kh < k_; ++kh) {
                    int hi = oh * s_ + kh - p_;
                    if (hi < 0 || hi >= h_) { row += k_; continue; }
                    for (int kw = 0; kw < k_; ++kw) {
                        int wi = ow * s_ + kw - p_;
                        if (wi < 0 || wi >= w_)
                            cols(row++, col) = 0.0f;
                        else
                            cols(row++, col) = x(ci, hi * w_ + wi);
                    }
                }
            }
        }
    }
    return cols;
}

Eigen::MatrixXf Conv2D::col2im(const Eigen::MatrixXf& dcols) {
    Eigen::MatrixXf dx(in_c_, h_ * w_);
    dx.setZero();

    for (int oh = 0; oh < h_out_; ++oh) {
        for (int ow = 0; ow < w_out_; ++ow) {
            int col = oh * w_out_ + ow;
            int row = 0;
            for (int ci = 0; ci < in_c_; ++ci) {
                for (int kh = 0; kh < k_; ++kh) {
                    int hi = oh * s_ + kh - p_;
                    if (hi < 0 || hi >= h_) { row += k_; continue; }
                    for (int kw = 0; kw < k_; ++kw) {
                        int wi = ow * s_ + kw - p_;
                        if (wi < 0 || wi >= w_) { ++row; continue; }
                        dx(ci, hi * w_ + wi) += dcols(row++, col);
                    }
                }
            }
        }
    }
    return dx;
}

// ── Forward / Backward ───────────────────────────────────────────────────────

Eigen::MatrixXf Conv2D::forward(const Eigen::MatrixXf& x) {
    col_cache_ = im2col(x);
    Eigen::MatrixXf out = W_.data * col_cache_;
    for (int oc = 0; oc < out_c_; ++oc)
        out.row(oc).array() += b_.data(oc, 0);
    return out;
}

Eigen::MatrixXf Conv2D::backward(const Eigen::MatrixXf& dout) {
    W_.grad += dout * col_cache_.transpose();
    for (int oc = 0; oc < out_c_; ++oc)
        b_.grad(oc, 0) += dout.row(oc).sum();

    Eigen::MatrixXf dcols = W_.data.transpose() * dout;
    return col2im(dcols);
}

}  // namespace mnist
