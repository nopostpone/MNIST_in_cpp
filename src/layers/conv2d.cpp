#include "layers/conv2d.h"
#include <cstring>

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

static void im2col_pad0(const float* __restrict src,
                         int C, int H, int W, int K,
                         int H_out, int W_out,
                         float* __restrict dst) {
    int patch_sz = C * K * K;
    for (int ci = 0; ci < C; ++ci) {
        const float* ch = src + ci * H * W;
        float* dch = dst + ci * K * K;
        for (int oh = 0; oh < H_out; ++oh) {
            int src_row_off = oh * W;
            for (int ow = 0; ow < W_out; ++ow) {
                float* pd = dch + (oh * W_out + ow) * patch_sz;
                int src_off = src_row_off + ow;
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

void Conv2D::col2im_flat(const Eigen::MatrixXf& dcols, float* dsrc) {
    std::memset(dsrc, 0, sizeof(float) * in_c_ * h_ * w_);

    for (int oh = 0; oh < h_out_; ++oh) {
        for (int ow = 0; ow < w_out_; ++ow) {
            int col = oh * w_out_ + ow;
            int row = 0;
            for (int ci = 0; ci < in_c_; ++ci) {
                int ch_off = ci * h_ * w_;
                for (int kh = 0; kh < k_; ++kh) {
                    int hi = oh * s_ + kh - p_;
                    if (hi < 0 || hi >= h_) { row += k_; continue; }
                    for (int kw = 0; kw < k_; ++kw) {
                        int wi = ow * s_ + kw - p_;
                        if (wi < 0 || wi >= w_) { ++row; continue; }
                        dsrc[ch_off + hi * w_ + wi] += dcols(row++, col);
                    }
                }
            }
        }
    }
}

Eigen::MatrixXf Conv2D::forward(const Eigen::MatrixXf& x) {
    // x: (in_c_ * h_ * w_, B)
    int B = static_cast<int>(x.cols());
    int patch_sz = in_c_ * k_ * k_;
    int n_patches = h_out_ * w_out_;

    col_cache_.resize(patch_sz, n_patches * B);

    if (p_ == 0) {
        // Fast path: each column of x is a flat channel-major buffer
        for (int b = 0; b < B; ++b) {
            im2col_pad0(x.col(b).data(), in_c_, h_, w_, k_,
                        h_out_, w_out_,
                        col_cache_.data() + b * n_patches * patch_sz);
        }
    } else {
        // General path: column data is channel-major; use RowMajor Map
        // so x(ci, pos) = data[ci * (h_*w_) + pos]
        for (int b = 0; b < B; ++b) {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                       Eigen::Dynamic, Eigen::RowMajor>>
                x_b(x.col(b).data(), in_c_, h_ * w_);
            col_cache_.middleCols(b * n_patches, n_patches) = im2col(x_b);
        }
    }

    // out is (out_c_, n_patches * B) ColMajor = position-major in memory:
    //   [oc0_p0, oc1_p0, ..., oc_{C-1}_p0, oc0_p1, oc1_p1, ...]
    // But downstream expects channel-major:
    //   [oc0_p0, oc0_p1, ..., oc0_p_{N-1}, oc1_p0, oc1_p1, ...]
    // So we must reorder, NOT just resize.

    Eigen::MatrixXf out(W_.data.rows(), col_cache_.cols());
    out.noalias() = W_.data * col_cache_;

    for (int oc = 0; oc < out_c_; ++oc)
        out.row(oc).array() += b_.data(oc, 0);

    // Reorder: position-major → channel-major
    Eigen::MatrixXf out_cm(out_c_ * n_patches, B);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < n_patches; ++p)
            for (int oc = 0; oc < out_c_; ++oc)
                out_cm(oc * n_patches + p, b) = out(oc, b * n_patches + p);

    return out_cm;
}

Eigen::MatrixXf Conv2D::backward(const Eigen::MatrixXf& dout) {
    // dout: (out_c_ * n_patches, B) in channel-major
    // Convert back to position-major (out_c_, n_patches * B)
    int B = static_cast<int>(dout.cols());
    int n_patches = h_out_ * w_out_;
    int patch_sz = in_c_ * k_ * k_;

    Eigen::MatrixXf dout_pm(out_c_, n_patches * B);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < n_patches; ++p)
            for (int oc = 0; oc < out_c_; ++oc)
                dout_pm(oc, b * n_patches + p) = dout(oc * n_patches + p, b);

    W_.grad.noalias() += dout_pm * col_cache_.transpose();
    b_.grad += dout_pm.rowwise().sum();

    // Gradient w.r.t. im2col output
    Eigen::MatrixXf dcols = W_.data.transpose() * dout_pm;  // (patch_sz, n_patches*B)

    // col2im each sample back into dx columns (channel-major)
    Eigen::MatrixXf dx(in_c_ * h_ * w_, B);

    if (p_ == 0) {
        for (int b = 0; b < B; ++b) {
            col2im_pad0(dcols.data() + b * n_patches * patch_sz,
                        in_c_, h_, w_, k_, h_out_, w_out_,
                        dx.col(b).data());
        }
    } else {
        for (int b = 0; b < B; ++b) {
            col2im_flat(dcols.middleCols(b * n_patches, n_patches),
                        dx.col(b).data());
        }
    }
    return dx;
}

}  // namespace mnist
