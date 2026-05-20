#include "data/mnist_loader.h"

#include <fstream>
#include <stdexcept>

namespace mnist {

namespace {

uint32_t read_big_endian_u32(std::ifstream& f) {
    unsigned char buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    if (!f)
        throw std::runtime_error("Unexpected EOF in MNIST file");
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8)  |
           (static_cast<uint32_t>(buf[3]));
}

}  // namespace

Dataset Loader::load_training(const std::string& data_dir, bool normalize) {
    return load(data_dir + "/train-images-idx3-ubyte",
                data_dir + "/train-labels-idx1-ubyte",
                normalize);
}

Dataset Loader::load_test(const std::string& data_dir, bool normalize) {
    return load(data_dir + "/t10k-images-idx3-ubyte",
                data_dir + "/t10k-labels-idx1-ubyte",
                normalize);
}

Dataset Loader::load(const std::string& image_path,
                     const std::string& label_path,
                     bool normalize) {
    // ── Open files ──────────────────────────────────────────────────────────
    std::ifstream img_f(image_path, std::ios::binary);
    if (!img_f)
        throw std::runtime_error("Cannot open: " + image_path);

    std::ifstream lbl_f(label_path, std::ios::binary);
    if (!lbl_f)
        throw std::runtime_error("Cannot open: " + label_path);

    // ── Parse image header ─────────────────────────────────────────────────
    uint32_t img_magic = read_big_endian_u32(img_f);
    if (img_magic != 2051)
        throw std::runtime_error("Bad image magic: " + std::to_string(img_magic));

    uint32_t n_images = read_big_endian_u32(img_f);
    uint32_t n_rows   = read_big_endian_u32(img_f);
    uint32_t n_cols   = read_big_endian_u32(img_f);

    if (n_rows != 28 || n_cols != 28)
        throw std::runtime_error("Expected 28x28 images, got " +
                                 std::to_string(n_rows) + "x" + std::to_string(n_cols));

    // ── Parse label header ─────────────────────────────────────────────────
    uint32_t lbl_magic = read_big_endian_u32(lbl_f);
    if (lbl_magic != 2049)
        throw std::runtime_error("Bad label magic: " + std::to_string(lbl_magic));

    uint32_t n_labels = read_big_endian_u32(lbl_f);

    if (n_images != n_labels)
        throw std::runtime_error("Image/label count mismatch: " +
                                 std::to_string(n_images) + " vs " + std::to_string(n_labels));

    // ── Read pixel data ────────────────────────────────────────────────────
    constexpr int kPixels = 28 * 28;
    Dataset ds;
    ds.images.resize(n_images, kPixels);
    ds.labels.resize(n_images);

    std::vector<unsigned char> buf(static_cast<size_t>(n_images) * kPixels);
    img_f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if (!img_f)
        throw std::runtime_error("Truncated image data");

    const float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    for (size_t i = 0; i < static_cast<size_t>(n_images); ++i) {
        for (int j = 0; j < kPixels; ++j) {
            ds.images(static_cast<Eigen::Index>(i), j) =
                static_cast<float>(buf[i * kPixels + j]) * scale;
        }
    }

    // ── Read label data ────────────────────────────────────────────────────
    std::vector<unsigned char> lbl_buf(n_images);
    lbl_f.read(reinterpret_cast<char*>(lbl_buf.data()), n_images);
    if (!lbl_f)
        throw std::runtime_error("Truncated label data");

    for (size_t i = 0; i < n_images; ++i)
        ds.labels(static_cast<Eigen::Index>(i)) = static_cast<int>(lbl_buf[i]);

    return ds;
}

}  // namespace mnist
