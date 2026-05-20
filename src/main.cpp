#include <iostream>
#include <iomanip>
#include <cmath>
#include "data/mnist_loader.h"
#include "layers/flatten.h"
#include "layers/fully_connected.h"
#include "layers/conv2d.h"
#include "layers/max_pool.h"
#include "activations/relu.h"
#include "activations/softmax.h"
#include "loss/cross_entropy.h"

static int g_passed = 0, g_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_passed; } \
    else { ++g_failed; std::cerr << "FAIL: " << msg << "\n"; } \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) \
    CHECK(std::abs((a) - (b)) < (tol), msg)

// ── Numerical gradient check utility ─────────────────────────────────────────

template<typename F>
float num_grad(F&& loss_fn, const Eigen::MatrixXf& w, int ri, int ci,
               float eps = 1e-4f) {
    Eigen::MatrixXf wp = w, wm = w;
    wp(ri, ci) += eps;
    wm(ri, ci) -= eps;
    return (loss_fn(wp) - loss_fn(wm)) / (2.0f * eps);
}

// ── Tests ────────────────────────────────────────────────────────────────────

void test_flatten() {
    std::cout << "--- Flatten ---\n";
    mnist::Flatten flat(2, 2, 3);  // 2 channels, 2x3
    Eigen::MatrixXf x(2, 6);
    x << 1,2,3,4,5,6,
         7,8,9,10,11,12;

    auto out = flat.forward(x);
    CHECK(out.rows() == 12 && out.cols() == 1, "Flatten forward shape");
    CHECK_CLOSE(out(0,0), 1, 1e-6f, "Flatten val[0]");
    CHECK_CLOSE(out(6,0), 7, 1e-6f, "Flatten val[6]");

    auto dx = flat.backward(out);  // grad = out (identity check)
    CHECK(dx.rows() == 2 && dx.cols() == 6, "Flatten backward shape");
    CHECK_CLOSE(dx(0,0), 1, 1e-6f, "Flatten backward roundtrip");
    CHECK_CLOSE(dx(1,5), 12, 1e-6f, "Flatten backward roundtrip 2");

    std::cout << "  OK\n";
}

void test_linear() {
    std::cout << "--- Linear ---\n";
    mnist::Linear lin(4, 3);  // 4→3

    // Check parameter shapes
    auto params = lin.parameters();
    CHECK(params.size() == 2, "Linear has 2 params");
    CHECK(params[0]->data.rows() == 3 && params[0]->data.cols() == 4, "W shape");
    CHECK(params[1]->data.rows() == 3 && params[1]->data.cols() == 1, "b shape");

    // Forward shape
    Eigen::MatrixXf x(4, 1);
    x << 0.5f, -0.3f, 0.8f, -0.1f;
    auto out = lin.forward(x);
    CHECK(out.rows() == 3 && out.cols() == 1, "Linear forward shape");

    // Backward shape
    auto& W = params[0]->data;
    auto& b = params[1]->data;

    // Gradient check on loss = sum(out)
    lin.zero_grad();
    Eigen::MatrixXf ones(3, 1);
    ones.setOnes();
    auto dx = lin.backward(ones);
    CHECK(dx.rows() == 4 && dx.cols() == 1, "Linear backward shape");

    // d(loss)/dW[i,j] where loss = sum(out), out = W*x + b
    // d(loss)/dW[i,j] = x[j] (since d(sum(W*x+b))/dW[i,j] = x[j] summed over i)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            CHECK_CLOSE(params[0]->grad(i, j), x(j, 0), 1e-5f,
                       ("Linear W grad[" + std::to_string(i) + "," + std::to_string(j) + "]").c_str());

    // d(loss)/db[i] = 1
    for (int i = 0; i < 3; ++i)
        CHECK_CLOSE(params[1]->grad(i, 0), 1.0f, 1e-5f, ("Linear b grad[" + std::to_string(i) + "]").c_str());

    // Numerical gradient check with random input
    auto loss_fn = [&](const Eigen::MatrixXf& w) {
        auto& Wref = params[0]->data;
        Eigen::MatrixXf save = Wref;
        const_cast<Eigen::MatrixXf&>(Wref) = w;  // hack for quick check
        auto o = lin.forward(x);
        const_cast<Eigen::MatrixXf&>(Wref) = save;
        return o.sum();
    };
    float num = num_grad(loss_fn, W, 1, 2);
    float ana = params[0]->grad(1, 2);
    CHECK_CLOSE(num, ana, 1e-3f, ("Linear numerical grad W[1,2]: num=" +
              std::to_string(num) + " ana=" + std::to_string(ana)).c_str());

    std::cout << "  OK\n";
}

void test_conv2d_shape() {
    std::cout << "--- Conv2D shape ---\n";
    // Conv1: 1×28×28 → 32×26×26, k=3, s=1, p=0
    mnist::Conv2D conv1(1, 32, 3, 1, 0, 28, 28);
    auto params = conv1.parameters();
    CHECK(params[0]->data.rows() == 32, "Conv1 W out_c");
    CHECK(params[0]->data.cols() == 9,  "Conv1 W fan_in");  // 1*3*3=9

    Eigen::MatrixXf img(1, 784);
    img.setRandom();
    auto out = conv1.forward(img);
    CHECK(out.rows() == 32 && out.cols() == 26*26, "Conv1 forward shape (32, 676)");

    auto dout = Eigen::MatrixXf::Random(32, 676);
    auto dx = conv1.backward(dout);
    CHECK(dx.rows() == 1 && dx.cols() == 784, "Conv1 backward shape (1, 784)");

    // Conv2: 32×13×13 → 64×11×11, k=3, s=1, p=0
    mnist::Conv2D conv2(32, 64, 3, 1, 0, 13, 13);
    auto params2 = conv2.parameters();
    CHECK(params2[0]->data.rows() == 64, "Conv2 W out_c");
    CHECK(params2[0]->data.cols() == 288, "Conv2 W fan_in");  // 32*3*3=288

    Eigen::MatrixXf feat(32, 169);  // 13*13
    feat.setRandom();
    auto out2 = conv2.forward(feat);
    CHECK(out2.rows() == 64 && out2.cols() == 121, "Conv2 forward shape (64, 121)");

    std::cout << "  OK\n";
}

void test_conv2d_gradient() {
    std::cout << "--- Conv2D gradient check ---\n";
    // Small conv for numerical check: 1→1, k=2, 3×3 input → 2×2 output
    mnist::Conv2D conv(1, 1, 2, 1, 0, 3, 3);
    // fan_in = 1*2*2 = 4

    Eigen::MatrixXf x(1, 9);
    x << 1,2,3, 4,5,6, 7,8,9;  // 3×3 image, channel 0

    auto out = conv.forward(x);
    CHECK(out.rows() == 1 && out.cols() == 4, "Conv grad: forward shape");

    // sum loss → dout = ones
    conv.zero_grad();
    Eigen::MatrixXf dout = Eigen::MatrixXf::Ones(1, 4);
    conv.backward(dout);  // analytical grads stored in W_.grad, b_.grad

    auto* pW = conv.parameters()[0];  // &W_
    const auto W_orig = pW->data;     // snapshot

    for (int j = 0; j < 4; ++j) {
        // +eps
        pW->data = W_orig;
        pW->data(0, j) += 1e-4f;
        float lp = conv.forward(x).sum();
        // -eps
        pW->data = W_orig;
        pW->data(0, j) -= 1e-4f;
        float lm = conv.forward(x).sum();

        float num = (lp - lm) / 2e-4f;
        float ana = pW->grad(0, j);
        // Relative tolerance: gradients are ~10-30 so abs tol scales up
        float rel_tol = 1e-3f * std::max(1.0f, std::abs(ana));
        CHECK_CLOSE(num, ana, rel_tol,
                   ("Conv W grad[0," + std::to_string(j) + "]").c_str());
    }
    pW->data = W_orig;  // restore
    std::cout << "  OK\n";
}

void test_maxpool() {
    std::cout << "--- MaxPool2D ---\n";
    mnist::MaxPool2D pool(2, 4, 4, 2, 2);  // 2ch, 4x4, pool=2, stride=2 → 2×2 out

    // Channel 0: 0..15; Channel 1: 16..31
    Eigen::MatrixXf x(2, 16);
    for (int i = 0; i < 16; ++i) {
        x(0, i) = static_cast<float>(i);
        x(1, i) = static_cast<float>(i + 16);
    }

    auto out = pool.forward(x);
    CHECK(out.rows() == 2 && out.cols() == 4, "Pool forward shape (2,4)");

    // Manual check: 4×4 ch0 row-major:
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    // 2×2 windows → maxes: [5, 7, 13, 15]
    CHECK_CLOSE(out(0, 0), 5.0f,  1e-5f, "Pool[0,0]");
    CHECK_CLOSE(out(0, 1), 7.0f,  1e-5f, "Pool[0,1]");
    CHECK_CLOSE(out(0, 2), 13.0f, 1e-5f, "Pool[0,2]");
    CHECK_CLOSE(out(0, 3), 15.0f, 1e-5f, "Pool[0,3]");

    // Backward: gradient only flows to max positions
    Eigen::MatrixXf dout(2, 4);
    dout.setOnes();
    auto dx = pool.backward(dout);

    float sum_dx = dx.sum();
    CHECK_CLOSE(sum_dx, 8.0f, 1e-5f, "Pool backward sum");  // 2ch × 4 positions

    // Only 4 positions per channel have gradient
    int nz = 0;
    for (int i = 0; i < dx.cols(); ++i)
        if (dx(0, i) > 0.5f) ++nz;
    CHECK(nz == 4, "Pool backward sparsity");

    std::cout << "  OK\n";
}

void test_relu() {
    std::cout << "--- ReLU ---\n";
    mnist::ReLU relu;
    Eigen::MatrixXf x(3, 1);
    x << -1.0f, 0.0f, 2.0f;

    auto out = relu.forward(x);
    CHECK_CLOSE(out(0,0), 0.0f, 1e-6f, "ReLU[-1]=0");
    CHECK_CLOSE(out(1,0), 0.0f, 1e-6f, "ReLU[0]=0");
    CHECK_CLOSE(out(2,0), 2.0f, 1e-6f, "ReLU[2]=2");

    Eigen::MatrixXf grad(3, 1);
    grad << 1.0f, 2.0f, 3.0f;
    auto dx = relu.backward(grad);
    CHECK_CLOSE(dx(0,0), 0.0f, 1e-6f, "ReLU grad[-1]");
    CHECK_CLOSE(dx(1,0), 0.0f, 1e-6f, "ReLU grad[0]");
    CHECK_CLOSE(dx(2,0), 3.0f, 1e-6f, "ReLU grad[2]");

    std::cout << "  OK\n";
}

void test_softmax_ce() {
    std::cout << "--- Softmax + CrossEntropy ---\n";
    mnist::Softmax sm;
    mnist::CrossEntropyLoss ce;

    // Logits for 3 classes
    Eigen::MatrixXf logits(3, 1);
    logits << 2.0f, 1.0f, 0.1f;

    auto probs = sm.forward(logits);
    float s = probs.sum();
    CHECK_CLOSE(s, 1.0f, 1e-6f, "Softmax sums to 1");
    CHECK(probs(0, 0) > probs(1, 0) && probs(1, 0) > probs(2, 0), "Softmax order");

    // Cross entropy loss with correct label = 0
    float loss = ce.forward(probs, 0);
    CHECK(loss > 0, "CE loss positive");

    // Combined gradient: dL/d(logits) = probs - one_hot
    auto dlogits = ce.backward(probs, 0);
    CHECK_CLOSE(dlogits.sum(), 0.0f, 1e-6f, "CE grad sums to 0");
    CHECK(dlogits(0, 0) < 0, "CE grad for correct class is negative");
    CHECK(dlogits(1, 0) > 0, "CE grad for wrong class is positive");

    // Numerical check: small perturbation on logits
    auto loss_fn = [&](float eps, int idx) {
        Eigen::MatrixXf p = logits;
        p(idx, 0) += eps;
        auto pr = sm.forward(p);
        return ce.forward(pr, 0);
    };

    float eps = 5e-4f;  // must be large enough for float32 softmax precision
    for (int i = 0; i < 3; ++i) {
        float num = (loss_fn(eps, i) - loss_fn(-eps, i)) / (2.0f * eps);
        float ana = dlogits(i, 0);
        CHECK_CLOSE(num, ana, 1e-3f,
                   ("Softmax+CE numerical check[" + std::to_string(i) + "]").c_str());
    }

    std::cout << "  OK\n";
}

void test_end_to_end_shape() {
    std::cout << "--- End-to-end shape check ---\n";
    // Build full network and test forward shapes
    mnist::Conv2D   conv1(1,  32, 3, 1, 0, 28, 28);  // → 32×26×26
    mnist::ReLU     relu1;
    mnist::MaxPool2D pool1(32, 26, 26, 2, 2);         // → 32×13×13
    mnist::Conv2D   conv2(32, 64, 3, 1, 0, 13, 13);  // → 64×11×11
    mnist::ReLU     relu2;
    mnist::MaxPool2D pool2(64, 11, 11, 2, 2);         // → 64×5×5
    mnist::Flatten   flat(64, 5, 5);                   // → 1600
    mnist::Linear    fc1(1600, 128);                   // → 128
    mnist::ReLU     relu3;
    mnist::Linear    fc2(128, 10);                     // → 10
    mnist::Softmax   softmax;

    // Load one real image
    auto train = mnist::Loader::load_training("data", true);
    auto img = train.images.row(0);  // (1, 784) as Eigen row vector

    // Convert to (1, 784) matrix
    Eigen::MatrixXf x(1, 784);
    x = img;

    auto o1  = conv1.forward(x);     CHECK(o1.rows() == 32 && o1.cols() == 676,  "e2e: conv1");
    auto o2  = relu1.forward(o1);
    auto o3  = pool1.forward(o2);    CHECK(o3.rows() == 32 && o3.cols() == 169,  "e2e: pool1");
    auto o4  = conv2.forward(o3);    CHECK(o4.rows() == 64 && o4.cols() == 121,  "e2e: conv2");
    auto o5  = relu2.forward(o4);
    auto o6  = pool2.forward(o5);    CHECK(o6.rows() == 64 && o6.cols() == 25,   "e2e: pool2");
    auto o7  = flat.forward(o6);     CHECK(o7.rows() == 1600 && o7.cols() == 1,  "e2e: flatten");
    auto o8  = fc1.forward(o7);      CHECK(o8.rows() == 128 && o8.cols() == 1,   "e2e: fc1");
    auto o9  = relu3.forward(o8);
    auto o10 = fc2.forward(o9);      CHECK(o10.rows() == 10 && o10.cols() == 1,  "e2e: fc2");
    auto o11 = softmax.forward(o10); CHECK(o11.rows() == 10 && o11.cols() == 1,  "e2e: softmax");
    CHECK_CLOSE(o11.sum(), 1.0f, 1e-5f, "e2e: probabilities sum to 1");

    // Prediction check
    Eigen::Index pred;
    o11.col(0).maxCoeff(&pred);
    std::cout << "  Label: " << train.labels(0) << "  Pred: " << pred
              << "  (untrained, random guess)\n";

    std::cout << "  OK\n";
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    test_flatten();
    test_linear();
    test_conv2d_shape();
    test_conv2d_gradient();
    test_maxpool();
    test_relu();
    test_softmax_ce();
    test_end_to_end_shape();

    std::cout << "\n===== Results: " << g_passed << " passed, "
              << g_failed << " failed =====\n";

    return g_failed > 0 ? 1 : 0;
}
