"""
MNIST digit recognition web server.
Run:  python server.py
Requires: pip install flask
"""
import subprocess
import os
import sys
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CPP_BIN = os.path.join(PROJECT_ROOT, "build", "mnist_cpp.exe")
WEIGHTS = os.path.join(PROJECT_ROOT, "model_weights.bin")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    pixels = data.get("pixels", [])

    if len(pixels) != 784:
        return jsonify({"error": f"Expected 784 pixels, got {len(pixels)}"}), 400

    # Convert 0-255 grayscale values to raw bytes
    raw_bytes = bytes(int(max(0, min(255, p))) for p in pixels)

    try:
        proc = subprocess.run(
            [CPP_BIN, "predict"],
            input=raw_bytes,
            capture_output=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        output = proc.stdout.decode().strip().split()
        digit = int(output[0])
        probs = [float(x) for x in output[1:]] if len(output) > 1 else [1.0]
        return jsonify({"digit": digit, "probs": probs})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not os.path.exists(CPP_BIN):
        print(f"ERROR: C++ binary not found at {CPP_BIN}")
        print("Build the project first: cd build && mingw32-make")
        sys.exit(1)
    if not os.path.exists(WEIGHTS):
        print(f"WARNING: model weights not found at {WEIGHTS}")
        print("Run training first: ./build/mnist_cpp.exe train")
    print(f"C++ binary: {CPP_BIN}")
    print(f"Weights:    {WEIGHTS}")
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
