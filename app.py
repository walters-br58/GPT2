"""
EECE 4520 - Milestone 5
Flask API server exposing the GPT-2-like model as a chatbot endpoint.

Endpoints:
  GET  /              - Serves the chat UI (index.html)
  POST /generate      - Generate a response given a prompt
  GET  /health        - Health check

Usage:
  python app.py
"""

import os
import time
import logging
import traceback

import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model_factory import ModelFactory
from tokenizer_singleton import TokenizerSingleton
from decoding_strategy import GreedyStrategy, TopKStrategy, NucleusStrategy

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)  # allow requests from any origin (needed for local dev)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading — done once at startup
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
MODEL_PATH      = os.getenv("MODEL_PATH",     "gpt2_final.pt")
TOKENIZER_PATH  = os.getenv("TOKENIZER_PATH", "bpe_tokenizer.json")

logger.info(f"Loading model from '{MODEL_PATH}' on device '{DEVICE}'...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    factory    = ModelFactory()
    model      = factory.create_model("standard").to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

logger.info(f"Loading tokenizer from '{TOKENIZER_PATH}'...")
try:
    tokenizer = TokenizerSingleton.get_instance(TOKENIZER_PATH)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    tokenizer = None

# ─────────────────────────────────────────────────────────────────────────────
# Strategy map
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "greedy":  lambda _k, _p: GreedyStrategy(),
    "top_k":   lambda k,  _p: TopKStrategy(k=k),
    "nucleus": lambda _k,  p: NucleusStrategy(p=p),
}

# ─────────────────────────────────────────────────────────────────────────────
# Generation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    prompt:         str,
    strategy_name:  str   = "top_k",
    max_new_tokens: int   = 100,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = 0.9,
) -> str:
    """
    Autoregressively generate tokens from a prompt using the chosen
    DecodingStrategy.  Returns the full decoded string.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not loaded.")

    strategy_name = strategy_name.lower()
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )

    strategy = STRATEGIES[strategy_name](top_k, top_p)

    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / max(temperature, 1e-8)   # (1, V)
        next_id   = strategy.select_token(logits)               # (1, 1)
        idx       = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    return tokenizer.decode(idx[0].tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return send_from_directory("static", "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check — returns model/tokenizer load status."""
    return jsonify({
        "status":    "ok",
        "model":     model is not None,
        "tokenizer": tokenizer is not None,
        "device":    DEVICE,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate a continuation for the given prompt.

    Request JSON:
      {
        "prompt":         str,              # required
        "strategy":       str,              # optional: greedy | top_k | nucleus (default: top_k)
        "max_new_tokens": int,              # optional (default: 100)
        "temperature":    float,            # optional (default: 0.8)
        "top_k":          int,              # optional (default: 50)
        "top_p":          float             # optional (default: 0.9)
      }

    Response JSON:
      {
        "prompt":    str,
        "response":  str,
        "strategy":  str,
        "latency_ms": float
      }
    """
    # ── Parse request ────────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Field 'prompt' is required and cannot be empty."}), 400

    if len(prompt) > 500:
        return jsonify({"error": "Prompt exceeds 500 character limit."}), 400

    strategy       = data.get("strategy",       "top_k")
    max_new_tokens = int(data.get("max_new_tokens", 100))
    temperature    = float(data.get("temperature",   0.8))
    top_k          = int(data.get("top_k",           50))
    top_p          = float(data.get("top_p",         0.9))

    # ── Validate parameters ──────────────────────────────────────────────────
    if max_new_tokens < 1 or max_new_tokens > 300:
        return jsonify({"error": "max_new_tokens must be between 1 and 300."}), 400
    if temperature <= 0:
        return jsonify({"error": "temperature must be > 0."}), 400
    if top_k < 1:
        return jsonify({"error": "top_k must be >= 1."}), 400
    if not (0 < top_p <= 1.0):
        return jsonify({"error": "top_p must be in (0, 1]."}), 400

    # ── Generate ─────────────────────────────────────────────────────────────
    logger.info(
        f"Request — strategy={strategy} max_new_tokens={max_new_tokens} "
        f"temp={temperature} prompt='{prompt[:60]}...'"
    )

    start = time.time()
    try:
        response = generate_response(
            prompt         = prompt,
            strategy_name  = strategy,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": "Model not available."}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error."}), 500

    latency_ms = round((time.time() - start) * 1000, 1)
    logger.info(f"Response generated in {latency_ms}ms.")

    return jsonify({
        "prompt":     prompt,
        "response":   response,
        "strategy":   strategy,
        "latency_ms": latency_ms,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
