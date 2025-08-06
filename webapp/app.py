from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'finetuning-classifying')))
import importlib.util

 # Dynamically import spam_classifier from the parent directory
SPAM_CLASSIFIER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spam_classifier.py'))
spec = importlib.util.spec_from_file_location("spam_classifier", SPAM_CLASSIFIER_PATH)
spam_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spam_classifier)
classify_text = spam_classifier.classify_text
import tiktoken

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spam_classifier.pth'))

def load_model_and_tokenizer():
    # Import model and tokenizer setup from your code
    GPTModel = spam_classifier.GPTModel
    model_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }
    model = GPTModel(model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer

app = Flask(__name__)
model, tokenizer = load_model_and_tokenizer()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")

@app.route("/statistics")
def statistics():
    return render_template("statistics.html")

@app.route("/classify", methods=["GET", "POST"])
def classify():
    label = None
    confidence = None
    message = None
    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            max_length = 128
            label, confidence = classify_text(message, model, tokenizer, torch.device('cpu'), max_length)
            confidence = int(confidence * 100) if confidence is not None else None
    return render_template("classify.html", label=label, confidence=confidence, message=message)

@app.route("/contact")
def contact():
    return render_template("spa_contact.html")

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "active_classifiers": 500000,
        "messages_processed": 10000000,
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.96
    })

@app.route("/api/model")
def api_model():
    return jsonify({
        "model": "GPT-2 124M parameters",
        "vocab_size": 50257,
        "embedding_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "context_length": 1024,
        "activation": "GELU",
        "layer_norm": "Pre-norm",
        "classification_head": "2-class output (spam/not spam)"
    })

@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json()
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "No message provided."}), 400
    max_length = 128
    label, confidence = classify_text(message, model, tokenizer, torch.device('cpu'), max_length)
    return jsonify({"label": label, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
