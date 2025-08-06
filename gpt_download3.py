"""
GPT-2 Weight Downloader and Loader
Downloads pretrained GPT-2 weights from OpenAI's repository and loads them into Python
"""

import os
import json
import numpy as np
import tensorflow as tf
import tqdm
import urllib.request
import ssl
import requests

def download_file(url, filepath):
    """Download a file with progress bar using requests"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Use requests for better SSL handling
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filepath, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

def download_gpt2_files(model_size, models_dir):
    """Download GPT-2 model files"""
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    model_dir = os.path.join(models_dir, model_size)
    
    files = [
        "checkpoint",
        "encoder.json", 
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    ]
    
    for filename in files:
        url = f"{base_url}/{model_size}/{filename}"
        filepath = os.path.join(model_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File already exists and is up-to-date: {filepath}")
        else:
            print(f"Downloading {filename}...")
            download_file(url, filepath)

def load_gpt2_weights(model_size, models_dir):
    """Load GPT-2 weights from TensorFlow checkpoint"""
    model_dir = os.path.join(models_dir, model_size)
    
    # Load hyperparameters
    with open(os.path.join(model_dir, "hparams.json"), "r") as f:
        hparams = json.load(f)
    
    # Load vocabulary
    with open(os.path.join(model_dir, "encoder.json"), "r") as f:
        encoder = json.load(f)
    
    # Load BPE merges
    with open(os.path.join(model_dir, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()
    
    # Load TensorFlow checkpoint
    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    reader = tf.train.load_checkpoint(checkpoint_path)
    
    # Get all variable names
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    # Extract weights
    params = {}
    
    # Token embeddings
    params["wte"] = reader.get_tensor("model/wte")
    
    # Position embeddings  
    params["wpe"] = reader.get_tensor("model/wpe")
    
    # Final layer norm
    params["g"] = reader.get_tensor("model/ln_f/g")
    params["b"] = reader.get_tensor("model/ln_f/b")
    
    # Transformer blocks
    params["blocks"] = []
    n_layer = hparams["n_layer"]
    
    for layer in range(n_layer):
        block = {}
        
        # Attention weights
        attn = {}
        
        # Combined QKV weights and biases
        c_attn_w = reader.get_tensor(f"model/h{layer}/attn/c_attn/w")
        c_attn_b = reader.get_tensor(f"model/h{layer}/attn/c_attn/b")
        attn["c_attn"] = {"w": c_attn_w, "b": c_attn_b}
        
        # Output projection
        c_proj_w = reader.get_tensor(f"model/h{layer}/attn/c_proj/w")
        c_proj_b = reader.get_tensor(f"model/h{layer}/attn/c_proj/b")
        attn["c_proj"] = {"w": c_proj_w, "b": c_proj_b}
        
        block["attn"] = attn
        
        # MLP weights
        mlp = {}
        
        # First linear layer
        c_fc_w = reader.get_tensor(f"model/h{layer}/mlp/c_fc/w")
        c_fc_b = reader.get_tensor(f"model/h{layer}/mlp/c_fc/b")
        mlp["c_fc"] = {"w": c_fc_w, "b": c_fc_b}
        
        # Second linear layer
        c_proj_w = reader.get_tensor(f"model/h{layer}/mlp/c_proj/w")
        c_proj_b = reader.get_tensor(f"model/h{layer}/mlp/c_proj/b")
        mlp["c_proj"] = {"w": c_proj_w, "b": c_proj_b}
        
        block["mlp"] = mlp
        
        # Layer norms
        ln_1_g = reader.get_tensor(f"model/h{layer}/ln_1/g")
        ln_1_b = reader.get_tensor(f"model/h{layer}/ln_1/b")
        block["ln_1"] = {"g": ln_1_g, "b": ln_1_b}
        
        ln_2_g = reader.get_tensor(f"model/h{layer}/ln_2/g")
        ln_2_b = reader.get_tensor(f"model/h{layer}/ln_2/b")
        block["ln_2"] = {"g": ln_2_g, "b": ln_2_b}
        
        params["blocks"].append(block)
    
    return hparams, params

def download_and_load_gpt2(model_size="124M", models_dir="gpt2"):
    """
    Download and load GPT-2 weights from OpenAI's repository
    
    Args:
        model_size: Size of the model ("124M", "355M", "774M", "1558M")
        models_dir: Directory to store downloaded models
    
    Returns:
        settings: Model hyperparameters
        params: Model weights as numpy arrays
    """
    print(f"Downloading GPT-2 {model_size} model...")
    
    # Download files
    download_gpt2_files(model_size, models_dir)
    
    print(f"Loading GPT-2 {model_size} weights...")
    
    # Load weights
    settings, params = load_gpt2_weights(model_size, models_dir)
    
    print(f"Successfully loaded GPT-2 {model_size} model!")
    print(f"Model parameters: {sum(p.size for p in params.values() if isinstance(p, np.ndarray)):,}")
    
    return settings, params 