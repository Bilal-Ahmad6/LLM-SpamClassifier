"""
Spam Classifier using GPT-2 124M Fine-tuning
A clean implementation for SMS spam detection with interactive prediction
Optimized for laptops with 4GB VRAM (RTX 3050 Ti, etc.)
"""

import urllib.request
import ssl
import zipfile
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


# ============================================================================
# DATA DOWNLOAD AND PREPROCESSING
# ============================================================================

def download_spam_dataset() -> Path:
    """Download and extract the SMS spam collection dataset."""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    if data_file_path.exists():
        print(f"Dataset already exists at {data_file_path}")
        return data_file_path
    
    print("Downloading SMS spam dataset...")
    
    # Create unverified SSL context for download
    ssl_context = ssl._create_unverified_context()
    
    # Download the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    
    # Extract the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # Rename to add .tsv extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    
    # Clean up zip file
    os.remove(zip_path)
    
    print(f"Dataset downloaded and saved as {data_file_path}")
    return data_file_path


def prepare_dataset(data_file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare balanced train/validation/test splits."""
    # Load data
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    
    # Create balanced dataset
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    
    # Convert labels to numeric
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # Split dataset
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    train_end = int(len(balanced_df) * 0.7)
    validation_end = train_end + int(len(balanced_df) * 0.1)
    
    train_df = balanced_df[:train_end]
    validation_df = balanced_df[train_end:validation_end]
    test_df = balanced_df[validation_end:]
    
    print(f"Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}")
    
    # Save splits
    train_df.to_csv("train.csv", index=False)
    validation_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    
    return train_df, validation_df, test_df


# ============================================================================
# DATASET CLASS
# ============================================================================

class SpamDataset(Dataset):
    """Dataset class for SMS spam classification."""
    
    def __init__(self, csv_file: str, tokenizer, max_length: Optional[int] = None, 
                 pad_token_id: int = 50256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        
        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        # Set max length
        if max_length is None:
            self.max_length = self._get_longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        
        # Pad sequences
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _get_longest_encoded_length(self) -> int:
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


# ============================================================================
# GPT-2 MODEL ARCHITECTURE
# ============================================================================

class LayerNorm(nn.Module):
    """Layer normalization module."""
    
    def __init__(self, ndim: int, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0
        
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.emb_dim = cfg["emb_dim"]
        
        self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg.get("qkv_bias", False))
        self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg.get("qkv_bias", False))
        self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg.get("qkv_bias", False))
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=True)
        
        self.attn_dropout = nn.Dropout(cfg["drop_rate"])
        self.resid_dropout = nn.Dropout(cfg["drop_rate"])
        
        # Causal attention mask
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(cfg["context_length"], cfg["context_length"]))
            .view(1, 1, cfg["context_length"], cfg["context_length"])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        
        return y


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias=True),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias=True),
            nn.Dropout(cfg["drop_rate"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """GPT-2 model for text classification."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], 2, bias=True)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        B, T = in_idx.shape
        
        # Token and position embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(T, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Transformer blocks
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        # Output head
        logits = self.out_head(x)
        return logits


# ============================================================================
# WEIGHT LOADING
# ============================================================================

def assign(left: nn.Parameter, right: np.ndarray) -> nn.Parameter:
    """Assign weights with shape checking."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: dict):
    """Load pre-trained GPT-2 weights into the model."""
    # Position and token embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, np.squeeze(params['wpe']))
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, np.squeeze(params['wte']))
    
    # Transformer blocks
    for b in range(len(params["blocks"])):
        block = params["blocks"][b]
        
        # Attention weights
        c_attn_w = np.squeeze(block["attn"]["c_attn"]['w'])
        c_attn_b = np.squeeze(block["attn"]["c_attn"]['b'])
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=-1)
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=-1)
        
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        # Output projection
        out_proj_w = np.squeeze(block["attn"]["c_proj"]["w"])
        out_proj_b = np.squeeze(block["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, out_proj_w.T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, out_proj_b)
        
        # Feed-forward layers
        ff1_w = np.squeeze(block["mlp"]["c_fc"]["w"])
        ff1_b = np.squeeze(block["mlp"]["c_fc"]["b"])
        ff2_w = np.squeeze(block["mlp"]["c_proj"]["w"])
        ff2_b = np.squeeze(block["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, ff1_w.T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, ff1_b)
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, ff2_w.T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, ff2_b)
        
        # Layer norms
        ln1_g = np.squeeze(block["ln_1"]["g"])
        ln1_b = np.squeeze(block["ln_1"]["b"])
        ln2_g = np.squeeze(block["ln_2"]["g"])
        ln2_b = np.squeeze(block["ln_2"]["b"])
        
        gpt.trf_blocks[b].norm1.weight = assign(gpt.trf_blocks[b].norm1.weight, ln1_g)
        gpt.trf_blocks[b].norm1.bias = assign(gpt.trf_blocks[b].norm1.bias, ln1_b)
        gpt.trf_blocks[b].norm2.weight = assign(gpt.trf_blocks[b].norm2.weight, ln2_g)
        gpt.trf_blocks[b].norm2.bias = assign(gpt.trf_blocks[b].norm2.bias, ln2_b)
    
    # Final layer norm and output head
    final_g = np.squeeze(params["g"])
    final_b = np.squeeze(params["b"])
    wte = np.squeeze(params["wte"])
    
    gpt.final_norm.weight = assign(gpt.final_norm.weight, final_g)
    gpt.final_norm.bias = assign(gpt.final_norm.bias, final_b)
    gpt.out_head.weight = assign(gpt.out_head.weight, wte)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, 
                   model: nn.Module, device: torch.device) -> torch.Tensor:
    """Calculate loss for a batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Last token logits
    loss = F.cross_entropy(logits, target_batch)
    return loss


def calc_accuracy_loader(data_loader: DataLoader, model: nn.Module, 
                        device: torch.device, num_batches: Optional[int] = None) -> float:
    """Calculate accuracy over a data loader."""
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
    
    return correct_predictions / num_examples


def train_classifier(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, device: torch.device, 
                   num_epochs: int, eval_freq: int = 50, eval_iter: int = 5):
    """Train the classifier."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            # Evaluation step
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = sum(calc_loss_batch(input_batch, target_batch, model, device).item()
                                   for i, (input_batch, target_batch) in enumerate(train_loader)
                                   if i < eval_iter) / eval_iter
                    val_loss = sum(calc_loss_batch(input_batch, target_batch, model, device).item()
                                 for i, (input_batch, target_batch) in enumerate(val_loader)
                                 if i < eval_iter) / eval_iter
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                model.train()
        
        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train accuracy {train_accuracy*100:.2f}% | "
              f"Val accuracy {val_accuracy*100:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen


# ============================================================================
# SPAM DETECTION
# ============================================================================

def classify_text(text: str, model: nn.Module, tokenizer, device: torch.device, 
                 max_length: int, pad_token_id: int = 50256) -> Tuple[str, float]:
    """Classify a text as spam or not spam."""
    model.eval()
    
    # Prepare input
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    
    # Truncate if too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    
    # Pad if too short
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        probabilities = F.softmax(logits, dim=-1)
        predicted_label = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_label].item()
    
    result = "SPAM" if predicted_label == 1 else "NOT SPAM"
    return result, confidence


def interactive_spam_detection(model: nn.Module, tokenizer, device: torch.device, 
                             max_length: int):
    """Interactive spam detection interface."""
    print("\n" + "="*60)
    print("SPAM DETECTION INTERFACE")
    print("="*60)
    print("Enter text to classify as spam or not spam.")
    print("Type 'quit' to exit.")
    print("-"*60)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            result, confidence = classify_text(text, model, tokenizer, device, max_length)
            
            print(f"\nClassification: {result}")
            print(f"Confidence: {confidence*100:.2f}%")
            
            if result == "SPAM":
                print("⚠️  This message appears to be spam!")
            else:
                print("✅ This message appears to be legitimate.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Memory management for large model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory available: {gpu_memory:.1f} GB")
        if gpu_memory < 4:
            print("Warning: Your GPU has limited memory. Consider reducing batch size if you encounter memory issues.")
    else:
        print("Training on CPU will be slow but safe for this model size.")
    
    # Set random seed
    torch.manual_seed(123)
    
    # Download and prepare data
    data_file_path = download_spam_dataset()
    train_df, validation_df, test_df = prepare_dataset(data_file_path)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create datasets
    train_dataset = SpamDataset("train.csv", tokenizer=tokenizer, max_length=None)
    val_dataset = SpamDataset("validation.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)
    
    print(f"Dataset max length: {train_dataset.max_length}")
    
    # Create data loaders
    batch_size = 4  # Safe for 4GB VRAM with GPT-2 124M
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # Model configuration for GPT-2 124M (safe for 4GB VRAM)
    model_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }
    
    # Check context length
    assert train_dataset.max_length <= model_config["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model_config['context_length']}"
    )
    
    # Load pre-trained model
    try:
        from gpt_download3 import download_and_load_gpt2
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
        
        # Update model config with actual GPT-2 124M settings
        model_config.update({
            "emb_dim": settings["n_embd"],
            "n_layers": settings["n_layer"],
            "n_heads": settings["n_head"],
            "context_length": settings["n_ctx"]
        })
        
        model = GPTModel(model_config)
        load_weights_into_gpt(model, params)
        model.eval()
        
        # Freeze pre-trained weights
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace output head for classification
        model.out_head = nn.Linear(model_config["emb_dim"], 2)
        
        # Unfreeze last layer and final norm for fine-tuning
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        
        print("Pre-trained GPT-2 124M model loaded successfully!")
        print(f"Model configuration: {model_config}")
        
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load pre-trained model ({e}). Using untrained model.")
        model = GPTModel(model_config)
        model.out_head = nn.Linear(model_config["emb_dim"], 2)
    
    model = model.to(device)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    print("\nStarting training...")
    start_time = time.time()
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model, train_loader, val_loader, optimizer, device, num_epochs=2, eval_freq=50, eval_iter=5
    )
    
    training_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {training_time:.2f} minutes.")
    
    # Final evaluation
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    
    print(f"\nFinal Results:")
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "spam_classifier.pth")
    print("Model saved as 'spam_classifier.pth'")
    
    # Interactive spam detection
    interactive_spam_detection(model, tokenizer, device, train_dataset.max_length)

import matplotlib.pyplot as plt

labels = ['Ham', 'Spam']
counts = [4827, 747]

plt.figure(figsize=(5, 3))
plt.bar(labels, counts, color=['#4CAF50', '#F44336'])
plt.title("SMS Dataset Class Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/images/spam_ham_chart.png")

if __name__ == "__main__":
    main() 