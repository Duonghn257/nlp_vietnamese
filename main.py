#!/usr/bin/env python3
"""
Complete training script for Vietnamese Text Generation using Decoder-Only Transformer
Usage: python train_vietnamese_transformer.py
"""
import os

from torch.utils.data.datapipes import datapipe
from src import (
    VietnameseTrainer,
    VietnamesePreprocessor,
    VietnameseTextDataset,
    VietnameseTokenizer,
    VietnameseTransformer,
    prepare_vietnamese_dataset,
    test_generation,
    load_texts_from_folder,
)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.normalizers import NFD, Sequence as NormalizerSequence

from typing import List, Dict
import torch
import torch.nn as nn
from glob import glob
import matplotlib.pyplot as plt
import json
import argparse
import random


def setup_training_config():
    """Setup training configuration"""
    config = {
        # Data configuration
        "data_folder": "data1",
        "tokenizer_file": "vietnamese_tokenizer.json",
        "vocab_size": 25000,
        "max_seq_len": 512,
        "train_split": 0.8,
        # Model configuration
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "d_ff": 3072,
        "dropout": 0.1,
        # Training configuration
        "batch_size": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "warmup_steps": 5000,
        "device": "auto",  # 'cuda', 'cpu', or 'auto'
        # Generation configuration
        "temperature": 0.8,
        "top_k": 10,
        "top_p": 0.9,
        "max_new_tokens": 512,
        # Save configuration
        "model_save_path": "vietnamese_transformer_best.pt",
        "config_save_path": "training_config.json",
    }
    return config


def build_tokenizer(data_path: str, save_path: str, vocab_size: int):
    vn_tokenizer = VietnameseTokenizer()
    trainer = vn_tokenizer.build_tokenizer(vocab_size=vocab_size, min_frequency=2)

    # Get training files
    train_files = glob(os.path.join(data_path, "*.txt"))

    if not train_files:
        print(f"No training files found in {data_path}")
        print(f"Please add Vietnamese text files (.txt) to {data_path} directory")
        return

    print(f"Found {len(train_files)} training files")

    # Train the tokenizer
    print("Training tokenizer...")
    vn_tokenizer.train(train_files, trainer)

    # Setup post-processor
    vn_tokenizer.setup_post_processor()

    # Save tokenizer
    vn_tokenizer.save(save_path)
    print(f"Tokenizer saved as {save_path}")


def load_tokenizer(tokenizer_path: str) -> VietnameseTokenizer:
    tokenizer = VietnameseTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def plot_training_history(train_losses, val_losses, save_path="training_history.png"):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    plt.title("Vietnamese Transformer Training History", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add annotations
    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    plt.annotate(
        f"Best Val Loss: {min(val_losses):.4f}\nEpoch: {min_val_loss_epoch}",
        xy=(min_val_loss_epoch, min(val_losses)),
        xytext=(min_val_loss_epoch + len(val_losses) * 0.1, min(val_losses) + 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        ha="left",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"📊 Training history saved to: {save_path}")


def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load trained model and tokenizer"""
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    # Create model
    model = VietnameseTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load tokenizer
    tokenizer = VietnameseTokenizer()
    tokenizer.load(tokenizer_path)

    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Tokenizer loaded from: {tokenizer_path}")

    return model, tokenizer


def main():
    """Main training function"""
    print("🇻🇳 Vietnamese Text Generation Training")
    print("=" * 50)

    # Load configuration
    config = setup_training_config()

    # Step 1: Prepare dataset
    print(f"\n{'='*20} STEP 1: DATA PREPARATION {'='*20}")

    vietnam_tokenizer: VietnameseTransformer = None
    vietnam_tokenizer = load_tokenizer(config["tokenizer_file"])
    if vietnam_tokenizer == None:
        build_tokenizer(
            data_path=config["data_folder"],
            save_path=config["tokenizer_file"],
            vocab_size=config["vocab_size"],
        )

        print(f"📊 Training Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        vietnam_tokenizer = load_tokenizer(config["tokenizer_file"])

    tokenizer = vietnam_tokenizer.tokenizer
    train_loader, val_loader = prepare_vietnamese_dataset(
        data_folder=config["data_folder"],
        tokenizer=tokenizer,
        max_length=config["max_seq_len"],
    )

    print(f"✅ Dataset prepared successfully!")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")

    # Step 2: Create model
    print(f"\n{'='*20} STEP 2: MODEL CREATION {'='*20}")

    model = VietnameseTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )
    # exit()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Step 3: Initialize trainer
    print(f"\n{'='*20} STEP 3: TRAINING SETUP {'='*20}")

    trainer = VietnameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        device=config["device"],
    )

    print(f"✅ Trainer initialized!")
    print(f"   Device: {trainer.device}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")

    # Test initial generation (before training)
    print(f"\n{'='*20} INITIAL GENERATION TEST {'='*20}")
    print("Testing generation before training (should be random):")
    test_generation(
        model,
        tokenizer,
        trainer.device,
        ["thơ lục bát: ai ơi xa bến quê hương "],
        max_new_tokens=20,
    )

    # Step 4: Train the model
    print(f"\n{'='*20} STEP 4: TRAINING {'='*20}")
    print(f"Starting training for {config['num_epochs']} epochs...")
    print("Press Ctrl+C to stop training early\n")

    try:
        trainer.train(
            num_epochs=config["num_epochs"], save_path=config["model_save_path"]
        )

        print(f"\n🎉 Training completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print("Saving current model state...")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "train_losses": trainer.train_losses,
                "val_losses": trainer.val_losses,
            },
            "vietnamese_transformer_interrupted.pt",
        )
        print("Model saved as 'vietnamese_transformer_interrupted.pt'")

    # Step 5: Plot training history
    if trainer.train_losses and trainer.val_losses:
        print(f"\n{'='*20} STEP 5: TRAINING ANALYSIS {'='*20}")
        plot_training_history(trainer.train_losses, trainer.val_losses)

        print(f"Training Summary:")
        print(f"  Best validation loss: {min(trainer.val_losses):.4f}")
        print(f"  Final training loss: {trainer.train_losses[-1]:.4f}")
        print(f"  Final validation loss: {trainer.val_losses[-1]:.4f}")

    # Step 6: Final generation test
    print(f"\n{'='*20} STEP 6: FINAL GENERATION TEST {'='*20}")

    # Load best model for testing
    if os.path.exists(config["model_save_path"]):
        checkpoint = torch.load(config["model_save_path"], map_location=trainer.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded best model for testing")

    # Test with multiple examples
    test_generation(
        model, tokenizer, trainer.device, ["thơ lục bát: ai ơi xa bến quê hương"]
    )


def test(test_cases: list[str], max_new_tokens: int):
    config = setup_training_config()
    vietnam_tokenizer = load_tokenizer(config["tokenizer_file"])
    tokenizer = vietnam_tokenizer.tokenizer
    model = VietnameseTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )

    # Step 6: Final generation test
    print(f"\n{'='*20} STEP 6: FINAL GENERATION TEST {'='*20}")
    print(config["model_save_path"])
    # Load best model for testing
    if os.path.exists(config["model_save_path"]):
        checkpoint = torch.load(
            "vietnamese_transformer_interrupted.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded best model for testing")
    test_generation(
        model,
        tokenizer,
        device="cpu",
        test_cases=test_cases,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    # Add argument parser for command line options
    parser = argparse.ArgumentParser(
        description="Train Vietnamese Text Generation Model"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data",
        help="Path to the Vietnamese text data file",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=20000, help="Vocabulary size for tokenizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )

    args = parser.parse_args()

    # Update config with command line arguments
    config = setup_training_config()
    config["data_folder"] = args.data_folder
    # config["data_file"] = args.data_file
    config["vocab_size"] = args.vocab_size
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["device"] = args.device

    # Run training
    main()
    test(test_cases=["thơ lục bát: mùa đông để mộng nằm im "], max_new_tokens=150)
