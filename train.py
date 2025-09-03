#!/usr/bin/env python3
"""
Complete training script for Vietnamese Text Generation using Decoder-Only Transformer
Usage: python train_vietnamese_transformer.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    generate_text,
    setup_training_config,
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
    # vn_tokenizer.setup_post_processor()

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


def main():
    """Main training function"""
    print("🇻🇳 Vietnamese Text Generation Training")
    print("=" * 50)

    # Load configuration
    config = setup_training_config()

    # Step 1: Prepare dataset
    print(f"\n{'='*20} STEP 1: DATA PREPARATION {'='*20}")

    vietnam_tokenizer: VietnameseTransformer = None

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
        batch_size=config["batch_size"],
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
        max_new_tokens=50,
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
        model,
        tokenizer,
        trainer.device,
        ["thơ lục bát: ai ơi xa bến quê hương "],
        max_new_tokens=50,
    )


def invoke(prompt: str, max_new_tokens: int):
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
    model = model.to("cuda")
    # Load best model for testing
    if os.path.exists(config["model_save_path"]):
        checkpoint = torch.load(
            config["model_save_path"],
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        # print("✅ Loaded best model for testing")

    response = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        max_new_tokens=max_new_tokens,
    )
    return response


def test(test_cases: list[str], device: str, max_new_tokens: int):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

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
    model = model.to(device)

    # Step 6: Final generation test
    print(f"\n{'='*20} STEP 6: FINAL GENERATION TEST {'='*20}")
    print(config["model_save_path"])
    # Load best model for testing
    if os.path.exists(config["model_save_path"]):
        checkpoint = torch.load(
            "vietnamese_transformer_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded best model for testing")
    test_generation(
        model,
        tokenizer,
        device=device,
        test_cases=test_cases,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    # Run training
    main()
    # response = invoke(
    #     prompt="thơ lục bát: con tằm đắm đuối vì tơ còn ta thì đã ngẩn ngơ vì nàng nàng cho anh hỏi ",
    #     max_new_tokens=100,
    # )
    # print(response)
    # test(
    #     test_cases=[
    #         "thơ lục bát: con tằm đắm đuối vì tơ còn ta thì đã ngẩn ngơ vì nàng nàng cho anh hỏi có thể làm quen, con "
    #     ],
    #     device=config["device"],
    #     max_new_tokens=150,
    # )
