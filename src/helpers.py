#!/usr/bin/env python3
"""
Complete training script for Vietnamese Text Generation using Decoder-Only Transformer
Usage: python train_vietnamese_transformer.py
"""

import torch
import torch.nn as nn
import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import your modules (make sure these files are in the same directory)
from src.tokenizer import VietnamesePreprocessor, VietnameseTokenizer
from src.dataset import prepare_vietnamese_dataset
from src.trainer import VietnameseTrainer
from src.model import VietnameseTransformer


def setup_training_config():
    """Setup training configuration"""
    config = {
        # Data configuration
        "data_file": "truyen_kieu.txt",
        "tokenizer_file": "vietnamese_tokenizer.json",
        "vocab_size": 5000,
        "max_seq_len": 128,
        "train_split": 0.8,
        # Model configuration
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        # Training configuration
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 50,
        "warmup_steps": 1000,
        "device": "auto",  # 'cuda', 'cpu', or 'auto'
        # Generation configuration
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "max_new_tokens": 50,
        # Save configuration
        "model_save_path": "vietnamese_transformer_best.pt",
        "config_save_path": "training_config.json",
    }
    return config


def create_sample_data(file_path: str):
    """Create sample Vietnamese literature data if file doesn't exist"""
    sample_data = (
        """
    Trăm năm trong cõi người ta,
    Chữ tài chữ mệnh khéo là ghét nhau.
    Trải qua một cuộc bể dâu,
    Những điều trông thấy mà đau đớn lòng.
    Lạ gì bỉ sắc tư phong,
    Trời xanh quen thói má hồng đánh ghen.
    Cũng đành rằng số kiếp en,
    Vì chưng nàng sắc nên thêm nàng tài.

    Truyện Kiều được viết bởi Nguyễn Du vào thế kỷ XIX.
    Tác phẩm này được coi là kiệt tác của văn học Việt Nam.
    Câu chuyện kể về số phận của Thúy Kiều, một cô gái tài sắc vượt trội.
    Qua nhiều thăng trầm trong cuộc đời, cuối cùng nàng được đoàn tụ với gia đình.

    Văn học Việt Nam có nhiều tác phẩm nổi tiếng khác.
    Số Đỏ của Vũ Trọng Phụng là một tiểu thuyết hiện thực xuất sắc.
    Kim Lân nổi tiếng với những truyện ngắn về cuộc sống nông thôn.
    Nguyễn Tuân được biết đến với văn xuôi miêu tả thiên nhiên tuyệt đẹp.

    Thơ ca cổ điển Việt Nam thường sử dụng thể lục bát, thất ngôn tứ tuyệt.
    Truyền thống văn học dân gian rất phong phú với các câu chuyện cổ tích.
    Tấm Cám, Thạch Sanh, Sơn Tinh Thủy Tinh là những truyện nổi tiếng.
    Các câu ca dao, tục ngữ cũng thể hiện triết lý sống sâu sắc.

    Hồ Chí Minh cũng có những bài thơ nổi tiếng viết trong tù.
    Nhật ký trong tù thể hiện tinh thần kiên cường của người cộng sản.
    Văn học hiện đại Việt Nam phát triển mạnh mẽ từ đầu thế kỷ XX.
    Nhiều tác giả trẻ đã góp phần làm giàu kho tàng văn học dân tộc.

    Ngôn ngữ Việt Nam có đặc điểm là đơn âm tiết và có thanh điệu.
    Mỗi tiếng có thể mang nhiều nghĩa khác nhau tùy theo thanh điệu.
    Điều này tạo nên sự phong phú và đa dạng trong cách diễn đạt.
    Văn học Việt Nam khai thác triệt để vẻ đẹp của ngôn ngữ này.

    Truyện Kiều không chỉ là tác phẩm văn học mà còn là bức tranh xã hội.
    Nó phản ánh những mâu thuẫn sâu sắc của xã hội phong kiến.
    Số phận con người bị chi phối bởi hoàn cảnh xã hội.
    Tình yêu và lòng hiếu thảo là những giá trị được tôn vinh.

    Ngày nay, văn học Việt Nam tiếp tục phát triển với nhiều hình thức mới.
    Tiểu thuyết, truyện ngắn, thơ, kịch đều có những tác phẩm xuất sắc.
    Các nhà văn trẻ mang đến làn gió mới cho nền văn học.
    Văn học Việt Nam ngày càng được quốc tế quan tâm và đánh giá cao.
    """
        * 3
    )  # Repeat to have more training data

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sample_data)
    print(f"✅ Created sample data file: {file_path}")


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


def test_generation(
    model, tokenizer, device, test_cases=None, max_new_tokens: int = 20
):
    """Test text generation with various examples"""
    if test_cases is None:
        test_cases = [
            "Truyện Kiều được viết",
            "Văn học Việt Nam",
            "Nguyễn Du là",
            "Thúy Kiều",
            "Tác phẩm này",
        ]

    print("\n" + "=" * 60)
    print("🎯 TESTING TEXT GENERATION")
    print("=" * 60)

    model.eval()

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: '{prompt}'")

        # Encode input
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False).ids], device=device
        )

        # Generate with different settings
        generation_configs = [
            {
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "name": "Balanced",
            },
            {
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": max_new_tokens,
                "name": "Creative",
            },
            {
                "temperature": 0.3,
                "top_k": 10,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "name": "Conservative",
            },
        ]

        for config in generation_configs:
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=True,
                )

            generated_text = tokenizer.decode(generated[0].cpu().tolist())
            print(f"  {config['name']}: '{generated_text}'")


def save_model_and_config(model, tokenizer, config, model_path, config_path):
    """Save the trained model and configuration"""
    # Save model state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": config["vocab_size"],
                "d_model": config["d_model"],
                "n_heads": config["n_heads"],
                "n_layers": config["n_layers"],
                "d_ff": config["d_ff"],
                "max_seq_len": config["max_seq_len"],
                "dropout": config["dropout"],
            },
            "tokenizer_file": config["tokenizer_file"],
        },
        model_path,
    )

    # Save configuration
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"💾 Model saved to: {model_path}")
    print(f"⚙️  Configuration saved to: {config_path}")


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

    # Create data file if it doesn't exist
    if not os.path.exists(config["data_file"]):
        print(f"⚠️  Data file not found: {config['data_file']}")
        print("Creating sample data...")
        create_sample_data(config["data_file"])

    print(f"📊 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 1: Prepare dataset
    print(f"\n{'='*20} STEP 1: DATA PREPARATION {'='*20}")

    train_loader, val_loader, tokenizer = prepare_vietnamese_dataset(
        data_file=config["data_file"],
        tokenizer_file=config["tokenizer_file"],
        vocab_size=config["vocab_size"],
        max_length=config["max_seq_len"],
        train_split=config["train_split"],
        batch_size=config["batch_size"],
    )

    print(f"✅ Dataset prepared successfully!")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")

    # Step 2: Create model
    print(f"\n{'='*20} STEP 2: MODEL CREATION {'='*20}")

    model = VietnameseTransformer(
        vocab_size=len(tokenizer.vocab),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_token_id=tokenizer.word_to_id[tokenizer.PAD_TOKEN],
    )

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
    test_generation(model, tokenizer, trainer.device, ["Truyện Kiều được viết"])

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
    test_generation(model, tokenizer, trainer.device)

    # Step 7: Save final configuration
    save_model_and_config(
        model, tokenizer, config, config["model_save_path"], config["config_save_path"]
    )

    print(f"\n{'='*20} TRAINING COMPLETE {'='*20}")
    print("🎯 Your Vietnamese text generation model is ready!")
    print(f"📁 Model saved: {config['model_save_path']}")
    print(f"📁 Tokenizer: {config['tokenizer_file']}")
    print(f"📁 Config: {config['config_save_path']}")

    # Example usage instructions
    print(f"\n{'='*20} USAGE EXAMPLE {'='*20}")
    print("To use your trained model:")
    print("```python")
    print("# Load model and tokenizer")
    print(f"model, tokenizer = load_model_and_tokenizer(")
    print(f"    '{config['model_save_path']}', '{config['tokenizer_file']}')")
    print("")
    print("# Generate text")
    print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print("model.to(device)")
    print("model.eval()")
    print("")
    print("prompt = 'Truyện Kiều được viết'")
    print(
        "input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)"
    )
    print("generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8)")
    print("result = tokenizer.decode(generated[0].cpu().tolist())")
    print("print(result)")
    print("```")


if __name__ == "__main__":
    # Add argument parser for command line options
    parser = argparse.ArgumentParser(
        description="Train Vietnamese Text Generation Model"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/truyen_kieu.txt",
        help="Path to the Vietnamese text data file",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5000, help="Vocabulary size for tokenizer"
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
    config["data_file"] = args.data_file
    config["vocab_size"] = args.vocab_size
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["device"] = args.device

    # Run training
    main()
