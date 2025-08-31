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
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Import your modules (make sure these files are in the same directory)
from src.tokenizer import VietnamesePreprocessor, VietnameseTokenizer
from src.dataset import prepare_vietnamese_dataset
from src.trainer import VietnameseTrainer
from src.model import VietnameseTransformer

preprocessor = VietnamesePreprocessor()


def setup_training_config(config_path: str = "config.yaml"):
    """Setup training configuration from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        # Flatten the nested YAML structure for backward compatibility
        config = {}

        # Data configuration
        config.update(yaml_config.get("data", {}))

        # Model configuration
        config.update(yaml_config.get("model", {}))

        # Training configuration
        config.update(yaml_config.get("training", {}))

        # Generation configuration
        config.update(yaml_config.get("generation", {}))

        # Save configuration
        config.update(yaml_config.get("save", {}))

        # Convert string values to appropriate types
        config = convert_config_types(config)

        print(f"✅ Configuration loaded from {config_path}")
        return config

    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found. Using default configuration.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config file: {e}")
        return get_default_config()


def convert_config_types(config):
    """Convert string values in config to appropriate data types"""
    type_conversions = {
        # Data configuration
        "vocab_size": int,
        "max_seq_len": int,
        "train_split": float,
        # Model configuration
        "d_model": int,
        "n_heads": int,
        "n_layers": int,
        "d_ff": int,
        "dropout": float,
        # Training configuration
        "batch_size": int,
        "learning_rate": float,
        "weight_decay": float,
        "num_epochs": int,
        "warmup_steps": int,
        # Generation configuration
        "temperature": float,
        "top_k": int,
        "top_p": float,
        "max_new_tokens": int,
    }

    converted_config = {}
    for key, value in config.items():
        if key in type_conversions:
            try:
                converted_config[key] = type_conversions[key](value)
            except (ValueError, TypeError):
                print(
                    f"⚠️  Warning: Could not convert {key}={value} to {type_conversions[key].__name__}, keeping as string"
                )
                converted_config[key] = value
        else:
            # Keep other values as they are (strings, etc.)
            converted_config[key] = value

    return converted_config


def get_default_config():
    """Return default configuration if YAML file is not available"""
    return {
        # Data configuration
        "data_folder": "data/clean_data",
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
        "num_epochs": 50,
        "warmup_steps": 5000,
        "device": "auto",  # 'cuda', 'cpu', 'mps', or 'auto'
        # Generation configuration
        "temperature": 0.8,
        "top_k": 10,
        "top_p": 0.9,
        "max_new_tokens": 256,
        # Save configuration
        "model_save_path": "vietnamese_transformer_best.pt",
        "config_save_path": "training_config.json",
    }


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
    model: VietnameseTransformer, tokenizer, device, test_cases=None, max_new_tokens=15
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

        prompt = preprocessor.word_segment(prompt)

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

            generated_text = (
                tokenizer.decode(generated[0].cpu().tolist())
                .replace(" ##", "")
                .replace("_", " ")
                .replace("[LF]", "\n")
            )
            print(f"  {config['name']}: '{generated_text}'")


def generate_text(
    prompt: str,
    model: VietnameseTransformer,
    tokenizer,
    device,
    temperature=0.5,
    top_k=20,
    top_p=0.9,
    max_new_tokens=15,
):
    model.eval()
    prompt = preprocessor.clean_text(prompt)
    prompt = preprocessor.word_segment(prompt)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids]
    ).to(device)

    config = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "name": "Balanced",
    }

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
        return generated_text.replace(" ##", "").replace("_", " ").replace("[LF]", "\n")


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
