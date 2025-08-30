import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import random
from .tokenizer import VietnamesePreprocessor, VietnameseTokenizer
from tokenizers import Tokenizer
import glob
import os


class VietnameseTextDataset(Dataset):
    """Dataset class for Vietnamese text generation"""

    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Args:
            texts: List of preprocessed text strings
            tokenizer: Trained Tokenizer
            max_length: Maximum sequence length (default 512)
            stride: Stride for creating overlapping sequences when splitting long texts
        """
        self.tokenizer: Tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Create training sequences
        self.sequences = self._create_sequences(texts)

    def _create_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Create training sequences by tokenizing each text individually.
        If tokenized sequence > 512 tokens: split into overlapping chunks
        If tokenized sequence < 512 tokens: pad to max_length
        """
        # Ensure special tokens exist and get their IDs
        self.tokenizer.add_special_tokens(["[EOS]", "[PAD]"])
        pad_id = self.tokenizer.token_to_id("[PAD]")
        eos_id = self.tokenizer.token_to_id("[EOS]")

        all_sequences = []

        for text in texts:
            # Tokenize the text
            encoded = self.tokenizer.encode(text, add_special_tokens=False).ids

            # If the sequence is longer than max_length, split it into chunks with overlap
            if len(encoded) > self.max_length:
                for i in range(0, len(encoded), self.stride):
                    # Get chunk
                    chunk = encoded[i : i + self.max_length]

                    # If this is the last chunk and it's shorter than max_length, pad it
                    if len(chunk) < self.max_length:
                        padding_needed = self.max_length - len(chunk)
                        chunk.extend([pad_id] * padding_needed)

                    all_sequences.append(chunk)
            else:
                # If sequence is shorter than max_length, pad it
                padding_needed = self.max_length - len(encoded)
                encoded.extend([pad_id] * padding_needed)
                all_sequences.append(encoded)

        print(f"Created {len(all_sequences)} training sequences")
        return all_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns input and target sequences for causal language modeling
        Input: sequence[:-1], Target: sequence[1:]
        """
        sequence = self.sequences[idx]

        # Get pad token id
        pad_id = self.tokenizer.token_to_id("[PAD]")

        # For causal language modeling: input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        target_ids = torch.tensor(sequence, dtype=torch.long)

        # Attention mask should be 1 for non-padding tokens and 0 for padding tokens
        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }


def load_texts_from_folder(folder_path: str) -> List[str]:
    """Loads text from multiple files and returns a list of text strings, one per file."""
    all_texts = []
    # Use glob to find all files ending with .txt
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                all_texts.append(text)
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
    return all_texts


def prepare_vietnamese_dataset(
    data_folder: str,
    tokenizer: Tokenizer = None,
    *,
    max_length: int = 128,
    train_split: float = 0.8,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    Complete pipeline to prepare Vietnamese dataset

    Args:
        data_folder: Path to the folder contains text file (like ./data, ./dataset)
        tokenizer: Tokenizer
        vocab_size: Vocabulary size for tokenizer
        max_length: Maximum sequence length
        train_split: Ratio for train/validation split
        batch_size: Batch size for DataLoader

    Returns:
        train_loader, val_loader
    """

    # Initialize preprocessor
    preprocessor = VietnamesePreprocessor()

    print("=== Data Preparation Pipeline ===")

    # Step 1: Load and preprocess the raw texts
    print(f"1. Loading and preprocessing text from folder {data_folder}")

    # This now returns a list of texts, not a single large string
    raw_texts = load_texts_from_folder(data_folder)

    # Combine and flatten the list of sentences from all documents
    all_sentences = []
    for text in raw_texts:
        cleaned_text = preprocessor.clean_text(text)
        sentences_from_file = preprocessor.segment_sentences(cleaned_text)
        all_sentences.extend(
            sentences_from_file
        )  # Use extend to add all sentences to one list

    # print(all_sentences)
    print(f"   - Number of sentences: {len(all_sentences)}")

    # # Step 2: Train or load tokenizer
    print("2. Training tokenizer")
    if tokenizer:
        # try:
        # tokenizer.load(tokenizer_file)
        print(f"   - Loaded existing tokenizer successful")
    else:
        print(f"   - Tokenizer file not found. Training new tokenizer...")

    print(f"   - Vocabulary size: {tokenizer.get_vocab_size()}")

    # Step 3: Create datasets
    print("3. Creating datasets")

    # Split sentences into train and validation
    random.shuffle(all_sentences)
    split_idx = int(len(all_sentences) * train_split)
    train_sentences = all_sentences[:split_idx]
    val_sentences = all_sentences[split_idx:]

    print(f"   - Training sentences: {len(train_sentences)}")
    print(f"   - Validation sentences: {len(val_sentences)}")

    # # Create datasets
    train_dataset = VietnameseTextDataset(train_sentences, tokenizer, max_length)
    val_dataset = VietnameseTextDataset(val_sentences, tokenizer, max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def test_data_preparation():
    """Test the data preparation pipeline"""
    print("=== Testing Data Preparation ===")

    # Prepare dataset
    train_loader, val_loader, tokenizer = prepare_vietnamese_dataset(
        data_file="truyen_kieu.txt",
        tokenizer_file="vietnamese_tokenizer.json",
        vocab_size=5000,
        max_length=64,
        batch_size=4,
    )

    print("\n=== Testing Data Loader ===")

    # Test the data loader
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Only show first 2 batches
            break

        print(f"\nBatch {i + 1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Target IDs shape: {batch['target_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")

        # Show first sequence in batch
        input_seq = batch["input_ids"][0]
        target_seq = batch["target_ids"][0]

        print(f"  Sample input:  {input_seq[:10].tolist()}...")
        print(f"  Sample target: {target_seq[:10].tolist()}...")

        # Decode sample
        decoded_input = tokenizer.decode(input_seq.tolist())
        decoded_target = tokenizer.decode(target_seq.tolist())

        print(f"  Decoded input:  '{decoded_input[:50]}...'")
        print(f"  Decoded target: '{decoded_target[:50]}...'")

    print("\n=== Testing Text Generation Format ===")

    # Test specific example format
    input_text = "Truyện Kiều được viết"
    target_text = "bởi Nguyễn Du."

    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    print(f"Input: '{input_text}'")
    print(f"  Encoded: {input_ids}")
    print(f"  Decoded: '{tokenizer.decode(input_ids)}'")

    print(f"Target: '{target_text}'")
    print(f"  Encoded: {target_ids}")
    print(f"  Decoded: '{tokenizer.decode(target_ids)}'")

    # Show how to prepare for model training
    full_sequence = input_ids + target_ids
    model_input = torch.tensor(full_sequence[:-1], dtype=torch.long)
    model_target = torch.tensor(full_sequence[1:], dtype=torch.long)

    print(f"\nModel training format:")
    print(f"  Model input:  {model_input.tolist()}")
    print(f"  Model target: {model_target.tolist()}")

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Run the test
    train_loader, val_loader, tokenizer = test_data_preparation()

    print(f"\n=== Summary ===")
    print(f"Dataset prepared successfully!")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Ready for transformer model training!")
