import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import random
from .src.tokenizer import VietnamesePreprocessor, VietnameseTokenizer

class VietnameseTextDataset(Dataset):
    """Dataset class for Vietnamese text generation"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128, stride: int = 64):
        """
        Args:
            texts: List of preprocessed text strings
            tokenizer: Trained VietnameseTokenizer
            max_length: Maximum sequence length
            stride: Stride for creating overlapping sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Create training sequences
        self.sequences = self._create_sequences(texts)
        
    def _create_sequences(self, texts: List[str]) -> List[List[int]]:
        """Create training sequences from texts"""
        all_sequences = []
        
        for text in texts:
            # Encode the entire text
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Create sliding window sequences
            if len(encoded) <= self.max_length:
                # If text is shorter than max_length, use as is
                all_sequences.append(encoded)
            else:
                # Create overlapping sequences
                for i in range(0, len(encoded) - self.max_length + 1, self.stride):
                    sequence = encoded[i:i + self.max_length]
                    all_sequences.append(sequence)
        
        print(f"Created {len(all_sequences)} training sequences")
        return all_sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns input and target sequences for language modeling
        Input: sequence[:-1], Target: sequence[1:]
        """
        sequence = self.sequences[idx]
        
        # Pad sequence if needed
        if len(sequence) < self.max_length:
            pad_length = self.max_length - len(sequence)
            sequence = sequence + [self.tokenizer.word_to_id[self.tokenizer.PAD_TOKEN]] * pad_length
        
        # For language modeling: input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': (input_ids != self.tokenizer.word_to_id[self.tokenizer.PAD_TOKEN]).long()
        }


def prepare_vietnamese_dataset(
    data_file: str,
    tokenizer_file: str = None,
    vocab_size: int = 8000,
    max_length: int = 128,
    train_split: float = 0.8,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, VietnameseTokenizer]:
    """
    Complete pipeline to prepare Vietnamese dataset
    
    Args:
        data_file: Path to the raw text file (like truyen_kieu.txt)
        tokenizer_file: Path to save/load tokenizer
        vocab_size: Vocabulary size for tokenizer
        max_length: Maximum sequence length
        train_split: Ratio for train/validation split
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    
    # Initialize preprocessor and tokenizer
    
    
    preprocessor = VietnamesePreprocessor()
    tokenizer = VietnameseTokenizer(vocab_size=vocab_size)
    
    print("=== Data Preparation Pipeline ===")
    
    # Step 1: Load and preprocess the raw text
    print(f"1. Loading and preprocessing {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"File {data_file} not found. Creating sample data...")
        # Create sample Vietnamese literature text
        raw_text = """
        Trăm năm trong cõi người ta, Chữ tài chữ mệnh khéo là ghét nhau.
        Trải qua một cuộc bể dâu, Những điều trông thấy mà đau đớn lòng.
        Lạ gì bỉ sắc tư phong, Trời xanh quen thói má hồng đánh ghen.
        Cũng đành rằng số kiếp en, Vì chưng nàng sắc nên thêm nàng tài.
        Truyện Kiều được viết bởi Nguyễn Du vào thế kỷ XIX.
        Tác phẩm này được coi là kiệt tác của văn học Việt Nam.
        Câu chuyện kể về số phận của Thúy Kiều, một cô gái tài sắc.
        Qua nhiều thăng trầm, cuối cùng nàng được đoàn tụ với gia đình.
        """
        
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"Created sample data in {data_file}")
    
    # Clean and segment text
    cleaned_text = preprocessor.clean_text(raw_text)
    sentences = preprocessor.segment_sentences(cleaned_text)
    
    print(f"   - Raw text length: {len(raw_text)} characters")
    print(f"   - Cleaned text length: {len(cleaned_text)} characters") 
    print(f"   - Number of sentences: {len(sentences)}")
    
    # Step 2: Train or load tokenizer
    print("2. Training tokenizer")
    
    if tokenizer_file and tokenizer_file.endswith('.json'):
        try:
            tokenizer.load(tokenizer_file)
            print(f"   - Loaded existing tokenizer from {tokenizer_file}")
        except FileNotFoundError:
            print(f"   - Tokenizer file not found. Training new tokenizer...")
            tokenizer.train(sentences)
            tokenizer.save(tokenizer_file)
    else:
        tokenizer.train(sentences)
        if tokenizer_file:
            tokenizer.save(tokenizer_file)
    
    print(f"   - Vocabulary size: {len(tokenizer.vocab)}")
    
    # Step 3: Create datasets
    print("3. Creating datasets")
    
    # Split sentences into train and validation
    random.shuffle(sentences)
    split_idx = int(len(sentences) * train_split)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    print(f"   - Training sentences: {len(train_sentences)}")
    print(f"   - Validation sentences: {len(val_sentences)}")
    
    # Create datasets
    train_dataset = VietnameseTextDataset(train_sentences, tokenizer, max_length)
    val_dataset = VietnameseTextDataset(val_sentences, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer


def test_data_preparation(file_path: str = "truyen_kieu.txt"):
    """Test the data preparation pipeline"""
    print("=== Testing Data Preparation ===")
    
    # Prepare dataset
    train_loader, val_loader, tokenizer = prepare_vietnamese_dataset(
        data_file=file_path,
        tokenizer_file="vietnamese_tokenizer.json",
        vocab_size=5000,
        max_length=64,
        batch_size=4
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
        input_seq = batch['input_ids'][0]
        target_seq = batch['target_ids'][0]
        
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