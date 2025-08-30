import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pickle
import json
import os
import glob

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import (
    Punctuation,
    Sequence,
    Digits,
    Metaspace,
    Whitespace,
)
from tokenizers.normalizers import NFC, NFD, Lowercase, Strip, StripAccents
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline



class VietnamesePreprocessor:
    """Vietnamese text preprocessor handling normalization and cleaning"""

    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
    model = AutoModelForTokenClassification.from_pretrained(
        "NlpHUST/vi-word-segmentation"
    )

    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer)

    def __init__(self):
        # Vietnamese diacritics normalization patterns
        self.tone_patterns = {
            # Handle common tone mark variations
            "à|á|ạ|ả|ã": "a",
            "è|é|ẹ|ẻ|ẽ": "e",
            "ì|í|ị|ỉ|ĩ": "i",
            "ò|ó|ọ|ỏ|õ": "o",
            "ù|ú|ụ|ủ|ũ": "u",
            "ỳ|ý|ỵ|ỷ|ỹ": "y",
        }

    def word_segment(self, text: str) -> str:
        ner_results = self.nlp(text)
        example_tok = ""
        for e in ner_results:
            if "##" in e["word"]:
                example_tok = example_tok + e["word"].replace("##", "")
            elif e["entity"] == "I":
                example_tok = example_tok + "_" + e["word"]
            else:
                example_tok = example_tok + " " + e["word"]
        return example_tok

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to consistent form"""
        # Use NFC normalization for Vietnamese
        return unicodedata.normalize("NFC", text)

    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text"""
        # Normalize unicode first
        text = self.normalize_unicode(text)

        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()


        # Handle common punctuation normalization
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r"…", "...", text)

        return text

    def segment_sentences(self, text: str) -> List[str]:
        """Simple sentence segmentation for Vietnamese"""
        # Vietnamese sentence endings
        sentence_endings = r"[.!?…]+"
        # sentences = re.split(sentence_endings, text)
        sentences = text.split("<s>")

        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def preprocess_file(self, filepath: str, max_length: int = 512) -> List[str]:
        """Preprocess a Vietnamese text file"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean the text
        content = self.clean_text(content)

        # Split into sentences
        sentences = self.segment_sentences(content)

        # Filter sentences by length (in characters)
        filtered_sentences = []
        for sentence in sentences:
            if 10 <= len(sentence) <= max_length:  # Skip very short/long sentences
                filtered_sentences.append(sentence)

        return filtered_sentences


class VietnameseTokenizer:
    def __init__(self):
        self.tokenizer: Tokenizer = None

    def create_vietnamese_normalizer(self):
        """Create a specialized normalizer for Vietnamese text"""
        return normalizers.Sequence(
            [
                Strip(),  # Remove leading/trailing whitespace
                NFC(),  # Normalize Vietnamese diacritics properly
            ]
        )

    def preprocess_vietnamese_text(self, text: str) -> str:
        """Advanced Vietnamese text preprocessing"""
        # Handle common Vietnamese abbreviations and contractions
        vietnamese_contractions = {
            "ko": "không",
            "k": "không",
            "dc": "được",
            "đc": "được",
            "vs": "với",
            "tui": "tôi",
            "mik": "mình",
            "mk": "mình",
            "ny": "người yêu",
            "cx": "cũng",
            "hok": "không",
            "hjk": "không",
            "onl": "online",
            "offl": "offline",
        }

        # Replace contractions
        words = text.split()
        processed_words = []
        for word in words:
            clean_word = re.sub(r"[^\w\sÀ-ỹ]", "", word.lower())
            if clean_word in vietnamese_contractions:
                processed_words.append(vietnamese_contractions[clean_word])
            else:
                processed_words.append(word)
        text = " ".join(processed_words)

        # Normalize repeated characters (hahaha -> haha)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Fix common spacing issues around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])\s*", r"\1 ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).lstrip()

        return text

    def create_vietnamese_pretokenizer(self):
        """Create a specialized pre-tokenizer for Vietnamese"""
        return pre_tokenizers.Sequence(
            [
                # Split on whitespace first to handle words properly
                Whitespace(),
                # Handle digits - keep them together for Vietnamese (like years, phone numbers)
                Digits(individual_digits=False),
            ]
        )

    def build_tokenizer(
        self,
        vocab_size: int = 25000,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
    ):
        """Build the Vietnamese tokenizer"""

        if special_tokens is None:
            special_tokens = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[BOS]",
                "[EOS]",
                "[LF]",
            ]

        # Initialize tokenizer with BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Set normalizer
        self.tokenizer.normalizer = self.create_vietnamese_normalizer()

        # Set pre-tokenizer
        self.tokenizer.pre_tokenizer = self.create_vietnamese_pretokenizer()

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[BOS]",
                "[EOS]",
                "[LF]",
            ],
            initial_alphabet = [
                "a", "á", "à", "ả", "ã", "ạ",
                "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ",
                "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
                "b", "c", "d", "đ",
                "e", "é", "è", "ẻ", "ẽ", "ẹ",
                "ê", "ế", "ề", "ể", "ễ", "ệ",
                "g", "h", "i", "í", "ì", "ỉ", "ĩ", "ị",
                "k", "l", "m", "n",
                "o", "ó", "ò", "ỏ", "õ", "ọ",
                "ô", "ố", "ồ", "ổ", "ỗ", "ộ",
                "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
                "p", "q", "r", "s", "t",
                "u", "ú", "ù", "ủ", "ũ", "ụ",
                "ư", "ứ", "ừ", "ử", "ữ", "ự",
                "v", "x", "y", "ý", "ỳ", "ỷ", "ỹ", "ỵ",

                "A", "Á", "À", "Ả", "Ã", "Ạ",
                "Ă", "Ắ", "Ằ", "Ẳ", "Ẵ", "Ặ",
                "Â", "Ấ", "Ầ", "Ẩ", "Ẫ", "Ậ",
                "B", "C", "D", "Đ",
                "E", "É", "È", "Ẻ", "Ẽ", "Ẹ",
                "Ê", "Ế", "Ề", "Ể", "Ễ", "Ệ",
                "G", "H", "I", "Í", "Ì", "Ỉ", "Ĩ", "Ị",
                "K", "L", "M", "N",
                "O", "Ó", "Ò", "Ỏ", "Õ", "Ọ",
                "Ô", "Ố", "Ồ", "Ổ", "Ỗ", "Ộ",
                "Ơ", "Ớ", "Ờ", "Ở", "Ỡ", "Ợ",
                "P", "Q", "R", "S", "T",
                "U", "Ú", "Ù", "Ủ", "Ũ", "Ụ",
                "Ư", "Ứ", "Ừ", "Ử", "Ữ", "Ự",
                "V", "X", "Y", "Ý", "Ỳ", "Ỷ", "Ỹ", "Ỵ"
            ],
            continuing_subword_prefix="##", 
        )

        return trainer

    def train(self, files: List[str], trainer: WordPieceTrainer):
        """Train the tokenizer on Vietnamese text files"""
        # Preprocess files before training
        processed_files = []

        for file_path in files:
            processed_file = file_path.replace(".txt", "_processed.txt")
            with open(file_path, "r", encoding="utf-8") as infile, open(
                processed_file, "w", encoding="utf-8"
            ) as outfile:

                for line in infile:
                    cleaned_line = line
                    if cleaned_line:  # Only write non-empty lines
                        outfile.write(cleaned_line + "\n")

            processed_files.append(processed_file)

        # Train on processed files
        self.tokenizer.train(processed_files, trainer)

        # Clean up processed files
        for pf in processed_files:
            if os.path.exists(pf):
                os.remove(pf)

    def setup_post_processor(self):
        """Setup post-processor for Vietnamese text"""
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )

    def save(self, path: str):
        """Save the tokenizer"""
        self.tokenizer.save(path)

    def load(self, path: str) -> Tokenizer:
        """Load a saved tokenizer"""
        self.tokenizer = Tokenizer.from_file(path)
        return self.tokenizer

    def encode(self, text: str):
        """Encode text with preprocessing"""
        processed_text = self.preprocess_vietnamese_text(text)
        return self.tokenizer.encode(processed_text)

    def decode(self, ids: List[int]):
        """Decode token ids back to text"""
        return self.tokenizer.decode(ids)

    def test_tokenizer(self):
        """Test the tokenizer with Vietnamese examples"""
        test_sentences = [
            "Xin chào, tôi là một người Việt Nam.",
            "Hôm nay trời đẹp quá, mình đi chơi không?",
            "Tôi rất thích ăn phở và bánh mì.",
            "Anh ấy nói tiếng Anh rất giỏi.",
            "Cô giáo dạy môn Toán rất tốt.",
            "123 con gà, 456 con vịt đang bơi trong ao.",
            "Email: test@gmail.com, Phone: 0123-456-789",
        ]

        print("=== TESTING VIETNAMESE TOKENIZER ===")
        for sentence in test_sentences:
            encoded = self.encode(sentence)
            decoded = self.decode(encoded.ids)

            print(f"Original: {sentence}")
            print(f"Tokens: {encoded.tokens}")
            print(f"Decoded: {decoded}")
            print(f"Token count: {len(encoded.tokens)}")
            print("-" * 50)


# Usage example
def main():
    # Create tokenizer instance
    vn_tokenizer = VietnameseTokenizer()

    # Build tokenizer with Vietnamese-specific settings
    trainer = vn_tokenizer.build_tokenizer(vocab_size=25000, min_frequency=2)

    # Get training files
    train_files = glob.glob(os.path.join("./train_data", "*.txt"))

    if not train_files:
        print("No training files found in ./train_data/")
        print("Please add Vietnamese text files (.txt) to ./train_data/ directory")
        return

    print(f"Found {len(train_files)} training files")

    # Train the tokenizer
    print("Training tokenizer...")
    vn_tokenizer.train(train_files, trainer)

    # # Setup post-processor
    # vn_tokenizer.setup_post_processor()

    # Save tokenizer
    vn_tokenizer.save("vietnamese_enhanced_tokenizer.json")
    print("Tokenizer saved as 'vietnamese_enhanced_tokenizer.json'")

    # Test the tokenizer
    vn_tokenizer.test_tokenizer()


if __name__ == "__main__":
    main()
