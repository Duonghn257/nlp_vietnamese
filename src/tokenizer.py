import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pickle
import json


class VietnamesePreprocessor:
    """Vietnamese text preprocessor handling normalization and cleaning"""

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
    """Custom tokenizer for Vietnamese text using BPE-like approach"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = set()

        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.BOS_TOKEN = "<bos>"  # Beginning of sequence
        self.EOS_TOKEN = "<eos>"  # End of sequence

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        # Vietnamese syllable pattern
        self.vietnamese_syllable_pattern = re.compile(
            r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+",
            re.IGNORECASE,
        )

    def _extract_vietnamese_words(self, text: str) -> List[str]:
        """Extract Vietnamese words/syllables from text"""
        # Find Vietnamese syllables
        vietnamese_words = self.vietnamese_syllable_pattern.findall(text.lower())

        # Find punctuation and numbers
        other_tokens = re.findall(r"[^\w\s]|\d+", text)

        # Combine and maintain rough order (this is simplified)
        all_tokens = []
        words_iter = iter(vietnamese_words)
        other_iter = iter(other_tokens)

        # Simple tokenization: split by whitespace and extract tokens
        for token in text.split():
            if self.vietnamese_syllable_pattern.match(token.lower()):
                all_tokens.append(token.lower())
            else:
                # Handle punctuation and mixed tokens
                sub_tokens = re.findall(r"[^\w\s]|\w+", token.lower())
                all_tokens.extend(sub_tokens)

        return all_tokens

    def train(self, texts: List[str]):
        """Train tokenizer on Vietnamese texts"""
        print("Training Vietnamese tokenizer...")

        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = self._extract_vietnamese_words(text)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Start with special tokens
        vocab = self.special_tokens.copy()

        # Add most frequent tokens up to vocab_size
        most_common = token_counts.most_common(
            self.vocab_size - len(self.special_tokens)
        )
        for token, count in most_common:
            if len(vocab) >= self.vocab_size:
                break
            if token not in vocab:
                vocab.append(token)

        # Create mappings
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab = set(vocab)

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Most common tokens: {most_common[:10]}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._extract_vietnamese_words(text)

        # Convert to IDs
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.word_to_id[self.BOS_TOKEN])

        for token in tokens:
            if token in self.word_to_id:
                token_ids.append(self.word_to_id[token])
            else:
                token_ids.append(self.word_to_id[self.UNK_TOKEN])

        if add_special_tokens:
            token_ids.append(self.word_to_id[self.EOS_TOKEN])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)

        # Join tokens with spaces (simplified for Vietnamese)
        return " ".join(tokens)

    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "vocab": list(self.vocab),
            "special_tokens": self.special_tokens,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        self.vocab_size = tokenizer_data["vocab_size"]
        self.word_to_id = tokenizer_data["word_to_id"]
        # Convert string keys back to int for id_to_word
        self.id_to_word = {int(k): v for k, v in tokenizer_data["id_to_word"].items()}
        self.vocab = set(tokenizer_data["vocab"])
        self.special_tokens = tokenizer_data["special_tokens"]

        print(f"Tokenizer loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = VietnamesePreprocessor()

    # Sample Vietnamese text (Truyện Kiều excerpt)
    sample_text = """
    Trăm năm trong cõi người ta,
    Chữ tài chữ mệnh khéo là ghét nhau.
    Trải qua một cuộc bể dâu,
    Những điều trông thấy mà đau đớn lòng.
    Lạ gì bỉ sắc tư phong,
    Trời xanh quen thói má hồng đánh ghen.
    Cũng đành rằng số kiếp en,
    Vì chưng nàng sắc nên thêm nàng tài.
    """

    # Preprocess the text
    print("=== Text Preprocessing ===")
    cleaned_text = preprocessor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned_text[:100]}...")

    sentences = preprocessor.segment_sentences(cleaned_text)
    print(f"Number of sentences: {len(sentences)}")
    print(f"First sentence: {sentences[0]}")

    # Train tokenizer
    print("\n=== Tokenizer Training ===")
    tokenizer = VietnameseTokenizer(vocab_size=1000)

    # Use sentences for training
    tokenizer.train(sentences)

    # Test tokenization
    print("\n=== Tokenization Examples ===")
    test_sentence = "Truyện Kiều được viết bởi Nguyễn Du."

    # Encode
    encoded = tokenizer.encode(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Encoded: {encoded}")

    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test with your example
    print(f"\n=== Your Example ===")
    input_text = "Truyện Kiều được viết"
    target_text = "bởi Nguyễn Du."

    input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
    target_encoded = tokenizer.encode(target_text, add_special_tokens=False)

    print(f"Input: '{input_text}' -> {input_encoded}")
    print(f"Target: '{target_text}' -> {target_encoded}")

    # Save tokenizer for later use
    tokenizer.save("vietnamese_tokenizer.json")

    print(f"\nTokenizer vocabulary sample:")
    sample_vocab = list(tokenizer.vocab)[:20]
    for token in sample_vocab:
        print(f"  '{token}' -> {tokenizer.word_to_id[token]}")
