import re
import unicodedata
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import json


class VietnameseWordPieceTokenizer:
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ):
        """
        Vietnamese WordPiece Tokenizer

        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for tokens to be included
            unk_token: Unknown token
            pad_token: Padding token
            cls_token: Classification token
            sep_token: Separator token
            mask_token: Mask token for MLM
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Special tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        self.special_tokens = [pad_token, unk_token, cls_token, sep_token, mask_token]

        # Vocabularies
        self.vocab = {}
        self.inv_vocab = {}
        self.word_freq = Counter()

        # Compiled regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for text cleaning"""
        # HTML tags
        self.html_pattern = re.compile(r"<[^>]+>")

        # URLs
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Email addresses
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        # Multiple spaces
        self.multi_space_pattern = re.compile(r"\s+")

        # Fix punctuation spacing (Vietnamese specific)
        self.punct_space_patterns = [
            (re.compile(r"\s*([,;:!?.])\s*"), r" \1 "),
            (re.compile(r'\s*([()"\[\]{}])\s*'), r" \1 "),
            (re.compile(r"\s*([-–—])\s*"), r" \1 "),
        ]

        # Vietnamese specific patterns
        self.vietnamese_chars = re.compile(
            r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]"
        )

    def normalize_unicode(self, text: str) -> str:
        """Normalize text to NFC form"""
        return unicodedata.normalize("NFC", text)

    def remove_html_urls_noise(self, text: str) -> str:
        """Remove HTML tags, URLs, emails and other noise"""
        # Remove HTML tags
        text = self.html_pattern.sub(" ", text)

        # Remove URLs
        text = self.url_pattern.sub(" ", text)

        # Remove email addresses
        text = self.email_pattern.sub(" ", text)

        # Remove extra whitespace
        text = self.multi_space_pattern.sub(" ", text)

        return text.strip()

    def fix_spacing_punctuation(self, text: str) -> str:
        """Fix spacing around punctuation marks"""
        for pattern, replacement in self.punct_space_patterns:
            text = pattern.sub(replacement, text)

        # Clean up multiple spaces again
        text = self.multi_space_pattern.sub(" ", text)

        return text.strip()

    def preprocess_text(self, text: str, word_segment_func=None) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            word_segment_func: Your word_segment function
        """
        # 1. Unicode Normalization (NFC)
        text = self.normalize_unicode(text)

        # 2. Remove HTML, URLs, Noise
        text = self.remove_html_urls_noise(text)

        # 3. Fix Spacing & Punctuation
        text = self.fix_spacing_punctuation(text)

        # 4. Word Segmentation
        if word_segment_func:
            text = word_segment_func(text)

        return text

    def get_word_pairs(self, word: str) -> Set[Tuple[str, str]]:
        """Get all adjacent character pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get frequency statistics for character pairs"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            word_pairs = self.get_word_pairs(word)
            for pair in word_pairs:
                pairs[pair] += freq
        return pairs

    def merge_vocab(
        self, pair: Tuple[str, str], vocab: Dict[str, int]
    ) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word in vocab:
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def build_vocab(self, texts: List[str], word_segment_func=None):
        """
        Build WordPiece vocabulary from texts

        Args:
            texts: List of training texts
            word_segment_func: Your word_segment function
        """
        print("Preprocessing texts...")

        # Preprocess all texts and collect word frequencies
        word_freq = Counter()
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i}/{len(texts)}")

            processed_text = self.preprocess_text(text, word_segment_func)
            words = processed_text.split()

            # Add space between characters for WordPiece
            spaced_words = []
            for word in words:
                if len(word) > 1:
                    spaced_word = " ".join(list(word)) + " </w>"
                    spaced_words.append(spaced_word)
                    word_freq[spaced_word] += 1
                elif word:
                    word_freq[word + " </w>"] += 1

        # Filter by minimum frequency
        word_freq = {
            word: freq for word, freq in word_freq.items() if freq >= self.min_frequency
        }

        print(f"Initial vocabulary size: {len(word_freq)}")

        # Initialize vocabulary with characters
        vocab = word_freq.copy()

        # Add special tokens
        for token in self.special_tokens:
            vocab[token] = float("inf")  # Ensure special tokens are never removed

        print("Learning WordPiece merges...")

        # Learn merges
        merges = []
        for i in range(self.vocab_size - len(self.special_tokens)):
            if i % 100 == 0:
                print(
                    f"Merge iteration {i}/{self.vocab_size - len(self.special_tokens)}"
                )

            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            merges.append(best_pair)

        # Create final vocabulary
        self.vocab = {}
        self.inv_vocab = {}

        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inv_vocab[i] = token

        # Add learned tokens
        vocab_items = [
            (word, freq)
            for word, freq in vocab.items()
            if word not in self.special_tokens
        ]
        vocab_items.sort(key=lambda x: x[1], reverse=True)

        for i, (word, freq) in enumerate(
            vocab_items[: self.vocab_size - len(self.special_tokens)]
        ):
            idx = i + len(self.special_tokens)
            self.vocab[word] = idx
            self.inv_vocab[idx] = word

        self.merges = merges
        print(f"Final vocabulary size: {len(self.vocab)}")

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece"""
        if not word:
            return []

        # Add spaces between characters
        word = " ".join(list(word)) + " </w>"

        # Apply learned merges
        for pair in self.merges:
            bigram = re.escape(" ".join(pair))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            word = pattern.sub("".join(pair), word)

        # Split and filter empty tokens
        tokens = [token for token in word.split() if token]

        # Handle unknown tokens
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(token)
            else:
                result.append(self.unk_token)

        return result

    def tokenize(
        self, text: str, word_segment_func=None, add_special_tokens: bool = False
    ) -> List[str]:
        """
        Tokenize text into WordPiece tokens

        Args:
            text: Input text
            word_segment_func: Your word_segment function
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
        """
        # Preprocess text
        processed_text = self.preprocess_text(text, word_segment_func)

        # Tokenize words
        tokens = []
        if add_special_tokens:
            tokens.append(self.cls_token)

        words = processed_text.split()
        for word in words:
            word_tokens = self.tokenize_word(word)
            tokens.extend(word_tokens)

        if add_special_tokens:
            tokens.append(self.sep_token)

        return tokens

    def encode(
        self, text: str, word_segment_func=None, add_special_tokens: bool = False
    ) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text, word_segment_func, add_special_tokens)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.inv_vocab.get(id_, self.unk_token) for id_ in token_ids]

        # Remove special tokens
        tokens = [token for token in tokens if token not in self.special_tokens]

        # Join tokens and clean up
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file"""
        vocab_data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
        }

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data["vocab"]
        self.merges = vocab_data["merges"]
        self.special_tokens = vocab_data["special_tokens"]
        self.vocab_size = vocab_data["vocab_size"]
        self.min_frequency = vocab_data["min_frequency"]

        # Rebuild inverse vocabulary
        self.inv_vocab = {v: k for k, v in self.vocab.items()}


# Example usage
if __name__ == "__main__":
    # Example Vietnamese texts
    sample_texts = [
        "Xin chào, tôi là một trợ lý AI.",
        "Hôm nay thời tiết rất đẹp ở Hà Nội.",
        "Tôi thích ăn phở và bún chả.",
        "Việt Nam có nhiều danh lam thắng cảnh đẹp.",
    ]

    # Initialize tokenizer
    tokenizer = VietnameseWordPieceTokenizer(vocab_size=1000)

    # You would call this with your word_segment function
    # def your_word_segment(text: str) -> str:
    #     # Your implementation here
    #     return text

    # Build vocabulary
    # tokenizer.build_vocab(sample_texts, word_segment_func=your_word_segment)

    # Example without word segmentation
    tokenizer.build_vocab(sample_texts)

    # Test tokenization
    test_text = "Xin chào thế giới!"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)

    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")

    # Save vocabulary
    # tokenizer.save_vocab("vietnamese_wordpiece_vocab.json")
