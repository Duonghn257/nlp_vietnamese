from .tokenizer import VietnameseTokenizer, VietnamesePreprocessor
from .model import VietnameseTransformer
from .helpers import setup_training_config
import torch
from typing import Text


class VietnamesePoem:
    def __init__(
        self,
        config_path: str = "config.yaml",
        device: str = "cuda",
    ):
        self.model, self.tokenizer = VietnamesePoem.load_model_config(config_path)
        self.preprocessor = VietnamesePreprocessor()
        self.config_path = config_path
        self.device = device

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> VietnameseTokenizer:
        tokenizer = VietnameseTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer.tokenizer

    @staticmethod
    def load_model_config(
        config_path: str = "config.yaml",
    ) -> tuple[VietnameseTransformer, VietnameseTokenizer]:
        config = setup_training_config(config_path=config_path)
        tokenizer = VietnamesePoem.load_tokenizer(config["tokenizer_file"])

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

        return model, tokenizer

    def generate_poem(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Text:
        self.model.eval()
        self.model.to(self.device)

        prompt = self.preprocessor.clean_text(prompt)
        prompt = self.preprocessor.word_segment(prompt)

        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=False).ids]
        ).to(self.device)

        generated_tokens = self.model.generate(
            input_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            pad_token_id=self.tokenizer.token_to_id("[PAD]"),
            eos_token_id=self.tokenizer.token_to_id("[EOS]"),
        )
        generated_text = self.tokenizer.decode(generated_tokens[0].cpu().tolist())
        return generated_text.replace(" ##", "").replace("_", " ").replace("[LF]", "\n")
