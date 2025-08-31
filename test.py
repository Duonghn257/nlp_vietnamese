from src import VietnamesePoem
from typing import Text


def invoke(
    poem_generator: VietnamesePoem,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Text:

    return poem_generator.generate_poem(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


if __name__ == "__main__":
    poem_generator = VietnamesePoem(config_path="config.yaml", device="mps")
    print(invoke(poem_generator, "thơ lục bát: ai ơi xa bến quê hương "))
