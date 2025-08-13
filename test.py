#!/usr/bin/env python3
"""
Fine-tuning script for Qwen3-0.6B on Vietnamese literature dataset
for causal language modeling and text generation.
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from glob import glob
import re

def load_dataset_from_folder(data_folder):
    """Load all text files from the data folder and prepare dataset."""
    texts = []
    
    # Read all .txt files in the data folder
    txt_files = glob(os.path.join(data_folder, "*.txt"))
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by <s> tokens and clean up
        sentences = content.split('<s>')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short texts
                texts.append(sentence)
    
    return texts

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the texts for causal language modeling."""
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2-0.5B"  # Using Qwen2-0.5B as it's more readily available
    DATA_FOLDER = "data"
    OUTPUT_DIR = "./qwen-vietnamese-finetuned"
    MAX_LENGTH = 512
    
    # Training hyperparameters
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    print("Loading and preprocessing dataset...")
    texts = load_dataset_from_folder(DATA_FOLDER)
    print(f"Loaded {len(texts)} text samples")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Split dataset (80% train, 20% validation)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")

def test_generation(model_path, input_text="Bài thơ \"Sóng\" của nhà thơ Xuân"):
    """Test the fine-tuned model with text generation."""
    print("Loading fine-tuned model for testing...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")
    
    return generated_text

if __name__ == "__main__":
    # Run fine-tuning
    main()
    
    # Test the model after training
    model_path = "./qwen-vietnamese-finetuned"
    if os.path.exists(model_path):
        print("\n" + "="*50)
        print("Testing the fine-tuned model:")
        print("="*50)
        test_generation(model_path)