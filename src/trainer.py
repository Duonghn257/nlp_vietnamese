import torch
from .model import VietnameseTransformer


# Training utilities
class VietnameseTrainer:
    """Trainer for Vietnamese Transformer"""

    def __init__(
        self,
        model: VietnameseTransformer,
        train_loader,
        val_loader,
        tokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        device: str = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=warmup_steps, T_mult=2
        )

        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self) -> float:
        """Train for one epoch"""
        from tqdm import tqdm

        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training (epoch)", leave=False)
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["target_ids"].to(self.device)

            # Forward pass
            logits, loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_loss=True,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                progress_bar.set_postfix(
                    loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)

                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_loss=True,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs: int, save_path: str = "vietnamese_transformer.pt"):
        """Complete training loop with tqdm progress bar"""
        from tqdm import trange

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in trange(num_epochs, desc="Epochs"):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )
                print(f"💾 Saved best model with val_loss: {val_loss:.4f}")

            # Generate sample text
            if (epoch + 1) % 5 == 0:
                self.generate_sample()

    def generate_sample(self):
        """Generate sample text to monitor training progress"""
        self.model.eval()

        sample_input = "Truyện Kiều được viết"
        input_ids = torch.tensor(
            [self.tokenizer.encode(sample_input, add_special_tokens=False).ids],
            device=self.device,
        )

        with torch.no_grad():
            generated = self.model.generate(
                input_ids, max_new_tokens=10, temperature=0.8, do_sample=True
            )

        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        print(f"🎯 Sample: '{generated_text}'")
