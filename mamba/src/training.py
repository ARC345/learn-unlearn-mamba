import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine LR schedule with linear warmup (matches paper Appendix C)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def collate_fn(batch, tokenizer, max_length=1024):
    """Collate batch of text into tokenized tensors."""
    texts = [item["formatted_text"] for item in batch]
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings


def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    """Custom training loop for Mamba (bfloat16, cosine LR with warmup)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)

    tc = config["training"]
    batch_size = tc["per_device_train_batch_size"]
    num_epochs = tc["num_train_epochs"]
    lr = tc["learning_rate"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = tc.get("warmup_steps", 100)

    betas = (tc.get("adam_beta1", 0.9), tc.get("adam_beta2", 0.95))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"], betas=betas)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits.float()  # upcast logits for stable loss

            # Shift for causal LM loss
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save
    os.makedirs(tc["output_dir"], exist_ok=True)
    model.save_pretrained(os.path.join(tc["output_dir"], config["model"]["new_model"]))
    tokenizer.save_pretrained(os.path.join(tc["output_dir"], config["model"]["new_model"]))
