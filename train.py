import os
import re
import json
from datetime import datetime

import torch
import wandb
import click
import tiktoken
from tokenizers import Tokenizer

import tqdm

from src.gpt2 import GPT
from src.dataloader import DataLoader

from config.token import wandb_api_key, openai
wandb.login(key=wandb_api_key)

@click.command()
@click.option('--project-name', default="mythllm-test", show_default=True, help="WandB project name")
@click.option('--save-dir', default="./checkpoints/", show_default=True, help="Directory to save model checkpoints")
@click.option('--train-batch-size', default=16, show_default=True, help="Training batch size")
@click.option('--eval-batch-size', default=8, show_default=True, help="Evaluation batch size")
@click.option('--context-length', default=512, show_default=True, help="Number of tokens processed in a single batch")
@click.option('--train-split', default=0.8, show_default=True, help="Percentage of data used for training")
@click.option('--d-model', default=512, show_default=True, help="Size of embeddings (d_model)")
@click.option('--n-heads', default=4, show_default=True, help="Number of self-attention heads")
@click.option('--n-layers', default=1, show_default=True, help="Number of GPT blocks/layers")
@click.option('--data-dir', default='./datasets/myth_dataset_v1', show_default=True, help="Directory containing .txt files")
@click.option('--lr', default=5e-4, show_default=True, help="Learning rate")
@click.option('--epochs', default=50000, show_default=True, help="Number of training epochs")
@click.option('--eval-steps', default=500, show_default=True, help="Evaluation step interval")
@click.option('--accumulation-steps', default=4, show_default=True, help="Number of steps for gradient accumulation")
@click.option('--tokenizer', default=None, type=click.Path(exists=True), help="Path to custom tokenizer file (if not using tiktoken)")
@click.option('--config', default=None, type=click.Path(exists=True), help="Path to JSON config file")
def train_model(**kwargs):
    """
    Main training function that initializes wandb and starts training.
    """
    # Load JSON config file if provided
    if kwargs['config']:
        with open(kwargs['config'], 'r') as f:
            json_config = json.load(f)
        kwargs.update(json_config)

    wandb.init(project=kwargs["project_name"], config=kwargs)
    config = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess text data
    formatted_text = load_and_format_text(config.data_dir)

    # Tokenizer selection
    if config.tokenizer:
        tokenizer = Tokenizer.from_file(config.tokenizer)
        vocab_size = tokenizer.get_vocab_size()
        encoded_text = tokenizer.encode(formatted_text).ids
    else:
        tokenizer = tiktoken.get_encoding('gpt2')
        vocab_size = tokenizer.n_vocab
        encoded_text = tokenizer.encode(formatted_text)

    data = torch.tensor(encoded_text, dtype=torch.long, device=device)

    train_loader, eval_loader = initialize_data_loaders(
        data, config.train_split, config.train_batch_size, config.eval_batch_size, config.context_length
    )

    model = GPT(
        vocab_size=vocab_size, d_model=config.d_model, n_heads=config.n_heads,
        n_layers=config.n_layers, context_length=config.context_length
    ).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=config.lr * 0.01)

    # Create directory for saving models
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(config.save_dir, current_time)
    os.makedirs(model_save_dir, exist_ok=True)

    # Train and evaluate
    train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, model_save_dir, config)

    wandb.finish()

def initialize_data_loaders(data, train_split, train_batch_size, eval_batch_size, context_length):
    """Initialize data loaders for training and evaluation."""
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]

    train_loader = DataLoader(train_data, train_batch_size, context_length)
    eval_loader = DataLoader(eval_data, eval_batch_size, context_length)

    return train_loader, eval_loader

def load_and_format_text(data_dir):
    """Load and format text data from the specified directory."""
    combined_text = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_text += file.read() + "\n"

    print_middle_snippet(combined_text, "Original")
    formatted_text = format_text(combined_text)
    print_middle_snippet(formatted_text, "Formatted")

    return formatted_text

def print_middle_snippet(text, label):
    """Print a snippet of the text for verification."""
    middle_index = len(text) // 2
    snippet = text[middle_index - 250: middle_index + 250]
    print(f"\n{label} (Middle 500 characters):\n{snippet}")

def format_text(combined_text):
    """Format text by removing unnecessary line breaks and cleaning up the input."""
    combined_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", combined_text)
    combined_text = re.sub(r"(?<!\n)\n(?!\n)", " ", combined_text)
    return "\n".join(line for line in combined_text.splitlines() if line.strip())

def train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, save_dir, config):
    """Train and evaluate the model."""
    train_loss = {}
    accumulation_steps = config.accumulation_steps

    for e in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        total_steps = len(train_loader)

        # Add progress bar
        with tqdm(total=total_steps, desc=f"Epoch {e + 1}/{config.epochs}") as pbar:
            for step, (xb, yb) in enumerate(train_loader):
                logits, loss = model(xb, yb)
                loss = loss / accumulation_steps  # Normalize loss for accumulation
                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})

        train_loss[e] = epoch_loss / total_steps

        # Evaluation logic
        if e % config.eval_steps == 0 or e == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                for xvb, yvb in eval_loader:
                    _, eval_loss = model(xvb, yvb)
                    break  # Only evaluate one batch to save time

            print(f"\nEpoch: {e}\ttrain_loss: {train_loss[e]:.4f}\teval_loss: {eval_loss:.4f}")
            wandb.log({"train_loss": train_loss[e], "eval_loss": eval_loss.item()})

            save_model(model, optimizer, scheduler, e, train_loss[e], eval_loss.item(), save_dir, config)
def save_model(model, optimizer, scheduler, epoch, train_loss, eval_loss, save_dir, config):
    """Save the model state and relevant information."""
    save_path = os.path.join(save_dir, f"gpt_model_epoch_{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'config': dict(config)
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
