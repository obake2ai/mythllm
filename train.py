import os
import re
from datetime import datetime

import torch
import wandb
import click
import tiktoken
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.gpt2 import GPT
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
@click.option('--gpu-count', default=1, show_default=True, help="Number of GPUs to use")
def train_model(**kwargs):
    """
    Main training function that initializes wandb and starts training.
    """
    dist.init_process_group(backend='nccl', world_size=kwargs['gpu_count'], rank=0)  # Initialize process group
    device = torch.device(f'cuda:{dist.get_rank()}')  # Select the appropriate GPU

    # Initialize wandb with all passed options
    wandb.init(project=kwargs["project_name"], config=kwargs)
    config = wandb.config

    # Step 1: Load and preprocess text data
    formatted_text = load_and_format_text(config.data_dir)

    # Step 2: Tokenize the text
    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab
    data = torch.tensor(tokenizer.encode(formatted_text), dtype=torch.long, device=device)

    print(f"\nTensor shape: {data.shape}")

    # Step 3: Initialize dataset and data loaders
    train_data, eval_data = split_data(data, config.train_split)
    train_sampler = DistributedSampler(train_data)
    eval_sampler = DistributedSampler(eval_data)

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size, sampler=train_sampler)
    eval_loader = DataLoader(eval_data, batch_size=config.eval_batch_size, sampler=eval_sampler)

    # Step 4: Initialize model
    model = GPT(vocab_size=vocab_size, d_model=config.d_model, n_heads=config.n_heads,
                n_layers=config.n_layers, context_length=config.context_length).to(device)
    model = DDP(model, device_ids=[dist.get_rank()])

    # Step 5: Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=config.lr * 0.01)

    # Step 6: Create directory for saving models
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(config.save_dir, current_time)
    os.makedirs(model_save_dir, exist_ok=True)

    # Step 7: Train and Evaluate the model
    train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, model_save_dir, config, device)

    # Finalize wandb and destroy process group
    wandb.finish()
    dist.destroy_process_group()


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


def split_data(data, train_split):
    """Split data into training and evaluation sets."""
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]
    return train_data, eval_data


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


def train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, save_dir, config, device):
    """Train and evaluate the model."""
    train_loss = {}

    for epoch in range(config.epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()

            train_loss[epoch] = loss.item()

        # Evaluate periodically
        if epoch % config.eval_steps == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                for xvb, yvb in eval_loader:
                    xvb, yvb = xvb.to(device), yvb.to(device)
                    _, batch_loss = model(xvb, yvb)
                    eval_loss += batch_loss.item()

            print(f"Epoch: {epoch}\ttrain_loss: {loss:.4f}\teval_loss: {eval_loss:.4f}")
            wandb.log({"train_loss": loss.item(), "eval_loss": eval_loss})

            # Save the model
            save_model(model, optimizer, scheduler, epoch, loss.item(), eval_loss, save_dir, config)


def save_model(model, optimizer, scheduler, epoch, train_loss, eval_loss, save_dir, config):
    """Save the model state and relevant information."""
    save_path = os.path.join(save_dir, f"gpt_model_epoch_{epoch}.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),  # DDP対応
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
