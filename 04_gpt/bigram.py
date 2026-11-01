"""4.2 baseline: bigram model in PyTorch on Tiny Shakespeare.

Each token reads the logits for the next token straight from a lookup table
(nn.Embedding(vocab, vocab)). Sets up the pieces the GPT reuses: get_batch,
generate, estimate_loss.

Run: python 04_gpt/bigram.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_batch, load_splits
from train import estimate_loss

device = "mps" if torch.backends.mps.is_available() else "cpu"
BLOCK_SIZE, BATCH_SIZE, MAX_ITERS, LR = 8, 32, 5000, 1e-2


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, vocab)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        return logits, F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(idx.device)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    tok, train_data, val_data = load_splits()
    splits = {"train": train_data, "val": val_data}
    model = BigramLanguageModel(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1337)

    for step in range(MAX_ITERS + 1):
        if step % 1000 == 0:
            losses = estimate_loss(model, splits, BLOCK_SIZE, BATCH_SIZE, 200, device)
            print(f"step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        if step == MAX_ITERS:
            break
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device, g)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # token 0 is '\n'
    print(tok.decode(model.generate(context, 300, generator=torch.Generator().manual_seed(42))[0].tolist()))
