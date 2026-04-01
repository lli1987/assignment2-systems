import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cs336_basics import data
from cs336_basics.optimizer import AdamW
import tiktoken
import re


class TextDataset:
    def __init__(self, input_file, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = self.encode([input_file])

    def encode(self, files):
        ids = []
        for file in files:
            with open(file, "r") as f:
                docs = re.split(
                    "|".join([re.escape(special_token) for special_token in ["<|endoftext|>"]]),
                    f.read(),
                )

                for doc in docs:
                    for id in self.tokenizer.encode(doc):
                        ids.append(id)
        return np.array(ids)


def loading_data(ids, context_length):
    x, y = data.get_batch(dataset=ids, batch_size=4, context_length=context_length, device="cuda")
    return x, y


def get_optimizer(model, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return optimizer


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(
        data_dir="/Users/luyaoli/code/cs336/assignment2-systems/tests/training_data",
        tokenizer=tokenizer,
        context_length=256,
    )
