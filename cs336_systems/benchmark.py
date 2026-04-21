import argparse
import torch
import logging
import os
import numpy as np
from pathlib import Path
from torch.nn.functional import cross_entropy
from cs336_basics import model, nn_utils, bpe_training, bpe_encoding
from cs336_systems.utils import TextDataset, get_optimizer, loading_data
from cs336_systems.config import benchmark_config, local_benchmark_config
from cs336_systems.replacement import annotated_scaled_dot_product_attention
from timeit import default_timer
import torch.cuda.nvtx as nvtx


logger = logging.getLogger(__name__)


def get_time(is_local):
    if is_local:
        t0 = default_timer()
    else:
        if torch.cuda.is_available():
            t0 = default_timer()
            torch.cuda.synchronize()
    return t0


def serialize_dataset(serde_output_dir, obj, file):
    import pickle

    # Ensure the output directory exists
    Path(serde_output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(serde_output_dir, file), "wb") as f:
        pickle.dump(obj, f)


def deserialize_dataset(serde_output_dir, file):
    import pickle

    with open(os.path.join(serde_output_dir, file), "rb") as f:
        return pickle.load(f)


def main(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    theta: int,
    forward_only=True,
):
    config = benchmark_config
    # Ensure encoding output directory exists
    encoding_output_dir = config["encoding_output_dir"]
    serde_output_dir = config["serde_output_dir"]
    Path(encoding_output_dir).mkdir(parents=True, exist_ok=True)
    warmup_steps = config["warmup_steps"]
    training_steps = config["training_steps"]
    is_local = config["is_local"]
    device = config["device"]
    train_bpe = config["train_bpe"]
    encode_ids = config["encode_ids"]

    # Check if input file exists
    input_file = config["input_file"]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if train_bpe:
        bpe_training.train_bpe(
            name=config["name"],
            input_paths=[input_file],
            vocab_size=vocab_size,
            encoding_output_dir=encoding_output_dir,
            special_tokens=["<|endoftext|>"],
            num_counter_processes=64,
            num_merge_processes=4,
            num_chunks=1000,
        )

    if encode_ids:
        tokenizer = bpe_encoding.Tokenizer.get_tokenizer(config["name"], vocab_size, encoding_output_dir)
        dataset = TextDataset(
            input_file=input_file,
            tokenizer=tokenizer,
        )
        serialize_dataset(serde_output_dir, dataset, f"{config['name']}_{vocab_size}_dataset.pkl")
    else:
        dataset = deserialize_dataset(serde_output_dir, f"{config['name']}_{vocab_size}_dataset.pkl")

    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    m = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=theta,
    )
    m.to(device)

    optimizer = get_optimizer(m)
    ids = dataset.tokens

    for i in range(warmup_steps):
        x, y = loading_data(ids, context_length, device)

        optimizer.zero_grad()
        o = m.forward(x=x)
        loss = nn_utils.cross_entropy(o, y)
        loss.backward()
        optimizer.step()
        logger.warning(f"completed warm up iteration: {i}")

    durations = []

    for i in range(training_steps):
        x, y = loading_data(ids, context_length, device)
        t0 = get_time(is_local)
        optimizer.zero_grad()
        with nvtx.range("forward"):
            o = m.forward(x=x)

        if t0 and forward_only:
            t1 = get_time(is_local)
            durations.append(t1 - t0)

        loss = nn_utils.cross_entropy(o, y)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("step function"):
            optimizer.step()

        if t0 and not forward_only:
            t1 = get_time(is_local)
            durations.append(t1 - t0)

    mean = np.mean(durations)
    std = np.std(durations)
    if forward_only:
        logger.warning(
            f"Training time (forward-only) - mean: {mean:.4f}, std: {std:.4f} for model "
            f"vocab_size={vocab_size} context_length={context_length} d_model={d_model} "
            f"num_layers={num_layers} num_heads={num_heads} d_ff={d_ff} theta={theta}"
        )
    else:
        logger.warning(
            f"Training time (forward + backward) - mean: {mean:.4f}, std: {std:.4f} for model "
            f"vocab_size={vocab_size} context_length={context_length} d_model={d_model} "
            f"num_layers={num_layers} num_heads={num_heads} d_ff={d_ff} theta={theta}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark an LLM")
    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="Model size",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of heads",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        required=True,
        help="Feed forward size",
    )
    parser.add_argument(
        "--theta",
        type=int,
        required=True,
        help="Rope theta value",
    )

    args = parser.parse_args()
    main(10000, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.theta)
