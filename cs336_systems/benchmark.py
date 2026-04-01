import argparse
import torch
import time
import logging
import os
from pathlib import Path
from torch.nn.functional import cross_entropy
from cs336_basics import model, nn_utils, bpe_training, bpe_encoding
from cs336_systems.utils import TextDataset, get_optimizer, loading_data
from cs336_systems.config import benchmark_config
from timeit import default_timer


logger = logging.getLogger(__name__)


def main(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    theta: int,
    warmup_steps=10,
    training_steps=100,
    forward_only=True,
    device="cuda",
):
    # Ensure encoding output directory exists
    encoding_output_dir = benchmark_config["encoding_output_dir"]
    Path(encoding_output_dir).mkdir(parents=True, exist_ok=True)

    # Check if input file exists
    input_file = benchmark_config["input_file"]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    bpe_training.train_bpe(
        name=benchmark_config["name"],
        input_paths=[input_file],
        vocab_size=vocab_size,
        encoding_output_dir=encoding_output_dir,
        special_tokens=["<|endoftext|>"],
        num_counter_processes=64,
        num_merge_processes=4,
        num_chunks=1000,
    )
    tokenizer = bpe_encoding.Tokenizer.get_tokenizer(
        benchmark_config["name"], vocab_size, encoding_output_dir
    )
    dataset = TextDataset(
        input_file=input_file,
        tokenizer=tokenizer,
    )
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
        x, y = loading_data(ids, context_length)

        # logger.warning(f"x's dimension: {x.shape}")
        # logger.warning(f"y's dimension: {y.shape}")

        optimizer.zero_grad()
        o = m.forward(x=x)
        loss = nn_utils.cross_entropy(o, y)
        if not forward_only:
            loss.backward()
        optimizer.step()
        logger.warning(f"completed warm up iteration: {i}")

    for i in range(training_steps):
        x, y = loading_data(ids)

        if torch.cuda.is_available():
            t0 = default_timer()
            torch.cuda.synchronize()
        optimizer.zero_grad()
        o = m.forward(x=x)

        if forward_only:
            if torch.cuda.is_available():
                t1 = default_timer()
                torch.cuda.synchronize()
                logger.warning(
                    f"Total time of {t1 - t0:.4f} seconds to train model (forward-only) with "
                    f"vocab_size={vocab_size} context_length={context_length} d_model={d_model} "
                    f"num_layers={num_layers} num_heads={num_heads} d_ff={d_ff} theta={theta}"
                )

        loss = nn_utils.cross_entropy(o, y)
        loss.backward()
        optimizer.step()

        if not forward_only:
            if torch.cuda.is_available():
                t1 = default_timer()
                torch.cuda.synchronize()
                logger.warning(
                    f"Total time of {t1 - t0:.4f} seconds to train model (forward + backward) with "
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
    main(
        10000,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.theta,
    )
