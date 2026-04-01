import torch
import numpy as np
import logging
import multiprocessing as mp
import os

import wandb
import regex as re
from cs336_basics.model import LLM
from cs336_basics.bpe_encoding import Tokenizer


from cs336_basics.utils import find_chunk_boundaries


logger = logging.getLogger(__name__)


def _get_full_path(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def _chunk_doc_streaming(input_path, num_chunks, special, prev_id):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special.encode("utf-8"))
        id = prev_id
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            id += 1
            logger.warning(f"reading chunk with chuck_id: {id}")
            yield {"chunk_id": id, "chunk": chunk}


def _chunk_encoding(chunk_queue, shard_queue, tokenizer, tmp_output_dir, name):
    import os

    pid = os.getpid()
    print(f"[PID {pid}] Encoding process started", flush=True)
    chunk_count = 0
    while True:
        chunk_dict = chunk_queue.get()
        if chunk_dict is None:
            print(f"[PID {pid}] Received termination signal after processing {chunk_count} chunks", flush=True)
            break
        chunk_id, chunk = chunk_dict["chunk_id"], chunk_dict["chunk"]
        docs = re.split(
            "|".join([re.escape(special_token) for special_token in tokenizer.special_tokens]),
            chunk,
        )
        out = []
        for doc in docs:
            for id in tokenizer.encode(doc):
                out.append(id)

        chunk_id_file = _get_full_path(tmp_output_dir, f"{name}_chunk_{chunk_id:010d}_id_list.bin")
        arr = np.array(out, dtype=np.int64)
        arr.tofile(chunk_id_file)
        logger.warning(f"finished encoding chunk with chuck_id: {chunk_id}")
        print(f"[PID {pid}] About to put shard {chunk_id} into queue", flush=True)
        shard_queue.put(chunk_id_file)
        print(f"[PID {pid}] Successfully put shard {chunk_id} into queue", flush=True)
        chunk_count += 1
    print(f"[PID {pid}] Encoding process exiting", flush=True)


def _merge_shards(shard_queue, output_file):
    shard_files = []
    logger.warning("Starting to collect shard files from queue...")
    while True:
        shard_path = shard_queue.get()
        if shard_path is None:
            logger.warning("Received termination signal, all shards collected")
            break
        shard_files.append(shard_path)
        logger.warning(f"Collected shard {len(shard_files)}: {shard_path}")

    logger.warning(f"Total shards collected: {len(shard_files)}")
    logger.warning("Starting to merge shards...")
    with open(output_file, "wb") as out:
        for idx, shard in enumerate(sorted(shard_files), 1):
            logger.warning(f"Merging shard {idx}/{len(shard_files)}: {shard}")
            with open(shard, "rb") as f:
                out.write(f.read())
            os.remove(shard)
    logger.warning(f"Finished merging all shards into {output_file}")


def encode_training_data(
    memmap_output,
    name,
    encoding_output_dir,
    num_encoding_processes,
    training_files,
    num_chunks,
    tokenizer,
):

    encoding_processes = []

    chunk_queue = mp.Queue()
    shard_queue = mp.Queue()

    # Start merge process BEFORE encoding processes to consume from queue
    logger.warning("Starting merge process...")
    merge_process = mp.Process(
        target=_merge_shards,
        args=(shard_queue, _get_full_path(encoding_output_dir, memmap_output)),
        name="merge process",
    )
    merge_process.start()

    for i in range(num_encoding_processes):
        p = mp.Process(
            target=_chunk_encoding,
            args=(chunk_queue, shard_queue, tokenizer, encoding_output_dir, name),
            name=f"encoding process: {i}",
        )
        p.start()
        encoding_processes.append(p)

    prev_id, new_prev_id = 0, 0
    for training_file in training_files:
        logger.warning(f"Processing training file: {training_file}")
        for chunk_dict in _chunk_doc_streaming(training_file, num_chunks, "<|endoftext|>", prev_id):
            new_prev_id = chunk_dict["chunk_id"]
            chunk_queue.put(chunk_dict)
        prev_id = new_prev_id

    logger.warning(f"All chunks queued. Sending termination signals to {len(encoding_processes)} encoding processes...")
    for _ in range(len(encoding_processes)):
        chunk_queue.put(None)

    logger.warning("Waiting for all encoding processes to finish...")
    for p in encoding_processes:
        p.join()
    logger.warning("All encoding processes finished!")

    logger.warning("Sending termination signal to merge process...")
    shard_queue.put(None)

    logger.warning("Waiting for merge process to finish...")
    merge_process.join()
    logger.warning("Encoding and merging complete!")


def train(
    name,
    training_files,
    device,
    encoding_output_dir,
    memmap_output,
    tokenizer,
):
    encode_training_data(
        memmap_output=memmap_output,
        encoding_output_dir=encoding_output_dir,
        name=name,
        num_encoding_processes=64,
        training_files=training_files,
        num_chunks=1000,
        tokenizer=tokenizer,
    )

    ids = np.memmap(_get_full_path(encoding_output_dir, memmap_output), dtype=np.int64, mode="r")

    logger.warning(f"Total number of tokens in file: {len(ids)}")

    # Use only a portion of the data to fit in memory
    # For example, use first 100M tokens instead of all 2.87B tokens
    max_tokens = 100_000_000  # Adjust this based on your memory constraints
    if len(ids) > max_tokens:
        logger.warning(f"Using only first {max_tokens:,} tokens out of {len(ids):,}")
        ids = ids[:max_tokens]
    else:
        logger.warning(f"Using all {len(ids):,} tokens")

    # Now load this subset into memory as a regular numpy array
    logger.warning("Loading tokens into memory...")
    x = torch.from_numpy(ids[:].copy()).to(device)
    logger.warning(f"Loaded {len(x):,} tokens into device memory")
    return x
