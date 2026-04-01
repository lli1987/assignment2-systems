import regex as re
import time
from collections import defaultdict
import multiprocessing as mp
import os
from pathlib import Path

from cs336_basics.utils import (
    find_chunk_boundaries,
    get_pre_token_bytes,
)
import cs336_basics.constants as constants

import logging
import heapq

logger = logging.getLogger(__name__)

G_PRE_TOKEN_COUNT_PKL = "g_pre_tokens_count.pkl"
VOCAB_PKL = "vocab.pkl"
MERGES_PKL = "merges.pkl"
SPECIAL_TOKENS = "special_tokens.pkl"


class PairKey:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        # compare first element

        for key in [0, 1]:
            min_length = min(len(self.pair[key]), len(other.pair[key]))
            for idx in range(min_length):
                if self.pair[key][idx] < other.pair[key][idx]:
                    return False
                elif self.pair[key][idx] > other.pair[key][idx]:
                    return True
            if len(self.pair[key]) > len(other.pair[key]):
                return True
            elif len(self.pair[key]) < len(other.pair[key]):
                return False
        return False


def prefix_with_name(name, pkl):
    return f"{name}_{pkl}"


def serialize_bpe(encoding_output_dir, obj, file):
    import pickle

    # Ensure the output directory exists
    Path(encoding_output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(encoding_output_dir, file), "wb") as f:
        pickle.dump(obj, f)


def _chunk_doc_streaming(input_path, num_chunks, special):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special.encode("utf-8"))
        logger.warning(len(boundaries))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def _chunk_to_counter_process(chunk_queue, counter_queue, special_tokens):
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        docs = re.split(
            "|".join([re.escape(special_token) for special_token in special_tokens]),
            chunk,
        )
        pre_tokens_count: dict[tuple[bytes], int] = defaultdict(int)
        for doc in docs:
            if not doc:
                continue

            # counts track pre-token occurrence, key is bytes tuple
            for pre_token_group in re.finditer(constants.PAT, doc):
                pre_token = get_pre_token_bytes(pre_token_group)
                pre_token = [bytes([b]) for b in pre_token]
                key = tuple(pre_token)
                pre_tokens_count[key] += 1
        p = mp.current_process()
        logger.warning(f"Completed one chunk by {p.name}")
        counter_queue.put(pre_tokens_count)


def _counter_to_merge_process(counter_queue, merge_queue):
    merged_counter = defaultdict(int)
    while True:
        pre_token_count = counter_queue.get()
        if pre_token_count is None:
            break
        p = mp.current_process()
        logger.warning(f"Completed one count merge by {p.name}")
        for pre_token, cnt in pre_token_count.items():
            merged_counter[pre_token] += cnt
    merge_queue.put(merged_counter)


def _process_pretoken_count(input_paths, num_counter_processes, num_merge_processes, special_tokens, num_chunks):
    counter_queue = mp.Queue()
    chunk_queue = mp.Queue()
    merge_queue = mp.Queue(maxsize=num_merge_processes)

    counter_processes = []
    merge_processes = []

    for i in range(num_counter_processes):
        p = mp.Process(
            target=_chunk_to_counter_process,
            args=(chunk_queue, counter_queue, special_tokens),
            name=f"counter process {i}",
        )
        p.start()
        counter_processes.append(p)

    for i in range(num_merge_processes):
        p = mp.Process(
            target=_counter_to_merge_process,
            args=(counter_queue, merge_queue),
            name=f"merge process {i}",
        )
        p.start()
        merge_processes.append(p)

    for input_path in input_paths:
        for chunk in _chunk_doc_streaming(input_path=input_path, num_chunks=num_chunks, special="<|endoftext|>"):
            chunk_queue.put(chunk)

    for _ in range(num_counter_processes):
        chunk_queue.put(None)

    for p in counter_processes:
        p.join()

    for _ in range(num_merge_processes):
        counter_queue.put(None)

    merged_counter = defaultdict(int)

    for _ in range(num_merge_processes):
        pre_token_count = merge_queue.get()
        if pre_token_count is None:
            break
        for pre_token, cnt in pre_token_count.items():
            merged_counter[pre_token] += cnt

    for p in merge_processes:
        p.join()
    return merged_counter


def train_bpe(
    name: str,
    input_paths: list[str],
    vocab_size: int,
    encoding_output_dir: str,
    special_tokens: list[str],
    num_counter_processes: int,
    num_merge_processes: int,
    num_chunks: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Ensure the output directory exists
    Path(encoding_output_dir).mkdir(parents=True, exist_ok=True)

    # Check if input files exist
    for input_path in input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

    start_time = time.perf_counter()
    pre_tokens_count = _process_pretoken_count(
        input_paths=input_paths,
        num_counter_processes=num_counter_processes,
        num_merge_processes=num_merge_processes,
        special_tokens=special_tokens,
        num_chunks=num_chunks,
    )
    end_time = time.perf_counter()
    logger.warning(f"Time take to pre-tokenize: {end_time-start_time} secs")

    vocab, merges = merge(
        pre_tokens_count=pre_tokens_count,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_time_merge = time.perf_counter()
    logger.warning(f"Time take to merge: {end_time_merge-end_time} secs")
    name = f"{name}_{vocab_size}"
    serialize_bpe(encoding_output_dir, vocab, prefix_with_name(name, VOCAB_PKL))
    serialize_bpe(encoding_output_dir, merges, prefix_with_name(name, MERGES_PKL))
    serialize_bpe(encoding_output_dir, special_tokens, prefix_with_name(name, SPECIAL_TOKENS))
    return vocab, merges


def build_pair_counts_and_pair_to_pre_tokens(pre_tokens_count):
    counts: dict[tuple[bytes], int] = defaultdict(int)
    pair_to_pre_tokens: dict[tuple[bytes], set] = defaultdict(set)
    for pre_token, cnt in pre_tokens_count.items():
        # count pair occurrence across all pre-tokens
        for t1, t2 in zip(pre_token, pre_token[1:]):
            counts[(t1, t2)] += cnt
            pair_to_pre_tokens[(t1, t2)].add(pre_token)
    return counts, pair_to_pre_tokens


def build_heap(pair_counts):
    heap = []
    for pair, cnt in pair_counts.items():
        heap.append((-cnt, PairKey(pair), pair))
    heapq.heapify(heap)
    return heap


def update_data_per_affected_pre_tokens(
    affected_pre_tokens,
    merge_pair,
    pre_tokens_count,
    pair_to_pre_token_set,
    pair_count,
    heap,
):
    affected_pre_tokens_copy = [p for p in affected_pre_tokens]
    for affected_pre_token in affected_pre_tokens_copy:
        cnt = pre_tokens_count[affected_pre_token]
        new_pre_token = []
        idx = 0
        new_pairs = []
        old_pairs = []
        prev_was_merged = False
        while idx < len(affected_pre_token):
            if (
                idx < len(affected_pre_token) - 1
                and affected_pre_token[idx] == merge_pair[0]
                and affected_pre_token[idx + 1] == merge_pair[1]
            ):
                new_bytes = b"".join(merge_pair)
                # has left neighbor
                if new_pre_token:
                    left = new_pre_token[-1]
                    new_pair = (left, new_bytes)
                    old_left = merge_pair[1] if prev_was_merged else left
                    old_pair = (old_left, merge_pair[0])
                    new_pairs.append(new_pair)
                    old_pairs.append(old_pair)
                # has right neighbor
                if idx + 2 < len(affected_pre_token):
                    right = affected_pre_token[idx + 2]
                    next_is_merge = (
                        idx + 3 < len(affected_pre_token)
                        and affected_pre_token[idx + 2] == merge_pair[0]
                        and affected_pre_token[idx + 3] == merge_pair[1]
                    )
                    if not next_is_merge:
                        new_pair = (new_bytes, right)
                        old_pair = (merge_pair[1], right)
                        new_pairs.append(new_pair)
                        old_pairs.append(old_pair)

                new_pre_token.append(new_bytes)
                idx += 2
                prev_was_merged = True
            else:
                new_pre_token.append(affected_pre_token[idx])
                idx += 1
                prev_was_merged = False
        # 1. update pre_tokens count
        # 2. update pair -> pre_tokens mapping
        # 3. update heap
        # 4. update pair count

        final_new_pre_token = tuple(new_pre_token)
        pre_tokens_count[final_new_pre_token] += cnt
        # logger.warning(pre_tokens_count.get(affected_pre_token, None))
        pre_tokens_count.pop(affected_pre_token, 0)
        # logger.warning(pre_tokens_count.get(affected_pre_token, None))

        all_old_pairs = set(zip(affected_pre_token, affected_pre_token[1:]))
        directly_changed = set(new_pairs) | set(old_pairs) | {merge_pair}

        for pair in all_old_pairs - directly_changed:
            # These pairs survive unchanged but the pre_token key changed
            pair_to_pre_token_set[pair].discard(affected_pre_token)
            pair_to_pre_token_set[pair].add(final_new_pre_token)

        for new_pair in new_pairs:
            pair_to_pre_token_set[new_pair].add(final_new_pre_token)
            pair_count[new_pair] += cnt
            heapq.heappush(
                heap,
                (
                    -pair_count[new_pair],
                    PairKey(new_pair),
                    new_pair,
                ),
            )
        # Count occurrences of each pair in new_pre_token
        new_pre_token_pairs = set(zip(final_new_pre_token, final_new_pre_token[1:]))

        for old_pair in old_pairs:
            pair_to_pre_token_set[old_pair].discard(affected_pre_token)
            # If the pair still exists in new pre-token, add new pre-token to the set
            if old_pair in new_pre_token_pairs:
                pair_to_pre_token_set[old_pair].add(final_new_pre_token)
            pair_count[old_pair] -= cnt
            if pair_count[old_pair] <= 0:
                pair_count.pop(old_pair, 0)
                pair_to_pre_token_set.pop(old_pair, None)
            else:
                heapq.heappush(heap, (-pair_count[old_pair], PairKey(old_pair), old_pair))
    pair_count.pop(merge_pair, 0)
    pair_to_pre_token_set.pop(merge_pair, None)


def merge(
    pre_tokens_count: dict[tuple[bytes], int], vocab_size, special_tokens
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {}
    idx = 0
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx + i] = bytes([i])

    # counts track pre-token occurrence, key is bytes tuple, each element
    # starts from one byte, then gets merged
    pair_counts, pair_to_pre_token_set = build_pair_counts_and_pair_to_pre_tokens(pre_tokens_count)
    heap = build_heap(pair_counts)

    for i in range(vocab_size - 256 - len(special_tokens)):
        logger.warning(f"Iteration {i}...")

        if not heap:
            break

        found_valid = False
        pair = None
        # if there is at least one pair
        while heap:
            # find the most occurred pair, if tie, pick lexicographically largest
            neg_cnt, _, pair = heapq.heappop(heap)
            exist_cnt = pair_counts.get(pair, -123456789)
            if exist_cnt == -neg_cnt:
                found_valid = True
                break
        if not found_valid:
            break
        new_index = len(vocab)
        # update vocabuary: new index -> new pair
        vocab[new_index] = pair[0] + pair[1]
        # update merges: new pair -> new index
        merges.append(pair)

        affected_pre_tokens = pair_to_pre_token_set[pair]
        # update pre-token pool
        update_data_per_affected_pre_tokens(
            affected_pre_tokens=affected_pre_tokens,
            merge_pair=pair,
            pre_tokens_count=pre_tokens_count,
            pair_to_pre_token_set=pair_to_pre_token_set,
            pair_count=pair_counts,
            heap=heap,
        )
    return vocab, merges
