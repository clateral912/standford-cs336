import heapq
import itertools
import multiprocessing
import os

import regex as re
from pretokenization_example import find_chunk_boundaries


def read_file(
    file_path: str | os.PathLike, chunk_pos: list[int]
) -> list[bytes]:
    text_chunks = []
    with open(file_path, "rb") as f:
        for idx in range(0, len(chunk_pos) - 1):
            start, end = idx, idx + 1
            length = end - start
            f.seek(start)
            chunked_text = f.read(length)
            assert len(chunked_text) > 0
            text_chunks.append(chunked_text)

    return text_chunks


def pretokenizer(raw_text: bytes) -> dict[bytes, list[int]]:
    # We just ignore the special tokens
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = re.finditer(PAT, raw_text)

    pre_tokens = []
    for match in matches:
        pre_token = match.group()
        start = match.span()[0]
        if pre_token not in pre_tokens:
            pre_tokens[pre_token] = []
        pre_tokens[pre_token].append(start)

    return pre_tokens


def init_pairs(
    pre_tokens: dict[bytes, list[int]],
) -> dict[tuple[bytes, bytes], tuple[int, list[int]]]:
    pairs = {}
    for token, start_list in pre_tokens.items():
        if len(token) == 1:
            continue
        cnt = len(start_list)
        for idx in range(0, len(token) - 1):
            pair = (token[idx], token[idx + 1])
            if pair not in pairs:
                pairs[pair] = (0, [])

            pair_cnt = pairs[pair][0]
            pair_positions = pairs[pair][1]

            pair_cnt += cnt
            pair_positions.extend([i + idx for i in start_list])

            pairs[pair] = (pair_cnt, pair_positions)

    return pairs


def merge(
    pairs_cnt_heap: heapq,
    pairs: dict[tuple[bytes, bytes], tuple[int, list[int]]],
    merge_pair: tuple[bytes, bytes],
):
    """
    pairs: Dict[Key: token pair, Value: (count, [positions])]
    Example:
        pairs:
            {
                (ea, st), (2, [13, 25]),
                (st, pip), (9, [15, ...]),
                (st, new), (11, [27, ...]),
                ...
            }
        merge:
            (ea, st)

    after merging:
        pairs:
            {
                # there is no pair (ea, st) anymore...
                (st, pip), (8, [36, ...]),      # next pair of (st, pip) will be at 36
                (st, new), (10, [57, ...]),
                (east, pip), (1, [15]),
                (east, new), (1, [27]),
                ...
            }

    """
    first_token, second_token = merge_pair
    new_token = b"".join(merge_pair)
    cnt, merge_pair_positions = pairs[merge_pair]
    second_token_positions = [
        i + len(first_token) for i in merge_pair_positions
    ]
    pairs_begin_with_second_token = [
        k for k in pairs.keys() if k[0] == second_token
    ]

    for pair in pairs_begin_with_second_token:
        cnt, positions = pairs[pair]
        overlapped_positions = list(
            set(second_token_positions) & set(positions)
        )
        if overlapped_positions:
            # insert new pairs
            pairs[(new_token, pair[1])] = (
                len(overlapped_positions),
                [i - len(first_token) for i in overlapped_positions],
            )
            # update old pairs
            updated_positions = list(set(positions) - set(overlapped_positions))
            updated_cnt = cnt - len(overlapped_positions)
            if updated_cnt == 0:
                del pairs[pair]
            else:
                pairs[pair] = (updated_cnt, updated_positions)
                heapq.heappush((-updated_cnt, pair))

    # delete merged pair
    del pairs[merge_pair]

    return None


def my_train_bpe(
    input_path: str | os.PathLike,
    vocab_size_limit: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = [bytes(special_token) for special_token in special_tokens]
    vocab.extend([bytes(i) for i in range(256)])
    vocab = [(i, token) for i, token in enumerate(vocab)]
    vocab = dict(vocab)
    vocab_size = len(vocab)

    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    text_chunks = read_file(input_path, boundaries)
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(pretokenizer, text_chunks)

    pre_tokens = list(itertools.chain.from_iterable(results))
    pairs = init_pairs(pre_tokens)

    # We're using a MAX heap
    pairs_heap = [(-count, pair) for pair, (count, _) in pairs]
    heapq.heapify(pairs_heap)

    merged_pairs = []
    while vocab_size < vocab_size_limit:
        (neg_count, pair_to_merge) = heapq.heappop(pairs_heap)
        count = -neg_count
        if pair_to_merge not in pairs:
            continue
        real_count = pairs[pair_to_merge][0]
        if real_count != count:
            continue

        merge(pairs_heap, pairs, pair_to_merge)

        merged_pairs.append(pair_to_merge)
        # Note that vocab size is the newst token id
        vocab[vocab_size] = b"".join(pair_to_merge)

        vocab_size += 1

    return vocab, merged_pairs
