import heapq
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
            start, end = chunk_pos[idx], chunk_pos[idx + 1]
            length = end - start
            f.seek(start)
            chunked_text = f.read(length)
            assert len(chunked_text) > 0
            text_chunks.append(chunked_text)

    return text_chunks


def pretokenizer(raw_text: bytes) -> dict[tuple[bytes, ...], int]:
    # We just ignore the special tokens
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = re.finditer(PAT, raw_text)

    split_dict = {}
    for match in matches:
        pre_token = match.group()
        result = tuple(bytes([x]) for x in pre_token)
        if result not in split_dict:
            split_dict[result] = 0
        split_dict[result] += 1

    return split_dict


def init_pairs(
    splited_tokens: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    pairs = {}
    for token, cnt in splited_tokens.items():
        if len(token) == 1:
            continue
        for idx in range(0, len(token) - 1):
            pair = (token[idx : idx + 1], token[idx + 1 : idx + 2])
            if pair not in pairs:
                pairs[pair] = 0
            pairs[pair] += cnt

    return pairs


def merge(
    pairs_cnt_heap: heapq,
    pairs: dict[tuple[bytes, bytes], int],
    splited_dict: dict[tuple[bytes, ...], int],
    merge_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    first_token, second_token = merge_pair
    new_token = b"".join(merge_pair)
    new_split_dict = {}
    changed_pairs = set()
    for word, cnt in splited_dict.items():
        # word: (b"a", b"c", b"d")
        new_word = []
        idx = 0
        while idx < len(word) - 1:
            curr, next = word[idx], word[idx + 1]
            if (curr != first_token) or (
                curr == first_token and next != second_token
            ):
                new_word.append(word[idx])
                idx += 1
            if curr == first_token and next == second_token:
                new_word.append(new_token)

                if idx > 0:
                    prev_pair = (word[idx - 1], word[idx])
                    new_prev_pair = (word[idx - 1], new_token)
                    pairs[prev_pair] -= cnt
                    if new_prev_pair not in pairs:
                        pairs[new_prev_pair] = 0
                    pairs[new_prev_pair] += cnt
                    changed_pairs.update([prev_pair, new_prev_pair])

                if idx < len(word) - 2:
                    next_pair = (word[idx + 1], word[idx + 2])
                    new_next_pair = (new_token, word[idx + 2])
                    pairs[next_pair] -= cnt
                    if new_next_pair not in pairs:
                        pairs[new_next_pair] = 0
                    pairs[new_next_pair] += cnt
                    changed_pairs.update([next_pair, new_next_pair])

                idx += 2
        if idx < len(word):
            new_word.append(word[idx])

        new_key = tuple(new_word)
        if new_key not in new_split_dict:
            new_split_dict[new_key] = 0
        new_split_dict[new_key] += cnt

    for pair in changed_pairs:
        heapq.heappush(pairs_cnt_heap, (-pairs[pair], pair))

    del pairs[merge_pair]

    return new_split_dict


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

    split_dict = {}
    for res in results:
        for splited, count in res.items():
            if splited not in split_dict:
                split_dict[splited] = 0
            split_dict[splited] += count

    pairs = init_pairs(split_dict)
    # We're using a MAX heap
    pairs_heap = [(-count, pair) for pair, count in pairs]
    heapq.heapify(pairs_heap)

    merged_pairs = []
    while vocab_size < vocab_size_limit:
        (neg_count, pair_to_merge) = heapq.heappop(pairs_heap)
        count = -neg_count
        if pair_to_merge not in pairs:
            continue
        real_count = pairs[pair_to_merge]
        if real_count != count:
            continue

        split_dict = merge(pairs_heap, pairs, pair_to_merge)

        merged_pairs.append(pair_to_merge)
        # Note that vocab size is the newst token id
        vocab[vocab_size] = b"".join(pair_to_merge)

        vocab_size += 1

    return vocab, merged_pairs
