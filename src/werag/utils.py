from typing import List

import tiktoken


def remove_none_from_dict(d: dict) -> dict:
    non_keys = []
    for key, val in d.items():
        if val is None: non_keys.append(key)
    for k in non_keys: del d[k]
    return d


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Return numbers of token"""
    # https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request
    # https://github.com/openai/tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_string_into_slices(s, slice_size):
    return [s[i:i + slice_size] for i in range(0, len(s), slice_size)]


def limit_tokens(content: str, max_token: int):
    slice_size = 10
    split_: List[str] = split_string_into_slices(content, slice_size)
    last_string = ""
    for sp in split_:
        new_token = count_tokens(last_string + sp)
        if new_token < max_token:
            last_string += sp
        else:
            break
    return last_string
