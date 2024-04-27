def remove_none_from_dict(d: dict) -> dict:
    non_keys = []
    for key, val in d.items():
        if val is None: non_keys.append(key)
    for k in non_keys: del d[k]
    return d
