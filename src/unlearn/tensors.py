from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

CARRYOVER = "model-base-carryover.safetensors"
INDEX = "model.safetensors.index.json"


_ITEMSIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2,
             "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: Path) -> tuple[dict, int]:

    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def shard_index(model_dir: Path) -> dict[str, str]:

    model_dir = Path(model_dir)
    idx = model_dir / INDEX
    if idx.is_file():
        return json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"{model_dir} has no model.safetensors or index — is it a HF checkpoint?")
    header, _ = read_header(single)
    return {k: single.name for k in header if k != "__metadata__"}


def missing_tensors(src: Path, dst: Path) -> list[str]:

    return sorted(set(shard_index(src)) - set(shard_index(dst)))


def carry_over_missing(src: Path, dst: Path) -> list[str]:


    src, dst = Path(src), Path(dst)
    base = shard_index(src)
    have = shard_index(dst)
    missing = sorted(set(base) - set(have))
    if not missing:
        return []

    raw: dict[str, tuple[str, list, bytes]] = {}
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in missing:
        by_shard[base[name]].append(name)
    for shard, names in by_shard.items():
        header, start = read_header(src / shard)
        with (src / shard).open("rb") as f:
            for name in names:
                info = header[name]
                a, b = info["data_offsets"]
                f.seek(start + a)
                raw[name] = (info["dtype"], info["shape"], f.read(b - a))

    order = sorted(missing, key=lambda n: (-_ITEMSIZE.get(raw[n][0], 1), n))
    header, offset = {}, 0
    for name in order:
        dtype, shape, data = raw[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header["__metadata__"] = {"format": "pt",
                              "provenance": f"copied byte-for-byte from {src.name}: tensors "
                                            f"transformers did not instantiate, so never trained"}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)          
    with (dst / CARRYOVER).open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in order:
            f.write(raw[name][2])

    weight_map = dict(have)
    weight_map.update({name: CARRYOVER for name in missing})
    total = 0
    for shard in sorted(set(weight_map.values())):
        h, _ = read_header(dst / shard)
        total += sum(v["data_offsets"][1] - v["data_offsets"][0]
                     for k, v in h.items() if k != "__metadata__")
    (dst / INDEX).write_text(json.dumps({"metadata": {"total_size": total},
                                         "weight_map": dict(sorted(weight_map.items()))}, indent=2),
                             encoding="utf-8")
    return missing
