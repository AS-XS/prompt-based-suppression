#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import items_path, split_path  # noqa: E402
from src.io import read_dicts  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(items: list[dict], split: dict, which: str) -> dict[str, list[dict]]:
    if which not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if not isinstance(split.get("val"), dict) or not split["val"]:
        raise ValueError("missing frozen forget validation IDs")
    val = {i for ids in split["val"].values() for i in ids}
    seen = set()
    groups = defaultdict(list)
    for row in items:
        ident = row["item_id"]
        if ident in seen:
            raise ValueError(f"duplicate item ID: {ident}")
        seen.add(ident)
        tier = row["probe_type"]
        if tier not in {"forget", "neighbor", "utility"}:
            continue
        choices, key = row["meta"].get("choices"), row.get("answer_key")
        if not isinstance(choices, list) or len(choices) != 4 or key not in list("ABCD"):
            raise ValueError(f"not a four-choice MC item: {ident}")

        content = json.dumps([row["question"], choices], ensure_ascii=False)
        is_val = ident in val if tier == "forget" else (
            int(hashlib.sha256(content.encode()).hexdigest(), 16) % 2 == 0)
        if is_val != (which == "val"):
            continue
        subject = row["meta"].get("mmlu_subject") or {
            "wmdp-bio": "biology", "wmdp-cyber": "cyber security",
        }[row["concept_id"]]
        subject = subject.replace("_", " ")
        prompt = (f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                  + row["question"].strip() + "\n"
                  + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
                  + "\nAnswer:")
        group = re.sub(r"[^a-z0-9_]", "_", f"{row['concept_id']}_{tier}")
        groups[group].append(dict(item_id=ident, prompt=prompt, answer="ABCD".index(key)))
    if not val <= seen:
        raise ValueError("frozen split refers to IDs absent from the item file")
    if not groups:
        raise ValueError("no selected MC items")
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="wmdp8")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True, type=Path, help='Out.')
    args = ap.parse_args()
    source, frozen = items_path(args.run), split_path(args.run)
    items = list(read_dicts(source))
    groups = prepare(items, json.loads(frozen.read_text()), args.split)
    args.out.mkdir(parents=True, exist_ok=False)
    names = []
    for group, rows in sorted(groups.items()):
        name = f"unlearn_{args.split}_{group}"
        names.append(name)
        data = args.out / f"{name}.jsonl"
        data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        task = dict(task=name, dataset_path="json",
                    dataset_kwargs={"data_files": {"test": str(data.resolve())}},
                    test_split="test", num_fewshot=0, output_type="multiple_choice",
                    doc_to_text="{{prompt}}", doc_to_choice=list("ABCD"),
                    doc_to_target="answer", target_delimiter=" ",
                    metric_list=[dict(metric="acc", aggregation="mean", higher_is_better=True)],
                    metadata={"version": 1})
        (args.out / f"{name}.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    manifest = dict(protocol="unlearn-likelihood-v1-no-chat", run=args.run, split=args.split,
                    items_sha256=digest(source), split_sha256=digest(frozen),
                    retain_split="sha256(question+choices) parity; val=even",
                    counts={k: len(v) for k, v in groups.items()}, tasks=names)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(",".join(names))


if __name__ == "__main__":
    main()
