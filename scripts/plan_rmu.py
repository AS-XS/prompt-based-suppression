#!/usr/bin/env python3


from __future__ import annotations

import argparse
import shlex


def candidates(family: str, concept: str) -> list[dict]:
    if family == "gemma-e2b":

        settings = [(8, 81, 300, 150), (8, 122, 300, 150), (8, 162, 300, 150),
                    (8, 81, 100, 150), (8, 81, 1000, 150), (8, 81, 300, 300)]
        model, lr = "gemma-4-E2B-it", "0.00025"
    elif family == "llama":

        settings = [(7, 6.5, 1200, 150), (7, 20, 300, 150),
                    (7, 20, 600, 150), (7, 20, 1200, 150),
                    (7, 30, 600, 150), (7, 20, 600, 300)]
        model, lr = "Meta-Llama-3-8B-Instruct", "0.00005"
    else:
        settings = [("auto", 1.1, 300, 600), ("auto", 1.65, 300, 600),
                    ("auto", 2.2, 300, 600), ("auto", 1.1, 1000, 600)]
        model, lr = "gemma-4-31B-it", "0.00005"
    return [dict(model=model, concept=concept, layer=layer, c=c, alpha=alpha,
                 steps=steps, lr=lr, seed=42,
                 large=family == "gemma-31b",
                 name=f"rmu-next-{family}-{concept.removeprefix('wmdp-')}-p{i:02d}")
            for i, (layer, c, alpha, steps) in enumerate(settings)]


def command(row: dict, forget_file: str | None) -> str:
    env = dict(MODEL=row["model"], CONCEPT=row["concept"], LAYER=row["layer"],
               NAME=row["name"], STEERING_COEFF=row["c"], ALPHA=row["alpha"],
               LR=row["lr"], MAX_BATCHES=row["steps"], SEED=row["seed"],
               DTYPE="bfloat16", FROZEN_DTYPE="bfloat16")
    if row["concept"] == "wmdp-bio":
        if not forget_file:
            raise ValueError("bio requires --forget-file (authorized corpus on the server)")
        if any(c in forget_file for c in ",\n\r"):
            raise ValueError("SGE -v cannot safely represent commas or newlines in the file path")
        env["FORGET_FILE"] = forget_file
    resources = []
    if row["large"]:
        env.pop("STEERING_COEFF")
        env.update(STEERING_RATIO=row["c"], EARLY_STOP=1,
                   FROZEN_DEVICE="cuda:1", BATCH_SIZE=1)
        resources = ["-l", "gpus=2,gpu_memory=80G"]
    return shlex.join(["qsub", *resources, "-N", row["name"], "-v",
                       ",".join(f"{k}={v}" for k, v in env.items()),
                       "scripts/qsub_unlearn.sh"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=["llama", "gemma-e2b", "gemma-31b"])
    ap.add_argument("--concept", required=True, choices=["wmdp-bio", "wmdp-cyber"])
    ap.add_argument("--forget-file")
    ap.add_argument("--only", type=int, help='Only.')
    ap.add_argument("--smoke", action="store_true", help='Smoke.')
    args = ap.parse_args()
    try:
        rows = candidates(args.family, args.concept)
        if args.only is not None:
            if not 0 <= args.only < len(rows):
                raise ValueError("candidate index outside the search")
            rows = [rows[args.only]]
        if args.smoke:
            for row in rows:
                row.update(steps=2, name=row["name"] + "-smoke")
        commands = [command(row, args.forget_file) for row in rows]
    except ValueError as exc:
        ap.error(str(exc))
    print("# PROPOSED candidates, not submitted. Run anchor p00 first; inspect weight deltas.")
    print("# Select on validation likelihood + held-out retain guardrails, never strict accuracy.")
    print("# Existing names are refused by unlearn.py. Do not run concurrent duplicate sweeps.")
    print("\n".join(commands))


if __name__ == "__main__":
    main()
