from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit(default: str = "nogit") -> str:

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return default


def config_hash(config: dict[str, Any]) -> str:

    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _sampling_key(row) -> str:

    import json as _json

    return _json.dumps(row.get("sampling"), sort_keys=True)


def select_one_config(rows: list, chosen: str | None = None, *, what: str = "generations") -> list:


    if chosen:
        kept = [r for r in rows if r.get("config_hash") == chosen]
        if not kept:
            present = sorted({r.get("config_hash") for r in rows if r.get("config_hash")})
            raise SystemExit(f"--config-hash {chosen} matches no {what} (present: {present})")
        return kept

    per_model: dict[str, dict[str, int]] = {}
    for r in rows:
        per_model.setdefault(r.get("model"), {}).setdefault(_sampling_key(r), 0)
        per_model[r.get("model")][_sampling_key(r)] += 1
    mixed = {m: s for m, s in per_model.items() if len(s) > 1}
    if mixed:
        detail = "\n".join(f"    model={m}\n" + "\n".join(f"      sampling={k}  ->  {n} rows"
                                                          for k, n in sorted(s.items()))
                           for m, s in sorted(mixed.items(), key=lambda kv: str(kv[0])))
        raise SystemExit(
            f"these {what} give a model MORE THAN ONE sampling regime:\n{detail}\n"
            "  `sampling` is not a grouping key, so those rows would be summed into one cell and\n"
            "  quietly averaged — two experiments reported as one.\n"
            "  Fix: move the superseded rows out of the generations dir (e.g. results/<run>/\n"
            "  _superseded_<cap>/), or pass --config-hash <hash> to pick one.\n"
            "  (Multiple MODELS are fine — every table groups by model.)")
    return rows
