"""Lightweight vs pro tier configs.

Both tiers answer the SAME request and return the SAME response shape
(API_IO_SPEC.md) — only the model behind the rankings differs, so a client
switches tiers without touching its parsing code.

    python pipeline/tier.py --check          # assert tier parity + model files exist
    python pipeline/tier.py --show pro       # dump one tier's config
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).parent
CONFIG_DIR = BASE.parent / "configs"
TIERS = ("lightweight", "pro")


def load_tier(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"unknown tier '{name}' — no config at {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("tier") != name:
        raise SystemExit(f"{path.name} declares tier '{cfg.get('tier')}', expected '{name}'")
    return cfg


def check() -> int:
    """Fail loudly if a tier drifts from the shared contract or points nowhere."""
    cfgs = {t: load_tier(t) for t in TIERS}
    failures = []

    contracts = {t: json.dumps(c["response"], sort_keys=True) for t, c in cfgs.items()}
    if len(set(contracts.values())) != 1:
        failures.append("tiers declare different response shapes — clients cannot switch tiers")

    feats = {t: c["features"] for t, c in cfgs.items()}
    if len({json.dumps(f) for f in feats.values()}) != 1:
        failures.append("tiers declare different feature vectors")

    for name, c in cfgs.items():
        if c.get("servable"):
            model = BASE / c["model_path"]
            if not model.exists():
                failures.append(f"{name}: servable but model missing at {model}")
        elif not c.get("blocked_on"):
            failures.append(f"{name}: not servable but no `blocked_on` reason given")
        if "ndcg@10" not in c.get("sota", {}):
            failures.append(f"{name}: no sota.ndcg@10 recorded")

    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        return 1
    for name, c in cfgs.items():
        s = c["sota"]
        state = "servable" if c.get("servable") else "NOT servable"
        print(f"  {name:<12} NDCG@10 {s['ndcg@10']:.4f}  P@5 {s['precision@5']:.4f}  "
              f"MRR {s['mrr']:.4f}  [{state}]")
    print("  parity OK: identical response contract and feature vector across tiers")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--show", choices=TIERS)
    a = ap.parse_args()
    if a.show:
        print(json.dumps(load_tier(a.show), indent=2, ensure_ascii=False))
    else:
        raise SystemExit(check())


if __name__ == "__main__":
    main()
