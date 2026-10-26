"""
Render the recommendation output the way the system would present it.

Uses data_pilot because every candidate there carries a judged grade, so each
recommendation can be annotated with what the grader actually thought of it.
"""
import json
import textwrap
from pathlib import Path

BASE = Path(__file__).parent
TAG = "data_pilot"

hirers = {h["hire_id"]: h for h in json.loads((BASE / TAG / "hirers.json").read_text())}
provs = {int(p["provider_id"]): p for p in json.loads((BASE / TAG / "providers.json").read_text())}
gt = json.loads((BASE / TAG / "ground_truth_llm.json").read_text())
rrf = json.loads((BASE / "results_data_pilot/rrf_k60.json").read_text())
ltr = json.loads((BASE / "results_data_pilot/pilot_tax.json").read_text())

LABEL = {0: "0  not relevant", 33: "1  weak match", 67: "2  good match", 100: "3  excellent match"}


def wrap(t, w=88, ind="    "):
    return "\n".join(textwrap.wrap(str(t or ""), w, initial_indent=ind, subsequent_indent=ind))


def show(hid, title):
    h, gt_row = hirers[hid], gt.get(hid, {})
    print("=" * 96)
    print(f"GIG  {hid}   {h['hire_title']}")
    print("=" * 96)
    print(wrap(h["hire_description"]))
    print(f"    durations: {h.get('duration_weeks_min')}-{h.get('duration_weeks_max')} weeks")
    top = h.get("tags") or []
    if top:
        spec = [s["name"] for c in top for s in c.get("specialisations", [])]
        print(f"    taxonomy: {top[0]['name']} > {', '.join(spec[:4])}")

    print(f"\n  {title}")
    print("  " + "-" * 94)
    print(f"  {'#':<3}{'provider':<46}{'RRF':>6}{'LTR':>6}   judge")
    for i, pid in enumerate(ltr[hid][:5], start=1):
        p = provs[pid]
        g = gt_row.get(str(pid))
        mark = "-" if g is None else LABEL.get(g, str(g))
        hit = "  <-- relevant" if g is not None and g >= 40 else ""
        print(f"  {i:<3}{str(p['about_title'])[:44]:<46}"
              f"{rrf[hid].index(pid)+1:>6}{i:>6}   {mark}{hit}")
        print(wrap(str(p["about_description"])[:150], 88, "       "))
    n_rel = sum(1 for v in gt_row.values() if v >= 40)
    n_rel_in5 = sum(1 for pid in ltr[hid][:5] if gt_row.get(str(pid), 0) >= 40)
    print(f"\n    P@5 = {n_rel_in5}/5 = {n_rel_in5/5:.2f}    "
          f"good matches known for this gig: {n_rel}")
    print()


show("G010", "TOP-5 RECOMMENDATIONS (a gig the system gets right)")
show("G006", "TOP-5 RECOMMENDATIONS (a gig with no good match in the pool)")
