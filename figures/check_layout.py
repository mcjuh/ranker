"""
Geometric QA for the workflow figure. "Ugly" is usually concrete:
  - two boxes partially overlapping (neither containing the other)
  - a container growing into a neighbouring band
  - words colliding with other words
This reports those, so layout is fixed by measurement rather than by guessing.
"""
import sys

import pymupdf

doc = pymupdf.open("/tmp/tw/senseigigs_workflow.pdf")
pg = doc[0]
mm = lambda v: round(v / 72 * 25.4, 1)  # noqa: E731

# ---- filled shapes big enough to be a box -------------------------------------
rects = []
for d in pg.get_drawings():
    r = d["rect"]
    if r.width > 12 and r.height > 7:
        rects.append(r)

def contains(a, b, tol=1.0):
    return (a.x0 - tol <= b.x0 and a.y0 - tol <= b.y0
            and a.x1 + tol >= b.x1 and a.y1 + tol >= b.y1)

# dedupe near-identical rects
uniq = []
for r in rects:
    if not any(abs(r.x0 - u.x0) < 0.6 and abs(r.y0 - u.y0) < 0.6
               and abs(r.x1 - u.x1) < 0.6 and abs(r.y1 - u.y1) < 0.6 for u in uniq):
        uniq.append(r)
rects = uniq

print(f"  boxes found: {len(rects)}")
bad = []
for i in range(len(rects)):
    for j in range(i + 1, len(rects)):
        a, b = rects[i], rects[j]
        inter = a & b
        if inter.is_empty or inter.width <= 0.7 or inter.height <= 0.7:
            continue
        if contains(a, b) or contains(b, a):
            continue                       # nesting is fine
        bad.append((a, b, inter))
print(f"  PARTIAL BOX OVERLAPS (neither contains the other): {len(bad)}")
for a, b, it in bad[:8]:
    print(f"    x{mm(a.x0)}..{mm(a.x1)} y{mm(a.y0)}..{mm(a.y1)}  vs  "
          f"x{mm(b.x0)}..{mm(b.x1)} y{mm(b.y0)}..{mm(b.y1)}   overlap {mm(it.width)}x{mm(it.height)} mm")

# ---- word collisions ----------------------------------------------------------
words = []
for blk in pg.get_text("dict")["blocks"]:
    for ln in blk.get("lines", []):
        for sp in ln["spans"]:
            if sp["text"].strip():
                words.append((pymupdf.Rect(sp["bbox"]), sp["text"].strip()))
coll = []
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        ri, ti = words[i]
        rj, tj = words[j]
        it = ri & rj
        if it.is_empty or it.width * it.height <= 2.0:
            continue
        coll.append((ti, tj, it.width * it.height))
print(f"  WORD COLLISIONS: {len(coll)}")
for ti, tj, ar in coll[:8]:
    print(f"    {ti!r} x {tj!r}  area {ar:.1f} pt2")

if not bad and not coll:
    print("\n  RESULT: clean - no partial box overlaps, no word collisions")
