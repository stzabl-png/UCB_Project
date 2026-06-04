#!/usr/bin/env python3
"""Per-object + per-category success rate for the DP3 A/B-diversity 116-object eval.

Reads per_ep_results.tar.gz DIRECTLY (no extraction needed). Group by obj_id (default coarse
category = id prefix), or pass --categories obj_id,category CSV for your semantic categories.

Usage:
    python compute_sr.py
    python compute_sr.py --categories my_map.csv
    python compute_sr.py --tar per_ep_results.tar.gz
"""
import argparse, csv, io, json, tarfile, collections, os


def coarse_category(obj_id: str) -> str:
    if obj_id.startswith("unseen"):
        return "unseen"
    if obj_id.startswith("ycb") or obj_id.startswith("Y"):
        return "dexycb"
    return "oakink"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=os.path.join(here, "per_ep_results.tar.gz"))
    ap.add_argument("--categories", default=None,
                    help="optional CSV 'obj_id,category' for semantic categories")
    args = ap.parse_args()

    cat_map = {}
    if args.categories:
        with open(args.categories) as f:
            for row in csv.DictReader(f):
                cat_map[row["obj_id"]] = row["category"]

    by_obj = collections.defaultdict(lambda: [0, 0])      # obj -> [n_success, n_total]
    with tarfile.open(args.tar, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".json"):
                continue
            d = json.load(io.TextIOWrapper(tf.extractfile(m)))
            obj = d.get("obj_id") or os.path.basename(m.name).split("_dp3")[0]
            by_obj[obj][0] += int(bool(d.get("success")))
            by_obj[obj][1] += 1

    n_s = sum(v[0] for v in by_obj.values())
    n_t = sum(v[1] for v in by_obj.values())
    print(f"OVERALL: {n_s}/{n_t} = {100*n_s/max(n_t,1):.2f}%  ({len(by_obj)} objects)\n")

    by_cat = collections.defaultdict(lambda: [0, 0])
    for o, (s, t) in by_obj.items():
        c = cat_map.get(o, coarse_category(o))
        by_cat[c][0] += s
        by_cat[c][1] += t
    label = "your category" if cat_map else "coarse category (id prefix)"
    print(f"=== SR per {label} ===")
    for c in sorted(by_cat):
        s, t = by_cat[c]
        print(f"  {c:>16}: {s:>4}/{t:<4} = {100*s/t:5.1f}%  ({t//40} objects)")

    print("\n=== per-object SR (sorted) ===")
    for o in sorted(by_obj, key=lambda x: -by_obj[x][0] / by_obj[x][1]):
        s, t = by_obj[o]
        print(f"  {o:>14} [{cat_map.get(o, coarse_category(o)):>7}]: {s:>2}/{t} = {100*s/t:5.1f}%")


if __name__ == "__main__":
    main()
