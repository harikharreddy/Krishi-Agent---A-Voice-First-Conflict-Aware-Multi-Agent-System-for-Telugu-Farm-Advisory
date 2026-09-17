"""
Category A follow-up: novelty claim #3 (confidence-aware responses) --
phrasing fluency/naturalness/correctness rubric, aggregate evidence
generator.

Consumes the raw rating files downloaded from the Phrasing Fluency
Rubric tool (docs/evidence/phrasing_fluency_rubric_tool.html, published
as an Artifact), one JSON file per rater, and produces
docs/evidence/phrasing_fluency_rubric_evidence.json in the same
evidence-file convention as the rest of this project: raw ratings,
method, aggregate scores, inter-rater agreement statistic, honest
caveats.

Does NOT invent or simulate ratings. Refuses to run with zero rater
files rather than produce an empty/fabricated evidence file -- this
script only aggregates real human input collected via the rubric tool.

Usage:
    python3 tests/generate_phrasing_fluency_evidence.py <rating_file1.json> [<rating_file2.json> ...]

    or, to pick up every krishi_agent_phrasing_rubric_*.json file in a
    directory (e.g. wherever the 4 raters' downloaded files were saved):

    python3 tests/generate_phrasing_fluency_evidence.py --dir /path/to/downloaded/ratings
"""

import argparse
import glob
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ITEMS_PATH = os.path.join(ROOT, "docs", "evidence", "phrasing_fluency_rubric_items.json")
OUT_PATH = os.path.join(ROOT, "docs", "evidence", "phrasing_fluency_rubric_evidence.json")

DIMS = ["fluency", "naturalness", "correctness"]
VALID_RATERS = {"Harikha", "Venky", "Amarthya", "Ranga Sarvesh"}


def load_items():
    with open(ITEMS_PATH, encoding="utf-8") as f:
        return json.load(f)["items"]


def load_rater_files(paths):
    raters = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rater = d.get("rater")
        if not rater:
            raise ValueError(f"{p}: missing 'rater' field")
        if rater in raters:
            raise ValueError(
                f"{p}: duplicate submission for rater '{rater}' "
                f"(already loaded from another file) -- resolve which is authoritative before running this script"
            )
        by_item = {r["item_id"]: r for r in d["ratings"]}
        raters[rater] = {"submitted_at": d.get("submitted_at"), "by_item": by_item}
    return raters


def validate(raters, items):
    item_ids = {it["id"] for it in items}
    problems = []
    for rater, data in raters.items():
        missing = item_ids - set(data["by_item"].keys())
        if missing:
            problems.append(f"{rater} is missing ratings for: {sorted(missing)}")
        for item_id, r in data["by_item"].items():
            for dim in DIMS:
                v = r.get(dim)
                if v is None or not (1 <= v <= 5):
                    problems.append(f"{rater}/{item_id}/{dim}: invalid value {v!r} (must be 1-5)")
    if problems:
        raise ValueError("Rating file validation failed:\n  " + "\n  ".join(problems))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def per_item_aggregate(raters, items):
    out = []
    for it in items:
        row = {"item_id": it["id"], "text_telugu": it["text_telugu"], "context": it["context"]}
        for dim in DIMS:
            vals = [raters[r]["by_item"][it["id"]][dim] for r in raters]
            row[dim] = {"values_by_rater": {r: raters[r]["by_item"][it["id"]][dim] for r in raters},
                        "mean": round(mean(vals), 3), "stdev": round(stdev(vals), 3)}
        out.append(row)
    return out


def overall_aggregate(raters, items):
    out = {}
    for dim in DIMS:
        all_vals = [raters[r]["by_item"][it["id"]][dim] for r in raters for it in items]
        out[dim] = {"mean": round(mean(all_vals), 3), "stdev": round(stdev(all_vals), 3), "n_ratings": len(all_vals)}
    grand_all = [v for dim in DIMS for r in raters for it in items for v in [raters[r]["by_item"][it["id"]][dim]]]
    out["grand_mean_all_dimensions"] = round(mean(grand_all), 3)
    return out


def cohens_kappa_linear_weighted(a_vals, b_vals, categories=(1, 2, 3, 4, 5)):
    """Linearly-weighted Cohen's kappa -- appropriate for ordinal 1-5
    ratings (a 1-vs-2 disagreement is penalized less than a 1-vs-5
    disagreement), unlike unweighted kappa which treats every
    disagreement as equally bad. Implemented directly (no external
    dependency beyond what's already used in this project) so the
    formula is auditable inline rather than trusted as a black box."""
    n = len(a_vals)
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    # Weight matrix: linear weights, 0 = perfect disagreement (max distance), 1 = perfect agreement
    def weight(i, j):
        return 1 - abs(i - j) / (k - 1)

    observed = [[0] * k for _ in range(k)]
    for a, b in zip(a_vals, b_vals):
        observed[cat_index[a]][cat_index[b]] += 1

    row_marg = [sum(observed[i]) for i in range(k)]
    col_marg = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    po = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(weight(i, j) * row_marg[i] * col_marg[j] for i in range(k) for j in range(k)) / (n * n)

    if pe == 1:
        return 1.0  # both raters used exactly one category, identically -- trivially perfect, not undefined
    return (po - pe) / (1 - pe)


def percent_agreement(a_vals, b_vals):
    exact = sum(1 for a, b in zip(a_vals, b_vals) if a == b) / len(a_vals)
    within_1 = sum(1 for a, b in zip(a_vals, b_vals) if abs(a - b) <= 1) / len(a_vals)
    return {"exact": round(exact, 3), "within_1_point": round(within_1, 3)}


def fleiss_kappa(rating_matrix, categories=(1, 2, 3, 4, 5)):
    """rating_matrix: list of rows, one per item, each row a list of
    the n_raters category values for that item. Standard Fleiss' kappa
    for >2 raters -- nominal (does not weight by ordinal distance,
    unlike the pairwise linear-weighted kappa above), reported as a
    supplementary, more conservative omnibus statistic."""
    n_items = len(rating_matrix)
    n_raters = len(rating_matrix[0])
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    counts = [[0] * k for _ in range(n_items)]
    for i, row in enumerate(rating_matrix):
        for v in row:
            counts[i][cat_index[v]] += 1

    P_i = [
        (sum(c * c for c in counts[i]) - n_raters) / (n_raters * (n_raters - 1))
        for i in range(n_items)
    ]
    P_bar = mean(P_i)

    p_j = [sum(counts[i][j] for i in range(n_items)) / (n_items * n_raters) for j in range(k)]
    P_e = sum(p * p for p in p_j)

    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def compute_agreement(raters, items):
    rater_names = sorted(raters.keys())
    n_raters = len(rater_names)
    result = {"n_raters": n_raters, "raters": rater_names}

    if n_raters < 2:
        result["note"] = (
            "Only 1 rater submitted -- no inter-rater agreement statistic can be computed "
            "(kappa and percent-agreement both require >=2 raters). Descriptive means above "
            "reflect a single person's judgment only, not yet cross-validated."
        )
        return result

    pairwise = {}
    for r1, r2 in itertools.combinations(rater_names, 2):
        pair_key = f"{r1} vs {r2}"
        pairwise[pair_key] = {}
        for dim in DIMS:
            a_vals = [raters[r1]["by_item"][it["id"]][dim] for it in items]
            b_vals = [raters[r2]["by_item"][it["id"]][dim] for it in items]
            pairwise[pair_key][dim] = {
                "cohens_kappa_linear_weighted": round(cohens_kappa_linear_weighted(a_vals, b_vals), 3),
                "percent_agreement": percent_agreement(a_vals, b_vals),
            }
    result["pairwise"] = pairwise

    if n_raters > 2:
        fleiss = {}
        for dim in DIMS:
            matrix = [[raters[r]["by_item"][it["id"]][dim] for r in rater_names] for it in items]
            fleiss[dim] = round(fleiss_kappa(matrix), 3)
        result["fleiss_kappa_omnibus"] = fleiss
        result["fleiss_kappa_note"] = (
            "Nominal kappa across all raters simultaneously -- does not credit near-misses (a "
            "4-vs-5 disagreement counts the same as a 1-vs-5 one), so this is expected to read "
            "more conservative (lower) than the pairwise linear-weighted kappas above. Reported "
            "as a supplementary omnibus statistic, not the headline number."
        )

    return result


def kappa_interpretation(k):
    if k is None:
        return "n/a"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Rater JSON files")
    parser.add_argument("--dir", help="Directory to glob krishi_agent_phrasing_rubric_*.json from")
    args = parser.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "krishi_agent_phrasing_rubric_*.json")))

    if not paths:
        print(
            "No rater files given. This script refuses to produce an evidence file with zero "
            "real ratings -- pass one or more downloaded rating JSON files, or --dir <folder>.",
            file=sys.stderr,
        )
        sys.exit(1)

    items = load_items()
    raters = load_rater_files(paths)
    validate(raters, items)

    unexpected = set(raters.keys()) - VALID_RATERS
    if unexpected:
        print(f"WARNING: rater name(s) not in the expected set {VALID_RATERS}: {unexpected} -- "
              f"included anyway, but check for a typo in the tool's name field.", file=sys.stderr)

    per_item = per_item_aggregate(raters, items)
    overall = overall_aggregate(raters, items)
    agreement = compute_agreement(raters, items)

    if "pairwise" in agreement:
        for pair, dims in agreement["pairwise"].items():
            for dim, stats in dims.items():
                stats["interpretation"] = kappa_interpretation(stats["cohens_kappa_linear_weighted"])

    evidence = {
        "metric": "Novelty claim #3 (confidence-aware responses) -- phrasing fluency/naturalness/correctness rubric",
        "method": (
            "14 items rated 1-5 on fluency, naturalness, and correctness by team members acting as "
            "raters, via docs/evidence/phrasing_fluency_rubric_tool.html (published as a Claude "
            "Artifact, no live cross-rater data sharing -- each rater's page shows only their own "
            "in-progress ratings, never anyone else's, so blindness is structural, not just "
            "requested). Items 1-11 are the 10 distinct (resolution, confidence) Phrasing Templates "
            "in orchestrator/phrasing_templates.py's TEMPLATES dict plus its unresolved_conflict "
            "fallback entry, verbatim. Items 12-14 extend coverage to the 3 distinct Phase 6.9 "
            "confidence-hedged Weather/Price Agent answer variants (Weather Medium, Price Medium, "
            "Price Low) -- see docs/evidence/phrasing_fluency_rubric_items.json for full item "
            "provenance, including which 2 of the 3 extension items were freshly generated by "
            "calling the real get_weather_advice()/get_price_advice() functions with the network "
            "call mocked at the fetch boundary only (same pattern as tests/test_weather_agent.py "
            "and tests/test_price_agent.py), not authored text."
        ),
        "n_items": len(items),
        "n_raters": len(raters),
        "raters": sorted(raters.keys()),
        "per_item_aggregate": per_item,
        "overall_aggregate": overall,
        "inter_rater_agreement": agreement,
        "honest_caveats": [
            "Small n on both axes -- 14 items, 2-4 raters. Point estimates and kappa values on "
            "this sample size should be read as indicative, not statistically powered in the "
            "sense the project's other n=800+ evaluations are.",
            "Raters are the system's own build team, not naive end users or independent domain "
            "experts -- they know what each item is SUPPOSED to communicate before rating it, "
            "which plausibly inflates 'correctness' scores relative to a farmer hearing the "
            "sentence cold with no context. Fluency/naturalness judgments are less susceptible to "
            "this bias (a team member's ear for natural Telugu doesn't depend on knowing the "
            "pipeline logic) but not immune to it either.",
            "Blindness here means 'raters could not see each other's scores while rating' (enforced "
            "structurally by the tool -- no shared state, no live view of other submissions), not "
            "'raters were blind to which system produced these sentences' -- every rater already "
            "knows this is Krishi-Agent's own phrasing, which the project's own novelty claim #3 is "
            "about. This is a genuine limitation on how independent the ratings really are, stated "
            "plainly rather than glossed over.",
            "The 3 confidence-hedged Weather/Price items (12-14) are single examples of each hedge "
            "variant, not a sample of many -- they show whether the hedge phrase reads naturally "
            "in one realistic sentence, not a distribution across many possible price/weather values.",
        ],
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(raters)} rater(s): {sorted(raters.keys())}")
    print(f"Overall aggregate: {json.dumps(overall, indent=2)}")
    if "pairwise" in agreement:
        for pair, dims in agreement["pairwise"].items():
            print(f"\n{pair}:")
            for dim, stats in dims.items():
                print(f"  {dim}: kappa={stats['cohens_kappa_linear_weighted']} "
                      f"({stats['interpretation']}), exact agreement={stats['percent_agreement']['exact']}, "
                      f"within-1={stats['percent_agreement']['within_1_point']}")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
