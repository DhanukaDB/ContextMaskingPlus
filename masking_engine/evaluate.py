"""
evaluate.py — Evaluation: Precision, Recall, F1 per Entity Type
R26-CS-012: Context-Aware Masking + Instruction Engine

Runs the engine against the synthetic dataset (synthetic_dataset.json)
and computes per-entity-type and overall:
  - Precision = TP / (TP + FP)   — of what we masked, how much was truly sensitive
  - Recall    = TP / (TP + FN)   — of what was sensitive, how much did we catch
  - F1        = 2 * P * R / (P + R)

Also reports:
  - Masking action distribution (mask_immediate / mask_warn / log_suspected / ignore)
  - Adversarial detection rate
  - Co-occurrence elevation accuracy
  - False positive rate for edge/ambiguity cases
"""

import sys
import os
import json
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.normalizer import normalize
from engine.detector import detect
from engine.confidence_scorer import score_all, resolve_overlapping_entities
from engine.masker import mask
from engine.token_registry import TokenRegistry
from engine.ml_anomaly import apply_safety_net, is_available as ml_layer_available, get_ml_flag


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "synthetic_dataset.json")


def load_dataset():
    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Run: python data/generate_dataset.py  to create it first.")
        sys.exit(1)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# EVALUATION LOGIC
# ─────────────────────────────────────────────

def evaluate():
    dataset  = load_dataset()
    registry = TokenRegistry(session_id="eval_session")

    # Per-entity-type counters
    tp_counts = defaultdict(int)   # detected AND in ground truth
    fp_counts = defaultdict(int)   # detected BUT not in ground truth
    fn_counts = defaultdict(int)   # in ground truth BUT not detected

    # Aggregate action distribution
    action_dist = defaultdict(int)

    # Adversarial & edge tracking
    adversarial_total    = 0
    adversarial_detected = 0
    edge_fp_count        = 0   # edge cases that should NOT be masked but were
    edge_total           = 0

    # ML safety-net (Layer 2) tracking — only ever runs when Layer 1 found
    # nothing at all (see engine/ml_anomaly.py). Split by whether ground
    # truth for that prompt was actually non-empty: a flag on a prompt with
    # real ground-truth entities is Layer 2 catching a genuine Layer 1 miss
    # (a recall win); a flag on a prompt with EMPTY ground truth is a false
    # alarm (a precision/review-burden cost). Both numbers matter for an
    # honest "how much does this layer actually help" read.
    ml_layer_on            = ml_layer_available()
    ml_flags_total         = 0
    ml_flags_true_miss     = 0   # flagged AND ground_truth non-empty
    ml_flags_false_alarm   = 0   # flagged AND ground_truth empty
    ml_masks_applied       = 0   # of ml_flags_total, how many located+masked a span
                                  # (vs. falling back to a review-only flag)

    results_rows = []

    for record in dataset:
        prompt_id    = record["id"]
        text         = record["prompt"]
        prompt_type  = record["type"]
        ground_truth = set(record.get("entities", []))

        registry.next_prompt()

        # Run engine
        norm            = normalize(text)
        raw_entities    = detect(norm["normalized"], norm["despaced"], norm["despaced_map"])
        scored_entities = score_all(raw_entities, norm["normalized"])
        scored_entities = resolve_overlapping_entities(scored_entities)
        masked_result   = mask(norm["normalized"], scored_entities, registry)

        apply_safety_net(norm["normalized"], masked_result, registry)
        ml_flag = get_ml_flag(masked_result)
        if ml_flag is not None:
            ml_flags_total += 1
            if "replacement" in ml_flag:
                ml_masks_applied += 1
            if ground_truth:
                ml_flags_true_miss += 1
            else:
                ml_flags_false_alarm += 1

        # Collect detected & actually masked entity types. ML_FLAGGED_ANOMALY
        # is excluded here — it's not a taxonomy entity type (ground truth
        # never contains it), so counting it in the per-type TP/FP/FN loop
        # below would misreport every Layer-2 catch as a spurious FP under
        # a fake type. It's tracked separately via ml_flags_* above instead.
        detected_masked = {
            m["entity_type"]
            for m in masked_result.masked_entities
            if m["entity_type"] != "ML_FLAGGED_ANOMALY"
        }
        detected_logged = {
            sk["entity_type"]
            for sk in masked_result.skipped_entities
            if sk["reason"] == "low_confidence_logged"
        }
        all_detected = detected_masked | detected_logged

        # Action distribution
        for s in scored_entities:
            action_dist[s.action] += 1

        # TP / FP / FN
        for et in all_detected:
            if et in ground_truth:
                tp_counts[et] += 1
            else:
                fp_counts[et] += 1

        for et in ground_truth:
            if et not in all_detected:
                fn_counts[et] += 1

        # Adversarial tracking
        if prompt_type == "adversarial":
            adversarial_total += 1
            if all_detected:  # any entity caught
                adversarial_detected += 1

        # Edge/false-positive tracking
        if prompt_type == "edge":
            edge_total += 1
            note = record.get("note", "")
            if "should NOT" in note and detected_masked:
                edge_fp_count += 1

        results_rows.append({
            "id"              : prompt_id,
            "type"            : prompt_type,
            "ground_truth"    : "|".join(sorted(ground_truth)),
            "detected_masked" : "|".join(sorted(detected_masked)),
            "detected_logged" : "|".join(sorted(detected_logged)),
            "overall_risk"    : masked_result.overall_risk,
            "tp"              : len(all_detected & ground_truth),
            "fp"              : len(all_detected - ground_truth),
            "fn"              : len(ground_truth - all_detected),
        })

    # ─────────────────────────────────────────
    # COMPUTE METRICS
    # ─────────────────────────────────────────

    all_entity_types = set(tp_counts) | set(fp_counts) | set(fn_counts)

    per_type_metrics = {}
    for et in sorted(all_entity_types):
        tp  = tp_counts[et]
        fp  = fp_counts[et]
        fn  = fn_counts[et]
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_type_metrics[et] = {"tp": tp, "fp": fp, "fn": fn,
                                 "precision": p, "recall": r, "f1": f1}

    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    overall_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    adv_rate = (adversarial_detected / adversarial_total * 100) if adversarial_total > 0 else 0
    edge_fpr = (edge_fp_count / edge_total * 100) if edge_total > 0 else 0

    # ─────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────

    print(f"\n{'═'*70}")
    print(f"  R26-CS-012 — EVALUATION REPORT")
    print(f"  Dataset: {len(dataset)} prompts "
          f"({sum(1 for r in dataset if r['type']=='normal')} normal | "
          f"{sum(1 for r in dataset if r['type']=='adversarial')} adversarial | "
          f"{sum(1 for r in dataset if r['type']=='edge')} edge)")
    print(f"{'═'*70}")

    print(f"\n{'─'*70}")
    print(f"  PER-ENTITY-TYPE METRICS")
    print(f"{'─'*70}")
    print(f"  {'Entity Type':<25} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*25} {'─'*4} {'─'*4} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    for et, m in per_type_metrics.items():
        print(f"  {et:<25} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"  {m['precision']:>8.2%} {m['recall']:>7.2%} {m['f1']:>7.2%}")

    print(f"{'─'*70}")
    print(f"  {'OVERALL':<25} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
          f"  {overall_p:>8.2%} {overall_r:>7.2%} {overall_f1:>7.2%}")
    print(f"{'─'*70}")

    print(f"\n  MASKING ACTION DISTRIBUTION")
    for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 3)
        print(f"  {action:<20} {count:>4}  {bar}")

    print(f"\n  ADVERSARIAL DETECTION RATE : {adversarial_detected}/{adversarial_total} "
          f"({adv_rate:.1f}%)")
    print(f"  EDGE FALSE POSITIVE RATE   : {edge_fp_count}/{edge_total} "
          f"({edge_fpr:.1f}%) — lower is better")

    print(f"\n  ML SAFETY NET (Layer 2)    : {'available' if ml_layer_on else 'NOT AVAILABLE (skipped)'}")
    if ml_layer_on:
        print(f"    Flagged (total)          : {ml_flags_total} / {len(dataset)} prompts")
        print(f"      Masked (span located)  : {ml_masks_applied}")
        print(f"      Review-only (no span)  : {ml_flags_total - ml_masks_applied}")
        print(f"    True misses caught       : {ml_flags_true_miss}  (Layer 1 found nothing, ground truth non-empty)")
        print(f"    False alarms             : {ml_flags_false_alarm}  (Layer 1 found nothing, ground truth empty)")

    print(f"\n{'═'*70}\n")

    # ─────────────────────────────────────────
    # SAVE RESULTS CSV
    # ─────────────────────────────────────────

    results_path = os.path.join(os.path.dirname(__file__), "data", "eval_results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"  Detailed results saved → {results_path}\n")

    # Save per-type metrics JSON
    metrics_path = os.path.join(os.path.dirname(__file__), "data", "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": {"precision": overall_p, "recall": overall_r, "f1": overall_f1,
                        "tp": total_tp, "fp": total_fp, "fn": total_fn},
            "per_entity_type": per_type_metrics,
            "adversarial_detection_rate": adv_rate,
            "edge_false_positive_rate": edge_fpr,
            "action_distribution": dict(action_dist),
            "ml_safety_net": {
                "available": ml_layer_on,
                "flags_total": ml_flags_total,
                "masks_applied": ml_masks_applied,
                "flags_review_only": ml_flags_total - ml_masks_applied,
                "flags_true_miss": ml_flags_true_miss,
                "flags_false_alarm": ml_flags_false_alarm,
            },
        }, f, indent=2)
    print(f"  Metrics JSON saved       → {metrics_path}\n")


if __name__ == "__main__":
    evaluate()
