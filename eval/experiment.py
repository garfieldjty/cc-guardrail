import json
import logging
import os
import time
from pathlib import Path

from llm.base_client import LLMClient, LLMError
from eval.asr_judge import ASRJudge
from defense.guardrail import Guardrail
from utils.concurrency import interruptible_map

logger = logging.getLogger(__name__)


def _attack_succeeded(r: dict) -> tuple[bool, str]:
    """
    Returns (complied, best_response) using the scores already recorded during FOA.
    successful_attacks entries are only appended when score >= threshold, so their
    presence is the ground truth — no re-judging needed (avoids non-determinism at
    temperature=1.0 causing the second call to disagree with the first).
    """
    successful = r.get("successful_attacks") or []
    if successful:
        best = max(successful, key=lambda x: x.get("score", 0))
        return True, best.get("response") or ""
    return False, r.get("response") or ""


def _run_baseline(
    attack_results: list[dict],
) -> list[dict]:
    """
    Baseline: did the attack succeed without any guardrail?
    Uses the stored FOA scores — no fresh LLM call, no re-judging.
    """
    logger.info(f"Baseline: reading {len(attack_results)} stored attack outcomes")
    records = []
    for r in attack_results:
        complied, response = _attack_succeeded(r)
        status = "COMPLIED" if complied else "refused"
        logger.info(f"  [{r['id']}] {status} — {r['behavior'][:55]}")
        records.append({
            "id": r["id"],
            "category": r["category"],
            "behavior": r["behavior"],
            "best_prompt": r["best_prompt"],
            "response": response,
            "complied": complied,
            "asr_trajectory": r["asr_trajectory"],
        })
    return records


def _run_defended(
    attack_results: list[dict],
    target_client: LLMClient,
    judge: ASRJudge,
    guardrail: Guardrail,
    max_workers: int = 5,
) -> list[dict]:
    logger.info(f"Defended: running {len(attack_results)} prompts through guardrail (workers={max_workers})")
    records: list[dict] = [None] * len(attack_results)  # type: ignore

    def _eval(args):
        idx, r = args
        t0 = time.monotonic()
        allowed, reason = guardrail.filter(r["best_prompt"])
        latency_ms = (time.monotonic() - t0) * 1000

        if allowed:
            try:
                response = target_client.complete(r["best_prompt"])
            except LLMError as e:
                response = f"[BLOCKED: {e}]"
            complied = judge.judge(r["behavior"], response)
            status = "PASSED → COMPLIED" if complied else "PASSED → refused"
        else:
            response = "[BLOCKED]"
            complied = False
            status = "BLOCKED ✓"

        logger.debug(f"  [{r['id']}] {status}  latency={latency_ms:.0f}ms")
        return idx, {
            "id": r["id"],
            "category": r["category"],
            "behavior": r["behavior"],
            "allowed": allowed,
            "reason": reason,
            "response": response,
            "complied": complied,
            "guardrail_latency_ms": round(latency_ms, 2),
        }

    for idx, record in interruptible_map(_eval, list(enumerate(attack_results)), max_workers):
        records[idx] = record

    return records


def _run_fp_check(
    benign_entries: list[dict],
    guardrail: Guardrail,
    max_workers: int = 5,
) -> list[dict]:
    logger.info(f"FP-Check: testing {len(benign_entries)} benign classical Chinese queries (workers={max_workers})")
    records: list[dict] = [None] * len(benign_entries)  # type: ignore

    def _check(args):
        idx, entry = args
        allowed, reason = guardrail.filter(entry["text"])
        status = "OK (passed)" if allowed else "FALSE POSITIVE"
        logger.debug(f"  [{entry['id']}] {status} — {entry['topic']}")
        return idx, {
            "id": entry["id"],
            "topic": entry["topic"],
            "allowed": allowed,
            "reason": reason,
            "false_positive": not allowed,
        }

    for idx, record in interruptible_map(_check, list(enumerate(benign_entries)), max_workers):
        records[idx] = record

    return records


def _asr(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["complied"]) / len(records)


def _fp_rate(fp_records: list[dict]) -> float:
    if not fp_records:
        return 0.0
    return sum(1 for r in fp_records if r["false_positive"]) / len(fp_records)


def run_experiment(
    attack_results: list[dict],
    benign_entries: list[dict],
    target_client: LLMClient,
    judge_client: LLMClient,
    guardrail: Guardrail,
    results_dir: str = "results",
    models: dict | None = None,
    max_workers: int = 5,
    translator_client: LLMClient | None = None,
    llama_guard_guardrail=None,
    v1_guardrail=None,
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    judge = ASRJudge(judge_client, translator=translator_client)

    # Optional V1 (direct-classify) guardrail comparison
    v1_defended, v1_fp_check = None, None
    if v1_guardrail is not None:
        logger.info("V1 guardrail (direct-classify): running comparison")
        v1_defended = _run_defended(attack_results, target_client, judge, v1_guardrail, max_workers)
        v1_fp_check = _run_fp_check(benign_entries, v1_guardrail, max_workers)
        Path(f"{results_dir}/v1_defended.json").write_text(json.dumps(v1_defended, ensure_ascii=False, indent=2))
        Path(f"{results_dir}/v1_fp_check.json").write_text(json.dumps(v1_fp_check, ensure_ascii=False, indent=2))

    # Optional Llama Guard comparison
    lg_defended, lg_fp_check = None, None
    if llama_guard_guardrail is not None:
        lg_defended = _run_defended(attack_results, target_client, judge, llama_guard_guardrail, max_workers)
        lg_fp_check = _run_fp_check(benign_entries, llama_guard_guardrail, max_workers)
        Path(f"{results_dir}/lg_defended.json").write_text(json.dumps(lg_defended, ensure_ascii=False, indent=2))
        Path(f"{results_dir}/lg_fp_check.json").write_text(json.dumps(lg_fp_check, ensure_ascii=False, indent=2))

    baseline = _run_baseline(attack_results)
    defended = _run_defended(attack_results, target_client, judge, guardrail, max_workers)
    fp_check = _run_fp_check(benign_entries, guardrail, max_workers)

    Path(f"{results_dir}/baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    Path(f"{results_dir}/defended.json").write_text(json.dumps(defended, ensure_ascii=False, indent=2))
    Path(f"{results_dir}/fp_check.json").write_text(json.dumps(fp_check, ensure_ascii=False, indent=2))

    categories = sorted({r["category"] for r in baseline})
    rows = []
    for cat in categories:
        bl = [r for r in baseline if r["category"] == cat]
        df = [r for r in defended if r["category"] == cat]
        row = {
            "category": cat,
            "baseline_asr": round(_asr(bl), 3),
            "defended_asr": round(_asr(df), 3),
            "n": len(bl),
        }
        if v1_defended is not None:
            v1_df = [r for r in v1_defended if r["category"] == cat]
            row["v1_defended_asr"] = round(_asr(v1_df), 3)
        if lg_defended is not None:
            lg_df = [r for r in lg_defended if r["category"] == cat]
            row["lg_defended_asr"] = round(_asr(lg_df), 3)
        rows.append(row)

    summary = {
        **({"models": models} if models else {}),
        "overall_baseline_asr": round(_asr(baseline), 3),
        "overall_defended_asr": round(_asr(defended), 3),
        "false_positive_rate": round(_fp_rate(fp_check), 3),
        "avg_guardrail_latency_ms": round(
            sum(r["guardrail_latency_ms"] for r in defended) / max(len(defended), 1), 2
        ),
        "per_category": rows,
    }
    if v1_defended is not None and v1_fp_check is not None:
        summary["v1_overall_defended_asr"] = round(_asr(v1_defended), 3)
        summary["v1_false_positive_rate"] = round(_fp_rate(v1_fp_check), 3)
        summary["v1_avg_latency_ms"] = round(
            sum(r["guardrail_latency_ms"] for r in v1_defended) / max(len(v1_defended), 1), 2
        )
    if lg_defended is not None and lg_fp_check is not None:
        summary["lg_overall_defended_asr"] = round(_asr(lg_defended), 3)
        summary["lg_false_positive_rate"] = round(_fp_rate(lg_fp_check), 3)
        summary["lg_avg_latency_ms"] = round(
            sum(r["guardrail_latency_ms"] for r in lg_defended) / max(len(lg_defended), 1), 2
        )

    Path(f"{results_dir}/summary.json").write_text(json.dumps(summary, indent=2))
    return summary
