import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from llm import CloudflareClient, AnthropicClient, OpenAIClient, OpenRouterClient, LMStudioClient, LLMClient
from attack.prompt_builder import PromptBuilder
from attack.foa_optimizer import FOAOptimizer
from attack.attacker import Attacker
from defense.intent_judge import IntentJudge, DirectIntentJudge
from defense.guardrail import Guardrail
from defense.llama_guard import LlamaGuardGuardrail
from eval.experiment import run_experiment

logger = logging.getLogger(__name__)


def _configure_logging():
    """Configure global logging for the entire application."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt='[%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    
    # Suppress verbose library loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)


def _build_client(model_id: str, provider: str | None = None) -> LLMClient:
    provider = provider or config.PROVIDER
    if provider == "cloudflare":
        return CloudflareClient(
            account_id=config.CF_ACCOUNT_ID,
            api_token=config.CF_API_TOKEN,
            model_id=model_id,
        )
    if provider == "anthropic":
        return AnthropicClient(api_key=config.ANTHROPIC_API_KEY, model_id=model_id)
    if provider == "openai":
        return OpenAIClient(api_key=config.OPENAI_API_KEY, model_id=model_id)
    if provider == "openrouter":
        return OpenRouterClient(api_key=config.OPENROUTER_API_KEY, model_id=model_id)
    if provider == "lmstudio":
        return LMStudioClient(model_id=model_id, base_url=config.LMSTUDIO_BASE_URL)
    sys.exit(f"Unknown provider: {provider}")


def _model_metadata() -> dict:
    return {
        "provider": config.PROVIDER,
        "target_provider": config.TARGET_PROVIDER,
        "target_model": config.TARGET_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "builder_model": config.BUILDER_MODEL,
        "llama_guard_model": config.LLAMA_GUARD_MODEL,
        "foa": {
            "population": config.FOA_POPULATION,
            "survivors": config.FOA_SURVIVORS,
            "iterations": config.FOA_ITERATIONS,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def cmd_attack(args) -> list[dict]:
    target = _build_client(config.TARGET_MODEL, provider=config.TARGET_PROVIDER)
    judge = _build_client(config.JUDGE_MODEL)
    builder_client = _build_client(config.BUILDER_MODEL)
    builder = PromptBuilder(builder_client, templates_path=config.TEMPLATES_PATH,
                            max_workers=config.MAX_WORKERS)
    optimizer = FOAOptimizer(
        templates=json.loads(Path(config.TEMPLATES_PATH).read_text()),
        population_size=config.FOA_POPULATION,
        top_k=config.FOA_SURVIVORS,
        iterations=config.FOA_ITERATIONS,
        max_workers=config.MAX_WORKERS,
    )
    attacker = Attacker(target, judge, builder, optimizer, max_workers=config.MAX_WORKERS,
                        translator_client=builder_client)

    behaviors = json.loads(Path(config.BEHAVIORS_PATH).read_text())
    if args.limit:
        behaviors = behaviors[: args.limit]

    logger.info(f"Running CC-BOS attack on {len(behaviors)} behaviors [target={config.TARGET_MODEL}]")
    results = attacker.run_all(behaviors)

    out = Path(config.RESULTS_DIR) / "attack_results.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "metadata": _model_metadata(),
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info(f"Attack results saved to {out}")
    return results


def cmd_eval(args):
    attack_file = Path(config.RESULTS_DIR) / "attack_results.json"
    if not attack_file.exists():
        sys.exit("Run --mode attack first to generate attack_results.json")

    payload = json.loads(attack_file.read_text())
    # Support both wrapped {"metadata":..,"results":[..]} and legacy plain list
    if isinstance(payload, list):
        attack_results = payload
    else:
        attack_results = payload["results"]

    benign_entries = json.loads(Path(config.BENIGN_CC_PATH).read_text())

    target = _build_client(config.TARGET_MODEL, provider=config.TARGET_PROVIDER)
    judge_client = _build_client(config.JUDGE_MODEL)
    builder_client = _build_client(config.BUILDER_MODEL)
    intent_judge = IntentJudge(builder_client)
    guardrail = Guardrail(intent_judge)
    v1_judge = DirectIntentJudge(builder_client)
    v1_guardrail = Guardrail(v1_judge)
    lg_client = _build_client(config.LLAMA_GUARD_MODEL, provider="openrouter")
    llama_guard = LlamaGuardGuardrail(lg_client)

    logger.info(f"Running experiment: target={config.TARGET_MODEL}, defender={config.BUILDER_MODEL}, llama_guard={config.LLAMA_GUARD_MODEL}")
    summary = run_experiment(
        attack_results=attack_results,
        benign_entries=benign_entries,
        target_client=target,
        judge_client=judge_client,
        guardrail=guardrail,
        results_dir=config.RESULTS_DIR,
        models=_model_metadata(),
        max_workers=config.MAX_WORKERS,
        translator_client=builder_client,
        llama_guard_guardrail=llama_guard,
        v1_guardrail=v1_guardrail,
    )

    logger.info("=== Summary ===")
    logger.info(f"Target model:       {summary['models']['target_model']}")
    logger.info(f"Judge model:        {summary['models']['judge_model']}")
    logger.info(f"Baseline ASR:       {summary['overall_baseline_asr']:.1%}")
    logger.info(f"V2 (translate+judge) ASR: {summary['overall_defended_asr']:.1%}  (FP={summary['false_positive_rate']:.1%}  latency={summary['avg_guardrail_latency_ms']:.0f}ms)")
    if "v1_overall_defended_asr" in summary:
        logger.info(f"V1 (direct-classify) ASR: {summary['v1_overall_defended_asr']:.1%}  (FP={summary['v1_false_positive_rate']:.1%}  latency={summary['v1_avg_latency_ms']:.0f}ms)")
    if "lg_overall_defended_asr" in summary:
        logger.info(f"LlamaGuard ASR:     {summary['lg_overall_defended_asr']:.1%}  (FP={summary['lg_false_positive_rate']:.1%}  latency={summary['lg_avg_latency_ms']:.0f}ms)")
    logger.info("Per-category breakdown:")
    for row in summary["per_category"]:
        v1_col = f"  v1={row['v1_defended_asr']:.0%}" if "v1_defended_asr" in row else ""
        lg_col = f"  llama_guard={row['lg_defended_asr']:.0%}" if "lg_defended_asr" in row else ""
        logger.info(f"  {row['category']:20s}  baseline={row['baseline_asr']:.0%}  v2={row['defended_asr']:.0%}{v1_col}{lg_col}")


def main():
    _configure_logging()
    
    parser = argparse.ArgumentParser(description="CC-BOS attack & guardrail evaluation")
    parser.add_argument("--mode", choices=["attack", "eval", "all"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of behaviors")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without calling APIs")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry-run: validating config and data files...")
        for path in (config.BEHAVIORS_PATH, config.BENIGN_CC_PATH, config.TEMPLATES_PATH):
            assert Path(path).exists(), f"Missing: {path}"
        assert config.CF_ACCOUNT_ID or config.ANTHROPIC_API_KEY or \
               config.OPENAI_API_KEY or config.OPENROUTER_API_KEY, \
               "No API credentials found in .env"
        meta = _model_metadata()
        logger.info(f"  provider: {meta['provider']}")
        logger.info(f"  target_provider: {meta['target_provider']}")
        logger.info(f"  target_model: {meta['target_model']}")
        logger.info(f"  judge_model: {meta['judge_model']}")
        logger.info(f"  builder_model: {meta['builder_model']}")
        logger.info("Configuration validated OK")
        return

    if args.mode in ("attack", "all"):
        cmd_attack(args)
    if args.mode in ("eval", "all"):
        cmd_eval(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Aborted by user")
        sys.exit(1)
