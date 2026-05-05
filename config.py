import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")

# --- Provider & Model IDs ---
# "cloudflare" | "anthropic" | "openai" | "openrouter" | "lmstudio"
PROVIDER = os.getenv("PROVIDER", "openrouter")
TARGET_MODEL = os.getenv("TARGET_MODEL", "openai/gpt-oss-20b")         # victim model
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")          # ASR judge/evaluator
BUILDER_MODEL = os.getenv("BUILDER_MODEL", "deepseek/deepseek-v3.2")  # CC rendering + translation
LLAMA_GUARD_MODEL = os.getenv("LLAMA_GUARD_MODEL", "meta-llama/llama-guard-4-12b")

# TARGET_PROVIDER allows the victim to use a different provider (e.g. lmstudio locally)
# while judge/builder continue using PROVIDER (e.g. openrouter).
TARGET_PROVIDER = os.getenv("TARGET_PROVIDER", PROVIDER)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

# --- Concurrency ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))  # parallel LLM calls within each phase

# --- FOA hyperparameters ---
FOA_POPULATION = int(os.getenv("FOA_POPULATION", "5"))
FOA_SURVIVORS = int(os.getenv("FOA_SURVIVORS", "2"))
FOA_ITERATIONS = int(os.getenv("FOA_ITERATIONS", "2"))

# --- Experiment ---
BEHAVIORS_PATH = os.getenv("BEHAVIORS_PATH", "data/behaviors.json")
BENIGN_CC_PATH = os.getenv("BENIGN_CC_PATH", "data/benign_cc.json")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "data/cc_templates.json")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
