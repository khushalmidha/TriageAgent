import os
from pathlib import Path
from typing import List

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = ROOT_DIR / "eval"
RAW_CSV_PATH = DATA_DIR / "twcs.csv"

# Create dirs if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Selected brand for the take-home project
SELECTED_BRAND = "AppleSupport"

# Thread building configs
MIN_THREAD_LENGTH = 2
MAX_THREADS = 5000

# Taxonomy and Routing rules
INTENT_TAXONOMY = [
    "device_issue", "account_access", "billing_subscription",
    "app_store", "connectivity", "update_software",
    "product_inquiry", "service_outage", "feedback_complaint", "other"
]
SENSITIVE_INTENTS = ["billing_subscription", "account_access"]

# API Keys
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-chat")

# Pipeline settings
RETRIEVAL_TOP_K = 3
TOP_K_RETRIEVAL = 3
ESCALATION_THRESHOLD = 0.65
ESCALATION_CONFIDENCE_THRESHOLD = 0.4
ESCALATION_RETRIEVAL_THRESHOLD = 0.3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

def validate_config() -> list[str]:
    issues = []
    if not DEEPSEEK_API_KEY:
        issues.append("DEEPSEEK_API_KEY environment variable is not set")
    
    if not RAW_CSV_PATH.exists():
        issues.append(f"Dataset not found at {RAW_CSV_PATH}. Download from Kaggle.")
        
    return issues
