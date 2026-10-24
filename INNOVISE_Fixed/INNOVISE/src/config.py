import os

TOLERANCE_PP = 0.005  # absolute, for growth rates, margins and ratios
REL_TOLERANCE = 0.005  # relative, for dollar amounts
STALE_MONTHS = 12
SIM_THRESHOLD = 0.35  # placeholder, tuned later
TOP_K = 3
REPORT_PERIOD = 'FY2024'
REPORT_PERIOD_END = '2024-12-31'
DE_RISE = 0.05  # debt-to-equity rise that triggers a risk flag
MARGIN_DROP = 0.005  # margin fall that triggers a risk flag
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:4b')
LLM_BACKEND = os.getenv('LLM_BACKEND', 'ollama')  # 'ollama' or 'cache'

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, 'data')
DOCS_DIR = os.path.join(DATA_DIR, 'docs')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
AUDIT_DB = os.path.join(DATA_DIR, 'audit.db')
