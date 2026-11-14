"""Versioned compatibility surface. Frozen original common.py is untouched."""
from pathlib import Path
from rc3_relation_drl_20261113.common import ROOT, H_ID, verify_binding
from .persistence import Ledger, append, digest, put, read, replace_retry, sha

HOME = Path(__file__).resolve().parent
