# sentry/core/identity/types.py
from enum import Enum

class Identity(Enum):
    WHITELIST = "WHITELIST"
    BLACKLIST = "BLACKLIST"
    UNKNOWN   = "UNKNOWN"
