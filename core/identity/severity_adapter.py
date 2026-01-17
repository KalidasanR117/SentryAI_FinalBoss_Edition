# sentry/core/identity/severity_adapter.py
from core.identity.types import Identity

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def downgrade(sev):
    idx = SEVERITY_ORDER.index(sev)
    return SEVERITY_ORDER[max(0, idx - 1)]

def upgrade(sev):
    idx = SEVERITY_ORDER.index(sev)
    return SEVERITY_ORDER[min(len(SEVERITY_ORDER) - 1, idx + 1)]

def adjust_severity(rule_severity, identity):
    """
    Returns:
    - None → suppress alert
    - severity string → final severity
    """

    if identity == Identity.WHITELIST:
        if rule_severity in ["LOW", "MEDIUM"]:
            return None
        return downgrade(rule_severity)

    if identity == Identity.BLACKLIST:
        if rule_severity == "LOW":
            return "HIGH"
        return upgrade(rule_severity)

    return rule_severity
