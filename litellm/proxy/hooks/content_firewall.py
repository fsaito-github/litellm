# +------------------------------------+
#
#        AI Content Firewall
#        (Guardrail Engine)
#
# +------------------------------------+
# MVP guardrail engine providing regex, keyword, and pattern-based
# content filtering for prompt injection, jailbreak, PII leaks,
# toxic content, and data exfiltration detection.

import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FirewallRule(BaseModel):
    """A single firewall rule that can match against text content."""

    rule_id: str
    name: str
    description: str
    category: Literal[
        "prompt_injection",
        "jailbreak",
        "pii_leak",
        "toxic_content",
        "data_exfiltration",
        "custom_regex",
    ]
    pattern: Optional[str] = None  # regex pattern for regex-based rules
    keywords: Optional[List[str]] = None  # keyword list for keyword-based rules
    action: Literal["block", "warn", "log"] = "block"
    enabled: bool = True
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class FirewallViolation(BaseModel):
    """Represents a single violation detected by the firewall."""

    rule_id: str
    rule_name: str
    category: str
    severity: str
    action: str
    matched_text: str  # snippet of matched text (truncated to 100 chars)
    message_index: int  # which message triggered it


class ContentFirewallConfig(BaseModel):
    """Configuration for the content firewall."""

    enabled: bool = False
    check_input: bool = True
    check_output: bool = True
    custom_rules: List[FirewallRule] = Field(default_factory=list)
    log_violations: bool = True  # log to audit table


# ---------------------------------------------------------------------------
# Default built-in rules
# ---------------------------------------------------------------------------

_DEFAULT_RULES: List[FirewallRule] = [
    # --- Prompt Injection (3 rules) ---
    FirewallRule(
        rule_id="pi-001",
        name="Ignore Previous Instructions",
        description="Detects attempts to override prior instructions.",
        category="prompt_injection",
        pattern=r"(?i)\b(ignore|disregard|forget|skip|bypass)\b[^\n]{0,30}\b(previous|prior|above|earlier|all)\b[^\n]{0,30}\b(instructions?|prompts?|rules?|directives?)\b",
        action="block",
        severity="high",
    ),
    FirewallRule(
        rule_id="pi-002",
        name="You Are Now",
        description="Detects role reassignment attempts.",
        category="prompt_injection",
        pattern=r"(?i)\b(you\s+are\s+now|act\s+as\s+if\s+you\s+are|from\s+now\s+on\s+you\s+are|pretend\s+to\s+be)\b",
        action="block",
        severity="high",
    ),
    FirewallRule(
        rule_id="pi-003",
        name="System Prompt Override",
        description="Detects attempts to inject or override the system prompt.",
        category="prompt_injection",
        pattern=r"(?i)(new\s+system\s+prompt|override\s+system\s+(prompt|message)|system\s*:\s*you\s+are|<<\s*SYS\s*>>|\[SYSTEM\])",
        action="block",
        severity="critical",
    ),
    # --- Jailbreak (3 rules) ---
    FirewallRule(
        rule_id="jb-001",
        name="DAN Mode",
        description="Detects DAN (Do Anything Now) jailbreak attempts.",
        category="jailbreak",
        pattern=r"(?i)\b(DAN\s+mode|do\s+anything\s+now|DAN\s*[:\-]|enable\s+DAN|activate\s+DAN)\b",
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="jb-002",
        name="Bypass Safety",
        description="Detects requests to disable or bypass safety filters.",
        category="jailbreak",
        pattern=r"(?i)\b(bypass|disable|turn\s+off|remove|ignore)\b[^\n]{0,30}\b(safety|content\s+filter|guardrail|moderation|restriction|censorship)\b",
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="jb-003",
        name="No Restrictions",
        description="Detects attempts to remove model restrictions.",
        category="jailbreak",
        pattern=r"(?i)(pretend\s+you\s+have\s+no\s+restrictions|without\s+any\s+restrictions|you\s+have\s+no\s+limitations|you\s+can\s+say\s+anything|unrestricted\s+mode)",
        action="block",
        severity="high",
    ),
    # --- Data Exfiltration (3 rules) ---
    FirewallRule(
        rule_id="de-001",
        name="Repeat Everything Above",
        description="Detects attempts to exfiltrate the system prompt by asking the model to repeat it.",
        category="data_exfiltration",
        pattern=r"(?i)(repeat\s+(everything|all|the\s+text)\s+(above|before|from\s+the\s+beginning)|print\s+(the\s+)?(above|previous)\s+(text|prompt|instructions?))",
        action="block",
        severity="high",
    ),
    FirewallRule(
        rule_id="de-002",
        name="Show System Prompt",
        description="Detects requests to reveal the system prompt.",
        category="data_exfiltration",
        pattern=r"(?i)(show|display|reveal|output|print|tell)\s+(\w+\s+)?(your\s+)?(system\s+prompt|initial\s+prompt|original\s+instructions?|hidden\s+instructions?|system\s+message)",
        action="block",
        severity="high",
    ),
    FirewallRule(
        rule_id="de-003",
        name="Output Your Instructions",
        description="Detects requests to output training or operational instructions.",
        category="data_exfiltration",
        pattern=r"(?i)(output|print|list|give\s+me|what\s+are)\s+(your\s+)?(instructions?|configuration|training\s+data|fine[- ]?tuning|rules\s+you\s+follow)",
        action="block",
        severity="high",
    ),
    # --- PII Leak Prevention (3 rules) ---
    FirewallRule(
        rule_id="pii-001",
        name="SSN Pattern",
        description="Detects US Social Security Number patterns.",
        category="pii_leak",
        pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="pii-002",
        name="Credit Card Pattern",
        description="Detects common credit card number patterns (Visa, MC, Amex, Discover).",
        category="pii_leak",
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="pii-003",
        name="API Key Pattern",
        description="Detects common API key patterns (OpenAI sk-*, AWS AKIA*).",
        category="pii_leak",
        pattern=r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|xox[bprs]-[A-Za-z0-9\-]{10,})\b",
        action="block",
        severity="critical",
    ),
    # --- Toxic Content (3 rules) ---
    FirewallRule(
        rule_id="tox-001",
        name="Hate Speech Indicators",
        description="Detects common hate speech indicator phrases.",
        category="toxic_content",
        keywords=[
            "kill all",
            "death to",
            "exterminate",
            "ethnic cleansing",
            "white supremacy",
            "racial superiority",
            "gas the",
            "genocide",
        ],
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="tox-002",
        name="Threat Indicators",
        description="Detects direct threat language.",
        category="toxic_content",
        keywords=[
            "i will kill",
            "i will hurt",
            "bomb threat",
            "going to attack",
            "shoot up",
            "blow up",
            "going to murder",
        ],
        action="block",
        severity="critical",
    ),
    FirewallRule(
        rule_id="tox-003",
        name="Explicit Content Request",
        description="Detects requests for generating explicit or illegal content.",
        category="toxic_content",
        keywords=[
            "generate child exploitation",
            "create illegal content",
            "write a ransom note",
            "instructions for making a bomb",
            "how to make explosives",
            "how to synthesize drugs",
        ],
        action="block",
        severity="critical",
    ),
]


# ---------------------------------------------------------------------------
# ContentFirewall engine
# ---------------------------------------------------------------------------


class ContentFirewall:
    """
    Rule-based content firewall that scans text for policy violations.

    Supports regex patterns and keyword matching across categories
    including prompt injection, jailbreak, PII, toxicity, and
    data exfiltration.
    """

    def __init__(self, rules: Optional[List[FirewallRule]] = None) -> None:
        self._rules: Dict[str, FirewallRule] = {}
        self._compiled: Dict[str, "re.Pattern[str]"] = {}

        # Stats counters
        self._total_checks: int = 0
        self._total_blocks: int = 0
        self._violations_by_category: Dict[str, int] = defaultdict(int)

        source_rules = rules if rules is not None else _DEFAULT_RULES
        for rule in source_rules:
            self._register_rule(rule)

        verbose_proxy_logger.debug(
            "content_firewall: initialised with %d rules", len(self._rules)
        )

    # -- internal helpers ---------------------------------------------------

    def _register_rule(self, rule: FirewallRule) -> None:
        self._rules[rule.rule_id] = rule
        if rule.pattern:
            try:
                self._compiled[rule.rule_id] = re.compile(rule.pattern)
            except re.error as exc:
                verbose_proxy_logger.error(
                    "content_firewall: invalid regex in rule %s — %s",
                    rule.rule_id,
                    exc,
                )

    def _match_rule(
        self, rule: FirewallRule, text: str
    ) -> Optional[str]:
        """Return the matched snippet or ``None``."""
        if not rule.enabled:
            return None

        # Regex-based match
        if rule.pattern and rule.rule_id in self._compiled:
            m = self._compiled[rule.rule_id].search(text)
            if m:
                return m.group(0)

        # Keyword-based match
        if rule.keywords:
            text_lower = text.lower()
            for kw in rule.keywords:
                if kw.lower() in text_lower:
                    return kw

        return None

    # -- public API ---------------------------------------------------------

    _MAX_TEXT_LENGTH = 50000

    def check(self, text: str, message_index: int = 0) -> List[FirewallViolation]:
        """Run all enabled rules against *text* and return violations."""
        self._total_checks += 1
        violations: List[FirewallViolation] = []

        if len(text) > self._MAX_TEXT_LENGTH:
            verbose_proxy_logger.warning(
                "content_firewall: skipping regex checks — input length %d exceeds %d char limit",
                len(text),
                self._MAX_TEXT_LENGTH,
            )
            return violations

        for rule in self._rules.values():
            try:
                matched = self._match_rule(rule, text)
            except re.error as exc:
                verbose_proxy_logger.warning(
                    "content_firewall: regex error in rule %s — %s",
                    rule.rule_id,
                    exc,
                )
                continue
            if matched is not None:
                truncated = matched[:100]
                violation = FirewallViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    matched_text=truncated,
                    message_index=message_index,
                )
                violations.append(violation)
                self._violations_by_category[rule.category] += 1
                if rule.action == "block":
                    self._total_blocks += 1

                verbose_proxy_logger.warning(
                    "content_firewall: violation [%s] rule=%s cat=%s sev=%s matched='%s'",
                    rule.action.upper(),
                    rule.rule_id,
                    rule.category,
                    rule.severity,
                    truncated,
                )

        return violations

    def check_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[bool, List[FirewallViolation]]:
        """
        Check a list of chat messages (``{"role": ..., "content": ...}``).

        Returns ``(is_safe, violations)`` where *is_safe* is ``False``
        if any violation has ``action == "block"``.
        """
        all_violations: List[FirewallViolation] = []

        for idx, msg in enumerate(messages):
            content = msg.get("content")
            if not content or not isinstance(content, str):
                continue
            violations = self.check(content, message_index=idx)
            all_violations.extend(violations)

        is_safe = not any(v.action == "block" for v in all_violations)
        return is_safe, all_violations

    # -- rule management ----------------------------------------------------

    def add_rule(self, rule: FirewallRule) -> None:
        """Add (or replace) a firewall rule."""
        self._register_rule(rule)
        verbose_proxy_logger.debug(
            "content_firewall: added rule %s (%s)", rule.rule_id, rule.name
        )

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule by ID. No-op if not found."""
        self._rules.pop(rule_id, None)
        self._compiled.pop(rule_id, None)
        verbose_proxy_logger.debug(
            "content_firewall: removed rule %s", rule_id
        )

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> None:
        """Patch fields on an existing rule. Raises ``KeyError`` if missing."""
        if rule_id not in self._rules:
            raise KeyError(f"Rule '{rule_id}' not found")
        existing = self._rules[rule_id]
        updated = existing.model_copy(update=updates)
        self._register_rule(updated)
        verbose_proxy_logger.debug(
            "content_firewall: updated rule %s", rule_id
        )

    # -- stats --------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return violation statistics."""
        return {
            "total_checks": self._total_checks,
            "total_blocks": self._total_blocks,
            "block_rate": (
                round(self._total_blocks / self._total_checks, 4)
                if self._total_checks > 0
                else 0.0
            ),
            "violations_by_category": dict(self._violations_by_category),
            "active_rules": sum(1 for r in self._rules.values() if r.enabled),
            "total_rules": len(self._rules),
        }

    # -- audit integration --------------------------------------------------

    def log_violations_to_audit(
        self,
        violations: List[FirewallViolation],
        *,
        actor_id: str = "system",
    ) -> None:
        """
        Fire-and-forget audit log entries for each violation.

        Silently skips if the audit hook is unavailable.
        """
        if not violations:
            return
        try:
            from litellm.proxy.hooks.audit_log_hook import (
                fire_and_forget_audit_event,
            )

            for v in violations:
                fire_and_forget_audit_event(
                    action=f"content_firewall.{v.action}",
                    actor_id=actor_id,
                    actor_type="content_firewall",
                    resource_type="message",
                    resource_id=v.rule_id,
                    details={
                        "rule_name": v.rule_name,
                        "category": v.category,
                        "severity": v.severity,
                        "matched_text": v.matched_text,
                        "message_index": v.message_index,
                    },
                    status="blocked" if v.action == "block" else "flagged",
                )
        except Exception:
            verbose_proxy_logger.warning(
                "content_firewall: could not log to audit — %s",
                traceback.format_exc(),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

content_firewall: Optional[ContentFirewall] = None
