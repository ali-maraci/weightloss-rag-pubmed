"""Classifies query safety level and generates appropriate medical disclaimers."""
import logging
import re
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Risk level of a medical query."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# Patterns that indicate high-risk medical queries
_HIGH_RISK_PATTERNS = [
    # Dosage and administration
    r'\b(?:dose|dosage|dosing|how\s+much|increase|decrease|titrat|adjust)\b.*\b(?:mg|milligram|unit|inject|take|prescri)\b',
    r'\b(?:start|switch|stop|discontinu|taper|withdraw)\b.*\b(?:semaglutide|tirzepatide|liraglutide|ozempic|wegovy|mounjaro|saxenda|rybelsus)\b',
    # Drug interactions
    r'\b(?:interact\w*|combination|combin\w*|together|mix|alongside|concomitant)\b.*\b(?:drug|medic\w*|insulin|metformin|sulfonylurea|warfarin)\b',
    # Emergency and acute symptoms
    r'\b(?:emergency|urgent|acute|severe|danger|fatal|death|die|dying|suicid|overdose|anaphyla|pancreatit)\b',
    # Self-medication and personal medical decisions
    r'\b(?:should\s+i|can\s+i|is\s+it\s+safe)\b.*\b(?:take|use|start|stop|skip|double)\b',
    # Pregnancy and vulnerable populations
    r'\b(?:pregnan|breastfeed|nursing|child|pediatric|elderly|renal\s+failure|liver\s+failure|dialysis)\b',
]

_MODERATE_RISK_PATTERNS = [
    # Side effects and adverse events
    r'\b(?:side\s+effects?|adverse|risk|harm|complication|contraindic\w*)\b',
    # Specific concerning symptoms
    r'\b(?:thyroid|cancer|tumor|medullary|carcinoma|gallbladder|kidney|gastroparesis)\b',
    # Off-label use
    r'\b(?:off.?label|not\s+approved|unapproved|cosmetic)\b',
]

_HIGH_RISK_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HIGH_RISK_PATTERNS]
_MODERATE_RISK_COMPILED = [re.compile(p, re.IGNORECASE) for p in _MODERATE_RISK_PATTERNS]

_DISCLAIMERS = {
    SafetyLevel.HIGH: (
        "**Important:** This information is from published research literature only "
        "and is NOT medical advice. Dosing, drug interactions, and treatment decisions "
        "must be made by a qualified healthcare provider who knows your medical history. "
        "If you are experiencing a medical emergency, contact emergency services immediately."
    ),
    SafetyLevel.MODERATE: (
        "**Note:** The information below is sourced from published research. "
        "Consult your healthcare provider before making any treatment decisions."
    ),
}


class SafetyClassifier:
    """Classifies medical queries by risk level using keyword pattern matching."""

    def classify(self, query: str) -> SafetyLevel:
        """Classify a query's safety level based on keyword patterns."""
        for pattern in _HIGH_RISK_COMPILED:
            if pattern.search(query):
                logger.info(f"Query classified as HIGH risk: matched pattern")
                return SafetyLevel.HIGH

        for pattern in _MODERATE_RISK_COMPILED:
            if pattern.search(query):
                logger.info(f"Query classified as MODERATE risk")
                return SafetyLevel.MODERATE

        return SafetyLevel.LOW

    def get_disclaimer(self, level: SafetyLevel) -> str:
        """Return the appropriate disclaimer for a safety level, or empty string for LOW."""
        return _DISCLAIMERS.get(level, "")

    def classify_and_disclaim(self, query: str) -> Tuple[SafetyLevel, str]:
        """Convenience method: classify and return (level, disclaimer)."""
        level = self.classify(query)
        return level, self.get_disclaimer(level)
