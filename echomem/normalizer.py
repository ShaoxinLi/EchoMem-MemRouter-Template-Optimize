"""QueryNormalizer — low-cost text normalization for stable template matching.

Design principle: do not classify, do not decide backend, only standardize text.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Lightweight time expressions to preserve during normalization
_TIME_PATTERNS = re.compile(
    r"(上次|最近|昨天|今天|明天|上周|下周|上个月|下个月|"
    r"去年|今年|明年|刚刚|刚才|之前|以后|"
    r"\d{4}年|\d{1,2}月|\d{1,2}日|"
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"last\s+(week|month|year)|next\s+(week|month|year)|"
    r"yesterday|today|tomorrow)",
    re.IGNORECASE,
)

# Punctuation and whitespace to collapse (CJK + ASCII)
_PUNCTUATION_COLLAPSE = re.compile(r"[，。！？、；：\"'（）【】《》,.!?;:'\"()\[\]<>\s]+")


class QueryNormalizer:
    """Normalize raw user queries into stable matching text."""

    def normalize(self, raw_query: str) -> str:
        """Normalize a raw user query.

        Steps:
            1. Strip leading/trailing whitespace.
            2. Replace common punctuation with single spaces.
            3. Collapse consecutive whitespace.
            4. Lowercase (English only, preserves CJK).

        Args:
            raw_query: The original user query.

        Returns:
            Normalized query string suitable for embedding and template matching.
        """
        if not raw_query:
            logger.debug("normalize received empty query")
            return ""

        # Step 1: strip
        text = raw_query.strip()

        # Step 2: replace punctuation clusters with single space
        text = _PUNCTUATION_COLLAPSE.sub(" ", text)

        # Step 3: collapse whitespace
        text = " ".join(text.split())

        # Step 4: lowercase (preserves CJK characters safely)
        text = text.lower()

        logger.debug(
            "Normalized query: raw='%s' -> normalized='%s'",
            raw_query,
            text,
        )
        return text
