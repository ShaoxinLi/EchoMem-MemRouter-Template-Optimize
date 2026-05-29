"""Tests for QueryNormalizer."""

import pytest

from echomem.normalizer import QueryNormalizer


@pytest.fixture
def normalizer() -> QueryNormalizer:
    return QueryNormalizer()


def test_normalize_basic(normalizer: QueryNormalizer) -> None:
    assert normalizer.normalize("  Hello World  ") == "hello world"


def test_normalize_punctuation(normalizer: QueryNormalizer) -> None:
    # Punctuation removed, CJK words preserved
    assert normalizer.normalize("Jon 和 Gina 共同参加过哪些活动？") == "jon 和 gina 共同参加过哪些活动"


def test_normalize_empty(normalizer: QueryNormalizer) -> None:
    assert normalizer.normalize("") == ""


def test_normalize_mixed_case(normalizer: QueryNormalizer) -> None:
    assert normalizer.normalize("Do YoU ReMeMbEr?") == "do you remember"
