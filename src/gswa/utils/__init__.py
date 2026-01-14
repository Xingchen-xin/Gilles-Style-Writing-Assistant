"""GSWA utilities package."""

from gswa.utils.ngram import (
    tokenize,
    get_ngrams,
    build_ngram_index,
    compute_ngram_overlap,
    find_longest_match,
)
from gswa.utils.progress import (
    Colors,
    ProgressBar,
    StepProgress,
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info,
    confirm,
    select_option,
)

__all__ = [
    # ngram
    "tokenize",
    "get_ngrams",
    "build_ngram_index",
    "compute_ngram_overlap",
    "find_longest_match",
    # progress
    "Colors",
    "ProgressBar",
    "StepProgress",
    "print_header",
    "print_section",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "confirm",
    "select_option",
]
