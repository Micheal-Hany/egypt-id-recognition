"""Utility functions for digit normalization and text processing."""

import re
from config import AR_DIGIT_MAP


def normalize_digits(text: str) -> str:
    """
    Convert Arabic-Indic numerals to Latin and strip non-digit chars.
    
    Args:
        text: Input text potentially containing Arabic or Latin digits
        
    Returns:
        String containing only Latin digits
    """
    return re.sub(r'\D', '', text.translate(AR_DIGIT_MAP))


def normalize_digits_rtl(text: str) -> str:
    """
    For tokens where EasyOCR returned Arabic digit groups with spaces
    (e.g. '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣'), the groups are in RTL visual order —
    rightmost group appears first in the string.
    Reversing the groups gives LTR order suitable for regex matching.
    
    Args:
        text: Input text with Arabic digits
        
    Returns:
        Normalized digit string with groups reversed
    """
    translated = text.translate(AR_DIGIT_MAP)
    groups = translated.split()
    if len(groups) > 1:
        return re.sub(r'\D', '', "".join(reversed(groups)))
    return re.sub(r'\D', '', translated)


def fix_first_digit_misreads(digit_str: str) -> list[str]:
    """
    Return variants of digit_str with common OCR misreads on the first digit corrected.
    OCR often confuses 2↔5, 3↔8, 3↔9, 2↔7 on Arabic ID cards.
    Only the first digit matters for NID validity (must be 2 or 3).
    
    Args:
        digit_str: String of digits
        
    Returns:
        List of possible digit string variants
    """
    if not digit_str:
        return [digit_str]
    variants = [digit_str]
    first_digit_fixes = {
        '5': '3', '8': '3', '9': '3',
        '7': '2', '4': '2',
        '6': '2',
    }
    if digit_str[0] in first_digit_fixes:
        variants.append(first_digit_fixes[digit_str[0]] + digit_str[1:])
    return variants
