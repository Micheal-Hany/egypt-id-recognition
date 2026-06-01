"""National ID decoder for Egyptian ID cards."""

import re
from config import GOVERNORATES, CENTURY_MAP, NID_LENGTH, NID_VALID_FIRST_DIGITS
from utils.helpers import normalize_digits, normalize_digits_rtl, fix_first_digit_misreads


def decode_national_id(nid: str) -> dict:
    """
    Decode and validate an Egyptian national ID number.
    
    Args:
        nid: 14-digit national ID string
        
    Returns:
        Dictionary with decoded information or error message
    """
    nid = normalize_digits(nid)
    if len(nid) != NID_LENGTH:
        return {"valid": False, "error": f"يجب أن يكون الرقم {NID_LENGTH} خانة، تم إدخال {len(nid)} خانة"}

    century_digit = nid[0]
    if century_digit not in CENTURY_MAP:
        return {"valid": False, "error": "الخانة الأولى غير صحيحة (يجب أن تكون 2 أو 3)"}

    year = CENTURY_MAP[century_digit] + nid[1:3]
    month = nid[3:5]
    day = nid[5:7]
    gov = nid[7:9]
    seq = nid[9:13]
    gender_digit = int(nid[12])
    checksum = nid[13]

    try:
        if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            raise ValueError
    except ValueError:
        return {"valid": False, "error": "تاريخ الميلاد في الرقم القومي غير صحيح"}

    return {
        "valid": True,
        "national_id": nid,
        "birth_date": f"{day}/{month}/{year}",
        "gender": "ذكر" if gender_digit % 2 != 0 else "أنثى",
        "governorate": GOVERNORATES.get(gov, f"محافظة غير معروفة ({gov})"),
        "sequence": seq,
        "checksum_digit": checksum,
    }


def extract_national_id_from_text(ocr_results: list[dict]) -> str | None:
    """
    Robust NID extraction with RTL spatial ordering as the primary strategy.
    
    Strategies (in order):
    0. RTL group reversal — for tokens with space-separated groups in RTL order
    1. RTL-ordered per-variant sequences — spatially ordered results
    2. Per-token search — individual token normalization
    3. Concatenated search — all tokens joined
    4. Sliding window — search with first-digit misread correction
    5. Relaxed validation — date check disabled as last resort
    
    Args:
        ocr_results: List of OCR result dictionaries from run_ocr()
        
    Returns:
        14-digit NID string or None if extraction failed
    """
    def is_valid_candidate(c: str) -> bool:
        if not re.fullmatch(r'[23]\d{13}', c):
            return False
        month = int(c[3:5])
        day = int(c[5:7])
        return 1 <= month <= 12 and 1 <= day <= 31

    def search_in_digit_string(digit_str: str, strict: bool = True) -> str | None:
        for variant in fix_first_digit_misreads(digit_str):
            m = re.search(r'[23]\d{13}', variant)
            if m:
                if strict and is_valid_candidate(m.group()):
                    return m.group()
                elif not strict and re.fullmatch(r'[23]\d{13}', m.group()):
                    return m.group()
        return None

    # ── Strategy -1: per-token RTL group reversal (highest priority) ─────────
    real_results = [r for r in ocr_results if r.get("confidence", 0) >= 0]

    for item in sorted(real_results, key=lambda x: -x.get("confidence", 0)):
        rtl_norm = normalize_digits_rtl(item["text"])
        found = search_in_digit_string(rtl_norm, strict=True)
        if found:
            return found
        ltr_norm = normalize_digits(item["text"])
        if ltr_norm != rtl_norm:
            found = search_in_digit_string(ltr_norm, strict=True)
            if found:
                return found

    # ── Strategy 0: RTL-ordered per-variant sequences ──────────────────────────
    variant_sequences = None
    for item in ocr_results:
        if item.get("confidence") == -1.0 and "_variant_sequences" in item:
            variant_sequences = item["_variant_sequences"]
            break

    if variant_sequences:
        for ordered_seq in variant_sequences:
            for norm_fn in (normalize_digits_rtl, normalize_digits):
                rtl_digits = "".join(norm_fn(item["text"]) for item in ordered_seq)
                found = search_in_digit_string(rtl_digits, strict=True)
                if found:
                    return found
                pure = "".join(ch for ch in rtl_digits if ch.isdigit())
                for i in range(len(pure) - 13):
                    found = search_in_digit_string(pure[i:i + 14], strict=True)
                    if found:
                        return found

    # ── Strategy A: per-token search ──────────────────────────────────────────
    normalized_texts = [normalize_digits(r["text"]) for r in real_results]

    for digits in normalized_texts:
        found = search_in_digit_string(digits, strict=True)
        if found:
            return found

    # ── Strategy B: concatenated (LTR fallback) ───────────────────────────────
    all_digits = "".join(normalized_texts)
    found = search_in_digit_string(all_digits, strict=True)
    if found:
        return found

    # ── Strategy C: sliding window ────────────────────────────────────────────
    concat = "".join(ch for ch in all_digits if ch.isdigit())
    for i in range(len(concat) - 13):
        found = search_in_digit_string(concat[i:i + 14], strict=True)
        if found:
            return found

    # ── Strategy D: relax date validation ─────────────────────────────────────
    if variant_sequences:
        for ordered_seq in variant_sequences:
            rtl_digits = "".join(normalize_digits(item["text"]) for item in ordered_seq)
            pure = "".join(ch for ch in rtl_digits if ch.isdigit())
            for i in range(len(pure) - 13):
                found = search_in_digit_string(pure[i:i + 14], strict=False)
                if found:
                    return found

    for i in range(len(concat) - 13):
        found = search_in_digit_string(concat[i:i + 14], strict=False)
        if found:
            return found

    return None
