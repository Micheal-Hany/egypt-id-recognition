#!/usr/bin/env python
"""Quick test of refactored modules."""

from core.nid_decoder import decode_national_id
from utils.helpers import normalize_digits

print("✅ Testing refactored modules...\n")

# Test normalize_digits
print("1. Testing normalize_digits:")
result = normalize_digits("١٢٣")
print(f"   Arabic digits ١٢٣ → {result}")
assert result == "123", "normalize_digits failed"
print("   ✓ Passed\n")

# Test NID decoder
print("2. Testing NID decoder:")
decoded = decode_national_id("30101282600418")
print(f"   Input: 30101282600418")
print(f"   Valid: {decoded['valid']}")
print(f"   Birth Date: {decoded['birth_date']}")
print(f"   Gender: {decoded['gender']}")
print(f"   Governorate: {decoded['governorate']}")
assert decoded["valid"] == True, "Decode failed"
print("   ✓ Passed\n")

print("🎉 All tests passed! Refactored modules are working correctly.")
