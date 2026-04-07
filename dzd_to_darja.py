"""
convert_dzd_to_darja(dzd_amount) → spoken Algerian Darja Arabic script.

Logic: all prices counted in Santiem (×100).
  5 DA   = 500 santiem  → خمسمية
  10 DA  = 1,000        → ألف
  50 DA  = 5,000        → خمس آلاف
  100 DA = 10,000       → عشر آلاف
  1000 DA= 100,000      → مية ألف
  10000 DA=1,000,000    → مليون
  180000 DA=18,000,000  → ثمنطاش مليون
"""

_HUNDREDS = {
    1: "مية", 2: "ميتين", 3: "تلتمية", 4: "ربعمية", 5: "خمسمية",
    6: "ستمية", 7: "سبعمية", 8: "ثمنمية", 9: "تسعمية",
}
_TENS = {
    2: "عشرين", 3: "ثلاثين", 4: "ربعين", 5: "خمسين",
    6: "ستين", 7: "سبعين", 8: "ثمانين", 9: "تسعين",
}
_TEENS = {
    11: "حداش", 12: "طناش", 13: "ثلثطاش", 14: "ربعطاش", 15: "خمسطاش",
    16: "سطاش", 17: "سبعطاش", 18: "ثمنطاش", 19: "تسعطاش",
}
_UNITS = {
    1: "واحد", 2: "زوج", 3: "ثلاثة", 4: "ربعة", 5: "خمسة",
    6: "ستة", 7: "سبعة", 8: "ثمانية", 9: "تسعة",
}
_SHORT = {
    3: "تلت", 4: "ربع", 5: "خمس", 6: "ست",
    7: "سبع", 8: "ثمن", 9: "تسع", 10: "عشر",
}


def _parts(n: int) -> list:
    """Spoken parts for integer 1-999 → list of Arabic strings."""
    if n <= 0:
        return []
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if r == 0:
        pass
    elif r == 10:
        parts.append("عشرة")
    elif 11 <= r <= 19:
        parts.append(_TEENS[r])
    else:
        u, t = r % 10, r // 10
        if u:
            parts.append(_UNITS[u])
        if t:
            parts.append(_TENS[t])
    return parts


def _thousands_phrase(n: int) -> str:
    """Spoken form of n × 1,000 santiem (n ≥ 1)."""
    if n == 1:
        return "ألف"
    if n == 2:
        return "ألفين"
    if 3 <= n <= 10:
        return _SHORT[n] + " آلاف"
    # 11+: build number then ألف
    if 11 <= n <= 19:
        return _TEENS[n] + " ألف"
    return " و ".join(_parts(n)) + " ألف"


def _millions_phrase(n: int) -> str:
    """Spoken form of n × 1,000,000 santiem (n ≥ 1)."""
    if n == 1:
        return "مليون"
    if n == 2:
        return "زوج ملاين"
    if 3 <= n <= 10:
        return _SHORT[n] + " ملاين"
    # 11+: مليون (singular)
    if 11 <= n <= 19:
        return _TEENS[n] + " مليون"
    return " و ".join(_parts(n)) + " مليون"


def convert_dzd_to_darja(dzd_amount) -> str:
    if not dzd_amount:
        return "والو"
    santiem = int(round(float(dzd_amount) * 100))
    if santiem <= 0:
        return "والو"

    result = []

    millions, santiem = divmod(santiem, 1_000_000)
    if millions:
        result.append(_millions_phrase(millions))

    thousands, santiem = divmod(santiem, 1_000)
    if thousands:
        result.append(_thousands_phrase(thousands))

    if santiem:
        result.append(" و ".join(_parts(santiem)))

    return " و ".join(result)


# ── Tests ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (5,      "خمسمية"),
        (10,     "ألف"),
        (20,     "ألفين"),
        (50,     "خمس آلاف"),
        (100,    "عشر آلاف"),
        (200,    "عشرين ألف"),
        (500,    "خمسين ألف"),
        (1000,   "مية ألف"),
        (1500,   "مية و خمسين ألف"),
        (1250,   "مية و خمسة و عشرين ألف"),
        (2000,   "ميتين ألف"),
        (2500,   "ميتين و خمسين ألف"),
        (5000,   "خمسمية ألف"),
        (10000,  "مليون"),
        (15500,  "مليون و خمسمية و خمسين ألف"),
        (20000,  "زوج ملاين"),
        (30000,  "تلت ملاين"),
        (45000,  "ربع ملاين و خمسمية ألف"),
        (100000, "عشر ملاين"),
        (180000, "ثمنطاش مليون"),
        (350000, "خمسة و ثلاثين مليون"),
    ]

    passed = 0
    for dzd, expected in tests:
        result = convert_dzd_to_darja(dzd)
        ok = result == expected
        if ok:
            passed += 1
            print(f"  ✅ {dzd:>8} DA → {result}")
        else:
            print(f"  ❌ {dzd:>8} DA → got:      {result}")
            print(f"                    expected: {expected}")
    print(f"\n{passed}/{len(tests)} passed")
