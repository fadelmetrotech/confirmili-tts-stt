import os, json, re
from pathlib import Path

DICTIONARY_PATH = Path(__file__).parent / "darija_french_dictionary.json"

_dictionary = {}
_pattern = None

def load_dictionary():
    global _dictionary, _pattern
    if not DICTIONARY_PATH.exists():
        print(f"Warning: Dictionary not found at {DICTIONARY_PATH}")
        return {}
    
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        _dictionary = json.load(f)
    
    # Sort by length descending to match longer phrases first
    sorted_keys = sorted(_dictionary.keys(), key=len, reverse=True)
    # Escape keys and build regex
    pattern_str = r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b'
    _pattern = re.compile(pattern_str, re.IGNORECASE)
    return _dictionary

def phonetic_transliterate(word: str) -> str:
    """Phonetic transliteration of French word to Arabic script."""
    # Multi-char patterns first
    replacements = [
        ("tion", "سيون"), ("sion", "زيون"), ("ment", "مون"),
        ("eur", "ور"), ("eux", "و"), ("euse", "وز"),
        ("oir", "وار"), ("oire", "وار"), ("aire", "ار"),
        ("ille", "اي"), ("elle", "ال"), ("ette", "ات"),
        ("aque", "اك"), ("ique", "يك"), ("que", "ك"),
        ("che", "ش"), ("ch", "ش"), ("ph", "ف"),
        ("th", "ت"), ("gn", "ني"), ("ou", "و"),
        ("au", "او"), ("eau", "و"), ("ai", "ي"),
        ("ei", "ي"), ("oi", "وا"), ("on", "ون"),
        ("an", "ون"), ("en", "ون"), ("in", "ان"),
        ("un", "ان"), ("eu", "و"), ("é", "ي"),
        ("è", "ا"), ("ê", "ا"), ("à", "ا"),
        ("â", "ا"), ("î", "ي"), ("ô", "و"),
        ("û", "و"), ("ù", "و"), ("ç", "س"),
        ("œ", "و"),
        # Single consonants
        ("b", "ب"), ("c", "ك"), ("d", "د"), ("f", "ف"),
        ("g", "ق"), ("h", ""), ("j", "ج"), ("k", "ك"),
        ("l", "ل"), ("m", "م"), ("n", "ن"), ("p", "ب"),
        ("q", "ك"), ("r", "ر"), ("s", "س"), ("t", "ت"),
        ("v", "ف"), ("w", "و"), ("x", "كس"), ("z", "ز"),
        # Vowels
        ("a", "ا"), ("e", ""), ("i", "ي"), ("o", "و"), ("u", "و"), ("y", "ي")
    ]
    res = word.lower()
    for old, new in replacements:
        res = res.replace(old, new)
    return res

def transliterate_french(text: str) -> str:
    """
    Convert French words in Latin script to their Darija Arabic equivalents.
    Uses a dictionary for common words and phonetic rules for unknowns.
    """
    if not _dictionary:
        load_dictionary()
    
    # 1. Handle common French articles/prepositions that might be attached (l', d')
    text = re.sub(r"\b[ldLD]'([a-zA-Z\u00C0-\u017F]+)", r"\1", text)
    
    # 2. Extract and replace words
    def _replace_match(match):
        word = match.group(0)
        word_lower = word.lower()
        if word_lower in _dictionary:
            return _dictionary[word_lower]
        # Only transliterate if it looks like a word (at least 2 chars)
        if len(word) >= 2:
            return phonetic_transliterate(word_lower)
        return word

    # Match Latin words 
    res = re.sub(r'[a-zA-Z\u00C0-\u017F]+', _replace_match, text)
    return res

if __name__ == "__main__":
    # Quick test
    test_str = "نحتاج une livraison gratuite لـ l'adresse تاعي"
    print(f"Original: {test_str}")
    print(f"Converted: {transliterate_french(test_str)}")
