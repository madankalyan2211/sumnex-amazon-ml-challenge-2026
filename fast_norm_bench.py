import time
import re
import unicodedata

# Optimized fast normalization
TRANS_TABLE = str.maketrans({
    '&': ' and ',
    '-': ' ',
    '/': ' ',
    '\\': ' ',
    ',': ' ',
    '.': ' ',
    ';': ' ',
    ':': ' ',
    '#': ' ',
    '(': ' ',
    ')': ' ',
    '[': ' ',
    ']': ' ',
    '{': ' ',
    '}': ' ',
    '_': ' ',
    '*': ' ',
    '+': ' ',
    '!': ' ',
    '?': ' ',
    '"': ' ',
    "'": ' ',
    '`': ' ',
    '<': ' ',
    '>': ' ',
    '=': ' ',
    '@': ' ',
    '$': ' ',
    '%': ' ',
    '^': ' ',
    '~': ' ',
    '|': ' ',
})

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'pllc', 'llp', 'ltd', 'limited',
    'co', 'company', 'lp', 'services', 'service', 'enterprises', 'enterprise', 'group',
    'holdings', 'holding', 'partners', 'associates', 'solutions', 'technologies', 'international',
    'consulting', 'consultants', 'management', 'industries', 'realty', 'properties', 'capital',
    'pvt', 'private', 'proprietorship',
    'sarl', 'sas', 'sasu', 'sa', 'sci', 'snc', 'eurl', 'ste', 'societe',
}

RE_NUMBERS = re.compile(r'\b\d+\b')
RE_DOMAIN = re.compile(r'\.(com|org|net|in|co|io|fr|us|biz|info)\b')

def fast_clean(text: str) -> str:
    if not text:
        return ""
    # Unicode NFKD
    norm = unicodedata.normalize('NFKD', text)
    # Remove combining marks
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.lower()
    norm = RE_DOMAIN.sub('', norm)
    norm = norm.translate(TRANS_TABLE)
    return " ".join(norm.split())

def fast_norm_name(name: str):
    cleaned = fast_clean(name)
    tokens = tuple(cleaned.split())
    base_tokens = tuple(t for t in tokens if t not in LEGAL_SUFFIXES)
    base = " ".join(base_tokens) if base_tokens else cleaned
    compact = "".join(tokens)
    return cleaned, base, compact, tokens, base_tokens

# Benchmark on 100,000 synthetic names
names = ["Orelee's Barbershop & Salon Inc.", "Team Air Pvt. Ltd.", "Grand Connecticut LLC #351", "175 Boulevard du Président Franklin Roosevelt"] * 25000

t0 = time.time()
res = [fast_norm_name(n) for n in names]
t_elapsed = time.time() - t0
print(f"Processed {len(names)} names in {t_elapsed:.3f}s ({len(names)/t_elapsed:.0f} names/sec)")
