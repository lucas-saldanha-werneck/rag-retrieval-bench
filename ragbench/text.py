import re
import unicodedata

# Short Portuguese stopword list. Enough for the demo, not a full linguistic list.
STOPWORDS = frozenset(
    """a o as os um uma de do da dos das em no na nos nas por para com sem que e ou
    se ao aos como mais ja eu meu minha seu sua voces voce nao ser ter foi sao
    esta estao qual quais quando onde""".split()
)


def fold(text: str) -> str:
    """Lowercase and remove accents, so 'cobrança' and 'cobranca' match."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", fold(text)) if t not in STOPWORDS]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams inside each word, padded with spaces at the edges."""
    grams: list[str] = []
    for tok in tokens(text):
        padded = f" {tok} "
        if len(padded) <= n:
            grams.append(padded)
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams
