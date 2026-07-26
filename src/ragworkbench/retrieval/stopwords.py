"""ragworkbench/retrieval/stopwords -- English + Indonesian stopword sets.

Pure data + stdlib. Lifted from ``koboi/rag/retriever.py`` and decoupled (zero
``koboi.*`` imports). Used by the lexical retrievers (Keyword/BM25) to filter
high-frequency function words on BOTH index and query tokens so common words
stop producing spurious matches on out-of-scope queries. Opt-in via the
``stopwords`` constructor param of :class:`KeywordRetriever` / :class:`BM25Retriever`.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


# Compact English stopword set (~100 high-frequency function words). Opt-in via
# ``stopwords=True`` or ``stopwords="en"`` on a lexical retriever so common words
# stop producing spurious matches on out-of-scope queries. Default off (preserves
# the prior behavior of no filtering).
STOPWORDS_EN: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the to was
    were will with this these those they them their there here which who whom whose what
    when where why how all any both each few more most other some such no nor not only
    own same so than too very can do does did doing would should could about into over
    under again further then once
    """.split()
)


# Compact Indonesian stopword set (~80 high-frequency function words). Opt-in via
# ``stopwords="id"``. Indonesian function words (yang/dan/di/ke/untuk/...) otherwise
# dominate BM25/TF-IDF scores at corpus scale and produce spurious matches --
# invisible on small corpora but distorts IDF on a production-scale ID corpus.
# Default off.
STOPWORDS_ID: frozenset[str] = frozenset(
    """
    yang dan di ke dari pada untuk dengan dalam atau juga karena maka jika kalau agar
    supaya sehingga tetapi tapi namun oleh ada adalah akan bisa dapat sudah telah saya
    aku kami kita kamu anda ia dia mereka ini itu tersebut satu sebuah seorang setiap
    beberapa banyak lebih sedikit sangat paling hanya masih belum tidak bukan tanpa
    tentang antara hingga sampai bagi sama menjadi sebagai saat ketika waktu tahun hari
    orang nya lah kah pun per para ada atas saya anda
    """.split()
)


def get_stopwords(spec: object) -> set[str] | None:
    """Resolve a ``stopwords`` spec to a lowercased set (or ``None`` = no filtering).

    Accepts:

    - ``None`` / ``False``       -> no filtering (returns ``None``)
    - ``True``                   -> English set (back-compat with koboi)
    - ``"en"`` / ``"id"``        -> named language set (case-insensitive)
    - ``set`` / ``frozenset``    -> passthrough (custom word list, lowercased)
    - any other string           -> warns + returns ``None`` (strict; an unknown
                                    language does NOT silently character-iterate
                                    into a bogus single-char set)
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        lang = spec.lower()
        if lang == "en":
            return set(STOPWORDS_EN)
        if lang == "id":
            return set(STOPWORDS_ID)
        _logger.warning("Unknown stopwords language %r; supported: 'en', 'id'. Ignoring.", spec)
        return None
    if spec is True:
        return set(STOPWORDS_EN)
    return {str(w).lower() for w in spec}  # type: ignore[arg-type]
