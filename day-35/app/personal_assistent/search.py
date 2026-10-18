"""Symmetric text normalization for the local Russian note index."""
import re
import unicodedata
from importlib.metadata import version

from snowballstemmer import stemmer

INDEX_VERSION = 'russian-snowball-2:' + version('snowballstemmer')
QUERY_VERSION = 'russian-related-empty-only-1'
# Filter query words only: keep the index/source text intact. Match whole words
# before stemming so a function word cannot erase a name or a related noun.
# Negation and time-bearing words (не, без, после, перед, через) stay searchable.
_FUNCTION_WORDS = frozenset('''
а и или но в во на с со у к ко о об обо от из за по для до при ради
я мы ты вы он она оно они мне меня нам нас тебе тебя вам вас
что кто где когда какой какая какое какие какого какую как зачем почему
ли же бы это
'''.split())
_WORDS = re.compile(r'[^\W_]+', re.UNICODE)
_RUSSIAN = re.compile(r'[а-я]+')
# Used synchronously by the single event loop, never across awaits or threads.
_STEMMER = stemmer('russian')

# A small, explicit retrieval vocabulary, not a semantic model. Groups contain
# related words, not assertions of equivalence (e.g. light can mean electricity).
# Only explicitly listed surface forms trigger expansion: selecting by stem
# would turn names such as Соня/Света into sleep/electricity queries. Ambiguous
# forms (света/свету/свете) are deliberately omitted. Candidate matching uses stems.
# Expansion is one hop, query-only, and used only when no original match exists.
_RELATED_GROUPS = (
    'свет светом электричество электричества электричеству '
    'электричеством электричестве электроэнергия электроэнергии электроэнергию электроэнергией',
    # Snowball leaves these short irregular noun forms unchanged.
    'сон сна сну сном сне сны снов снам снами снах спать заснуть засыпать бессонница',
    'одежда одежды одежде одежду одеждой текстиль текстиля текстилю текстилем текстиле '
    'футболка джинсы брюки рубашка платье куртка',
    'ремонт ремонта ремонту ремонтом ремонте ремонты ремонтов ремонтам ремонтами ремонтах '
    'ремонтировать починка чинить сервис сервиса сервису сервисом сервисе '
    'мастерская мастерской мастерскую',
    'отдых отдыха отдыху отдыхом отдыхе отдыхать отпуск отпуска отпуску отпуском отпуске каникулы',
)


def words(text):
    folded = unicodedata.normalize('NFC', text).casefold()
    # Drop residual combining accents rather than splitting a word at them.
    # Compose first so й stays й; casefold itself may introduce a mark (İ).
    folded = ''.join(char for char in folded if not unicodedata.combining(char))
    return _WORDS.findall(folded.replace('ё', 'е'))


def word_stem(word):
    # Keep Latin names, numbers and mixed product codes literal.
    return _STEMMER.stemWord(word) if _RUSSIAN.fullmatch(word) else word


def index_values(text):
    tokens = words(text)
    return ' '.join(tokens), ' '.join(word_stem(token) for token in tokens)


def query_tokens(text):
    return [token for token in dict.fromkeys(words(text))
            if token not in _FUNCTION_WORDS][:30]


def match_query(text):
    tokens = query_tokens(text)
    stems = list(dict.fromkeys(word_stem(token) for token in tokens))
    # Tokens contain only letters/numbers. FTS operators and punctuation from
    # input are never interpreted as query syntax. Keep the existing OR search.
    # The same stem reached via several word forms is one search term, not
    # repeated evidence of relevance. Keep exact forms as separate phrases.
    return ' OR '.join([f'content:"{token}"' for token in tokens]
                       + [f'normalized:"{stem}"' for stem in stems])


_RELATED_VOCABULARY = tuple(
    (frozenset(group.split()), tuple(dict.fromkeys(word_stem(w) for w in group.split())))
    for group in _RELATED_GROUPS)


def related_query(text):
    tokens = set(query_tokens(text))
    stems = {word_stem(token) for token in tokens}
    related = dict.fromkeys(stem for triggers, candidates in _RELATED_VOCABULARY
                            if tokens.intersection(triggers) for stem in candidates
                            if stem not in stems)
    return ' OR '.join(f'normalized:"{stem}"' for stem in related)
