"""
NLTK-based POS tagger for automatic text annotation.
Provides simple interface to tag sequences without manual POS assignment.
"""

import nltk
from typing import List, Tuple

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# Map NLTK POS tags to our simplified tag set
NLTK_TO_SIMPLIFIED = {
    # Nouns
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',

    # Verbs
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',

    # Adjectives
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',

    # Adverbs
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',

    # Determiners
    'DT': 'DET',

    # Pronouns
    'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',

    # Prepositions
    'IN': 'ADP',

    # Auxiliary verbs
    'MD': 'AUX', 'VB': 'AUX',

    # Punctuation
    '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', ';': 'PUNCT',
    '!': 'PUNCT', '?': 'PUNCT', '"': 'PUNCT', "'": 'PUNCT',
    '-': 'PUNCT', '(': 'PUNCT', ')': 'PUNCT',
}


def tag_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Tag a text with POS tags using NLTK.

    Args:
        text: Input text to tag.

    Returns:
        (tokens, pos_tags) - list of tokens and their simplified POS tags.
    """
    tokens = nltk.word_tokenize(text)
    nltk_tags = nltk.pos_tag(tokens)

    simplified_tags = []
    for token, nltk_tag in nltk_tags:
        simplified_tag = NLTK_TO_SIMPLIFIED.get(nltk_tag, 'NOUN')
        simplified_tags.append(simplified_tag)

    return tokens, simplified_tags


def tag_sequence(tokens: List[str]) -> List[str]:
    """
    Tag a pre-tokenized sequence with POS tags.

    Args:
        tokens: Pre-tokenized list of tokens.

    Returns:
        List of simplified POS tags.
    """
    nltk_tags = nltk.pos_tag(tokens)
    simplified_tags = []
    for token, nltk_tag in nltk_tags:
        simplified_tag = NLTK_TO_SIMPLIFIED.get(nltk_tag, 'NOUN')
        simplified_tags.append(simplified_tag)

    return simplified_tags
