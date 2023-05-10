langpairs_seen = [
    ("en", "de"), ("en", "it"), ("en", "ar"), ("en", "ru"),
    ("de", "it"), ("de", "nl"), ("de", "ru"),
    ("ar", "it"), ("ar", "ru"),
    ("de", "de"),
    ("ru", "ru"), ("ar", "ar"), ("nl", "nl"), ("it", "it"), ("en", "en"),
]

mlingpairs_seen = [
    ("xx", "en"),
    ("en", "xx"),
    ("xx", "xx"),
]

langpairs_unseen = [
    ("fr", "en"), # unseen query languages (CLIR)
    ("id", "nl"),
    ("es", "fr"),
    ("en", "pt"), # unseen document languages  (CLIR)
    ("de", "vt"),
    ("it", "zh"),
    ("fr", "pt"), # unseen both (CLIR)
    ("id", "vt"),
    ("pt", "zh"),
    ("fr", "fr"), # unseen MoIR
    ("id", "id"),
    ("es", "es"),
    ("pt", "pt"),
    ("zh", "zh"),
]

mlingpairs_unseen = [
    ("xx_unseen", "en"),
    ("en", "xx_unseen"),
    ("xx_unseen", "xx_unseen"),
]

langpairs = langpairs_unseen + langpairs_seen + mlingpairs_seen + mlingpairs_unseen

short2long = {
    "en": "english",
    "de": "german",
    "it": "italian",
    "nl": "dutch",
    "ru": "russian",
    "ar": "arabic",
    "xx": "multilingual_6l",
}
long2short = {v: k for k, v in short2long.items()}

short2long_unseen = {
    "fr": "french",
    "zh": "chinese",
    "hi": "hindi",
    "id": "indonesian",
    "ja": "japanese",
    "pt": "portuguese",
    "es": "spanish",
    "vt": "vietnamese",
    "xx_unseen": "multilingual_14l",
}

long2short_unseen = {v: k for k, v in short2long_unseen.items()}
