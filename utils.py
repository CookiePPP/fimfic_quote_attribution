from functools import lru_cache
import json

config = json.load(open('config.json', 'r'))

try:
    from tqdm import tqdm
except:
    # create mock tqdm with .write() method
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return self.iterable.__iter__()
        @staticmethod
        def write(*args, **kwargs):
            print(*args, **kwargs)


def sample_from_dict(d: dict, v_lens: dict = None, random=None):
    """
    Take a dictionary of {key: v} and {key: len(v), ...}
    and sample a random (key, value) with weighted probability based on the length of the value."""
    if random is None:
        random = __import__('random')
    if v_lens is None:
        v_lens = {k: len(v) for k, v in d.items()}
    total_len = sum(v_lens.values())
    r = random.randint(0, total_len)
    for k, v in d.items():
        r -= v_lens[k]
        if r <= 0:
            return k, v

from transformers.tokenization_utils import _is_punctuation, _is_control, _is_whitespace
def strip_punc(s):
    """Remove all punctuation from a string (not including whitespace)."""
    return ''.join([c for c in s if not _is_punctuation(c)])
def strip_to_word(s):
    """Remove all punctuation and whitespace from a string."""
    return ''.join([c for c in s if not _is_punctuation(c) and not _is_control(c) and not _is_whitespace(c)])

@lru_cache
def get_implications():
    lines = open(config['custom_implications_path'], 'r', encoding='utf-8').readlines()
    implications = []
    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('#'):
                continue
            s, t = line.split('->')
            s = strip_to_word(s)
            t = strip_to_word(t)
            implications.append((s, t))
    return implications # e.g: [('Twi', 'Twilight'), ...]

@lru_cache
def get_implications_ids(vocab_path=None):
    import os
    if vocab_path is None:
        vocab_path = os.path.join("data", "vocab.txt")
    vocab = open(vocab_path, 'r', encoding='utf-8').read().splitlines()
    vocab = {v.strip(): i for i, v in enumerate(vocab) if v.strip()}
    implication_ids = []
    for s, t in get_implications():
        implication_ids.append((vocab[s], vocab[t]))
    return implication_ids

@lru_cache
def get_character_blacklist_ids(vocab_path=None):
    import os
    if vocab_path is None:
        vocab_path = os.path.join("data", "vocab.txt")
    vocab = open(vocab_path, 'r', encoding='utf-8').read().splitlines()
    vocab = {v.strip(): i for i, v in enumerate(vocab) if v.strip()}
    implication_ids = []
    for c in config['character_blacklist_infer']:
        implication_ids.append(vocab[c])
    return implication_ids

def get_female_character_ids(vocab_path=None):
    import os
    if vocab_path is None:
        vocab_path = os.path.join("data", "vocab.txt")
    vocab = open(vocab_path, 'r', encoding='utf-8').read().splitlines()
    vocab = {v.strip(): i for i, v in enumerate(vocab) if v.strip()}
    implication_ids = []
    for c in config['female_characters']:
        implication_ids.append(vocab[c])
    return implication_ids

def process_character(character):
    character = strip_to_word(character)
    return character

@lru_cache
def get_verbs_for_said():
    verbs_for_said = [l.strip() for l in open(config['verbs_for_said_path'], 'r', encoding='utf-8').readlines() if l.strip()]
    return verbs_for_said

def zip_equal(*args):
    """Zip arguments together, but check that they are all the same length."""
    assert len(set([len(arg) for arg in args])) == 1, f"Arguments must be the same length. Got lengths: {[len(arg) for arg in args]}"
    return zip(*args)