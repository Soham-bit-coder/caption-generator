# utils.py — vocabulary: turning words into numbers and back

import re
import pickle
from collections import Counter

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from config import VOCAB_THRESHOLD


def clean(text):
    """Lowercase and remove punctuation from a caption."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()   # split into list of words


class Vocabulary:
    def __init__(self):
        # word → number
        self.word2idx = {}
        # number → word
        self.idx2word = {}

        # Add 4 special tokens first (they always get fixed indices)
        for token in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
            self._add(token)

    def _add(self, word):
        """Add a single word to the vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)       # next available index
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build(self, captions):
        """
        Go through all captions, count word frequencies,
        and add words that appear >= VOCAB_THRESHOLD times.
        """
        counter = Counter()
        for caption in captions:
            counter.update(clean(caption))

        for word, count in counter.items():
            if count >= VOCAB_THRESHOLD:
                self._add(word)

        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, caption):
        """
        Turn a caption string into a list of numbers.
        Example: "a dog runs" → [1, 45, 23, 89, 2]
        (1=SOS, 45=a, 23=dog, 89=runs, 2=EOS)
        """
        words = clean(caption)
        unk = self.word2idx["<UNK>"]
        ids = [self.word2idx["<SOS>"]]
        for word in words:
            ids.append(self.word2idx.get(word, unk))
        ids.append(self.word2idx["<EOS>"])
        return ids

    def decode(self, indices):
        """
        Turn a list of numbers back into a caption string.
        Stops at <EOS>, skips <SOS> and <PAD>.
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<UNK>")
            if word == "<EOS>":
                break
            if word in ("<SOS>", "<PAD>", "<UNK>"):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
