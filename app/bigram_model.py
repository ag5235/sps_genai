import random
import re
from collections import Counter, defaultdict


class BigramModel:
    """Simple bigram language model: P(next_word | current_word)."""

    def __init__(self, corpus: list[str]):
        # Count how often each word follows each other word
        self.bigram_counts = defaultdict(Counter)
        for text in corpus:
            tokens = self._tokenize(text)
            for w1, w2 in zip(tokens, tokens[1:]):
                self.bigram_counts[w1][w2] += 1

        # Convert counts to conditional probabilities
        self.bigram_probs = {
            w1: {w2: count / sum(followers.values()) for w2, count in followers.items()}
            for w1, followers in self.bigram_counts.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def generate_text(self, start_word: str, length: int = 10) -> str:
        current = start_word.lower()
        words = [current]
        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:  # no known continuation, stop early
                break
            current = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            words.append(current)
        return " ".join(words)
