import re
import urllib.request
from collections import Counter
import numpy as np

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DUMMY_CORPUS = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

class DataLoader:
    """
    DataLoader handles fetching, preprocessing, and structuring text data for Word2Vec.
    """
    def __init__(self, url=CORPUS_URL, min_count=1, window_size=2, num_ns=5, subset_chars=100000):
        """
        Args:
            url: URL of the text corpus
            min_count: Minimum frequency of a word to be included in the vocabulary
            window_size: Size of the context window
            num_ns: Number of negative samples per positive context word
            subset_chars: Number of characters to use from the corpus
        """
        # --- Internal Config & State ---
        self._url = url
        self._text = None
        self._sampling_probs : np.ndarray = None # Probabilities for negative sampling
        
        # --- Public Facing Attributes ---
        self.words : list[str] = None
        self.word_to_index : dict[str, int] = None
        self.index_to_word : dict[int, str] = None
        self.word_counts : Counter[str] = None
        self.training_data : list[tuple[int, int, int]] = None # List of (target, context, label)
        self.vocab_size : int = 0
        
        # Processing steps
        self._download_corpus(subset_chars)
        self._preprocess_text()
        self._build_vocab(min_count)
        self._generate_training_data(window_size, num_ns)

    def _download_corpus(self, subset_chars):
        """Step 1: Download text data"""
        print(f"Downloading corpus from {self._url}...")
        try:
            response = urllib.request.urlopen(self._url)
            self._text = response.read().decode('utf-8')[:subset_chars]
        except Exception as e:
            print(f"Failed to download: {e}")
            self._text = DUMMY_CORPUS

    def _preprocess_text(self):
        """Step 2: Clean and tokenize text"""
        text = self._text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        self.words = text.split()

    def _build_vocab(self, min_count):
        """Step 3: Build index mappings and filter rare words"""
        self.word_counts = Counter(self.words)
        
        vocab = [word for word, count in self.word_counts.items() if count >= min_count]
        vocab = sorted(vocab)
        
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)
        
        # Compute negative sampling probabilities (P(w) = count(w)^(3/4) / Z)
        # Reduce the probability of sampling very common words as negative samples
        # As done in https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
        counts = np.array([self.word_counts[self.index_to_word[i]] for i in range(self.vocab_size)])
        powered_counts = counts ** 0.75
        self._sampling_probs = powered_counts / np.sum(powered_counts)
        
    def _generate_training_data(self, window_size, num_ns):
        """Step 4: Create (target, context, label) training pairs"""
        self.training_data = []
        word_indices = [self.word_to_index[w] for w in self.words if w in self.word_to_index]
        
        # Iterate through each word in the corpus with a sliding window
        for i, target_idx in enumerate(word_indices):
            start = max(0, i - window_size)
            end = min(len(word_indices), i + window_size + 1)
            
            # Positive samples
            pos_contexts = []
            for j in range(start, end):
                if i != j:
                    context_idx = word_indices[j]
                    self.training_data.append((target_idx, context_idx, 1))
                    pos_contexts.append(context_idx)
            
            if not pos_contexts:
                continue
                
            # Negative samples
            # Draw num_ns specific to the number of positive contexts we just generated
            total_neg_samples = len(pos_contexts) * num_ns
            if total_neg_samples > 0:
                neg_indices = np.random.choice(
                    self.vocab_size, 
                    size=total_neg_samples, 
                    p=self._sampling_probs
                )
                
                for neg_idx in neg_indices:
                    self.training_data.append((target_idx, neg_idx, 0))