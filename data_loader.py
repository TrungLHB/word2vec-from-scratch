import re
import urllib.request
from collections import Counter

def download_corpus(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    """
    Downloads a small public domain corpus. We use Tiny-Shakespeare by default.
    """
    print(f"Downloading corpus from {url}...")
    try:
        response = urllib.request.urlopen(url)
        # Decode and take a subset (e.g., first 100,000 characters) to make 
        # training fast enough for a from-scratch pure-Python implementation.
        text = response.read().decode('utf-8')[:100000]
        return text
    except Exception as e:
        print(f"Failed to download: {e}")
        # Fallback to a very simple dummy text
        return "the quick brown fox jumps over the lazy dog. the dog barks. the fox runs away."

def preprocess_text(text):
    """
    Lowercases text, removes punctuation, and splits into words.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

def build_vocab(words, min_count=1):
    """
    Builds vocabulary by keeping words that appear at least min_count times.
    """
    word_counts = Counter(words)
    
    # Filter by min_count
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab = sorted(vocab) # Sort for deterministic ordering
    
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    return word_to_index, index_to_word, word_counts

def generate_training_data(words, word_to_index, window_size):
    """
    Generates (target_word_index, context_word_index) pairs.
    """
    training_data = []
    
    # Convert all words to indices, ignoring words not in vocab (if any filtering happened)
    word_indices = [word_to_index[w] for w in words if w in word_to_index]
    
    for i, target_idx in enumerate(word_indices):
        start = max(0, i - window_size)
        end = min(len(word_indices), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_idx = word_indices[j]
                training_data.append((target_idx, context_idx))
                
    return training_data
