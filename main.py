import time
from word2vec import Word2Vec

from data_loader import (
    download_corpus,
    preprocess_text,
    build_vocab,
    generate_training_data
)

def main():
    # 1. Download Corpus
    text = download_corpus()
    
    # 2. Preprocess Text
    words = preprocess_text(text)
    print(f"Total words in corpus: {len(words)}")
    
    # 3. Build Vocabulary
    # Filtering words that appear less than 3 times to reduce target vocabulary size 
    # and speed up training, since simple softmax is O(V) per step.
    word_to_index, index_to_word, word_counts = build_vocab(words, min_count=3)
    vocab_size = len(word_to_index)
    print(f"Vocabulary size after min_count filter: {vocab_size}")
    
    # 4. Generate Training Pairs
    window_size = 2
    training_data = generate_training_data(words, word_to_index, window_size)
    print(f"Total training pairs (target, context): {len(training_data)}")
    
    # 5. Initialize Model
    embedding_dim = 10
    model = Word2Vec(vocab_size, embedding_dim)
    
    # 6. Train Model
    epochs = 10
    learning_rate = 0.05
    print("\nStarting training...")
    start_time = time.time()
    
    model.train(training_data, epochs=epochs, learning_rate=learning_rate)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    # 7. Test word similarities
    test_words = ['king', 'queen', 'man', 'woman', 'romeo', 'juliet', 'brother', 'sister', 'good', 'bad', 'blood', 'sword']
    
    # Only test words that are actually in our vocabulary
    test_words_in_vocab = [w for w in test_words if w in word_to_index]
    
    # If none of our predefined test words are in the corpus (e.g. if we downloaded a very different corpus),
    # Let's just grab the 5 most frequent words to demonstrate
    if not test_words_in_vocab:
        test_words_in_vocab = [t[0] for t in word_counts.most_common(10)[5:]]
        
    print("\nTesting similar words:")
    for word in test_words_in_vocab:
        idx = word_to_index[word]
        similar_words = model.most_similar(idx, index_to_word, top_k=3)
        print(f"Words similar to '{word}':")
        for sim_word, score in similar_words:
            print(f"  - {sim_word} ({score:.4f})")

if __name__ == "__main__":
    main()
