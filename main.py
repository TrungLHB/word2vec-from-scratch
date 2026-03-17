from word2vec import Word2Vec
from data_loader import DataLoader

HYPE_PARAMETERS = {
    # corpus & vocab parameters
    "min_count": 3,
    "window_size": 2,
    "num_ns": 5, # Number of negative samples

    # model parameters
    "embedding_dim": 10,

    # training parameters
    "epochs": 50,
    "learning_rate": 0.01
}

def main():
    print("Initializing Data Pipeline...")
    
    # Initialize DataLoader which automatically runs the pipeline
    loader = DataLoader(
        min_count=HYPE_PARAMETERS["min_count"], 
        window_size=HYPE_PARAMETERS["window_size"],
        num_ns=HYPE_PARAMETERS["num_ns"]
    )
    
    print(f"Total words in corpus: {len(loader.words)}")
    print(f"Vocabulary size after min_count filter: {loader.vocab_size}")
    print(f"Total training pairs (target, context): {len(loader.training_data)}")
    
    # Initialize Model
    model = Word2Vec(loader.vocab_size, HYPE_PARAMETERS["embedding_dim"])
    
    # Train Model
    print("\nStarting training...")
    model.train(
        loader.training_data,
        epochs=HYPE_PARAMETERS["epochs"],
        learning_rate=HYPE_PARAMETERS["learning_rate"]
    )
    
    # Test word similarities
    test_words = ['king', 'queen', 'man', 'woman', 'romeo', 'juliet', 'brother', 'sister', 'good', 'bad', 'blood', 'sword']
    
    # Only test words that are actually in our vocabulary
    test_words_in_vocab = [w for w in test_words if w in loader.word_to_index]
    
    # If none of our predefined test words are in the corpus (e.g. if we downloaded a very different corpus),
    # Let's just grab the 5 most frequent words to demonstrate
    if not test_words_in_vocab:
        test_words_in_vocab = [t[0] for t in loader.word_counts.most_common(10)[5:]]
        
    print("\nTesting similar words:")
    for word in test_words_in_vocab:
        idx = loader.word_to_index[word]
        similar_words = model.most_similar(idx, loader.index_to_word, top_k=3)
        print(f"Words similar to '{word}':")
        for sim_word, score in similar_words:
            print(f"  - {sim_word} ({score:.4f})")

if __name__ == "__main__":
    main()
