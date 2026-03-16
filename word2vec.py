import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the Skip-gram Model.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # W1: Context embeddings (Input to Hidden layer)
        # Shape: (vocab_size, embedding_dim)
        self.W1 = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        
        # W2: Output embeddings (Hidden to Output layer)
        # Shape: (embedding_dim, vocab_size)
        self.W2 = np.random.uniform(-0.1, 0.1, (embedding_dim, vocab_size))

    def forward(self, target_idx):
        """
        Forward pass to compute softmax probabilities.
        """
        # Given a single input word target_idx (which is effectively a one-hot vector with a 1 at target_idx),
        # its multiplication with W1 just selects the target_idx row of W1.
        self.h = self.W1[target_idx]
        
        # Compute u = W2^T * h. 
        # Since W2 is (embedding_dim, vocab_size) and h is (embedding_dim,), their dot product is (vocab_size,)
        self.u = np.dot(self.h, self.W2)
        
        # Apply Softmax activation function
        # We subtract max(u) for numerical stability (prevent overflow in exp)
        exp_u = np.exp(self.u - np.max(self.u))
        self.y_pred = exp_u / np.sum(exp_u)
        
        return self.y_pred

    def backward(self, target_idx, context_idx, learning_rate):
        """
        Backward pass to compute gradients and adjust weights.
        """
        # 1. Error for the output layer
        # e = y_pred - y_true
        # y_true is a one hot encoded vector with 1 at context_idx.
        e = self.y_pred.copy()
        e[context_idx] -= 1.0
        
        # 2. Compute gradients
        # dW1_row is the gradient with respect to the input word's embedding
        # dW2 is the gradient with respect to the output matrix
        
        # dW2 = outer product of h (hidden state) and e (error)
        dW2 = np.outer(self.h, e)
        
        # dW1_row = dot product of W2 and e
        dW1_row = np.dot(self.W2, e)
        
        # 3. Update weights (Gradient Descent)
        self.W2 -= learning_rate * dW2
        self.W1[target_idx] -= learning_rate * dW1_row

        # Calculate loss (Cross-Entropy Loss)
        # Loss = -log(y_pred[context_idx])
        loss = -np.log(self.y_pred[context_idx] + 1e-10) # 1e-10 added for numerical stability
        return loss

    def train(self, training_data, epochs=50, learning_rate=0.01):
        """
        Trains the Word2Vec model using stochastic gradient descent.
        """
        for epoch in range(epochs):
            total_loss = 0
            # Shuffle training data at the start of each epoch
            np.random.shuffle(training_data)
            
            for target_idx, context_idx in training_data:
                self.forward(target_idx)
                loss = self.backward(target_idx, context_idx, learning_rate)
                total_loss += loss
            
            avg_loss = total_loss / len(training_data)
            print(f"Epoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def get_word_vector(self, word_idx):
        """
        Returns the embedding vector for a given word index.
        """
        return self.W1[word_idx]

    def most_similar(self, word_idx, index_to_word, top_k=5):
        """
        Finds the top_k most similar words using cosine similarity.
        """
        query_vec = self.W1[word_idx]
        
        # Compute cosine similarity between query_vec and all word vectors in W1
        dot_products = np.dot(self.W1, query_vec)
        norms = np.linalg.norm(self.W1, axis=1) * np.linalg.norm(query_vec)
        cosine_sim = dot_products / norms
        
        # Get indices of top_k highest similarities (skipping the query word itself at index 0 after sorting)
        top_indices = np.argsort(cosine_sim)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            results.append((index_to_word[idx], cosine_sim[idx]))
            
        return results
