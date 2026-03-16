import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the Skip-gram Model.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # W1 (vocab_size, embedding_dim): Context embeddings (Input -> Hidden layer)
        self.W1 = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        
        # W2 (embedding_dim, vocab_size): Output embeddings (Hidden -> Output layer)
        self.W2 = np.random.uniform(-0.1, 0.1, (embedding_dim, vocab_size))

    def forward(self, target_idx: int):
        """
        Forward pass to compute softmax probabilities.
        y_pred = softmax(u) = softmax(one_hot_vector * W1 * W2)
        """
        # h = one-hot vector * W1 -> (1, embedding_dim)
        self.h = self.W1[target_idx]
        assert self.h.shape == (self.embedding_dim,)
        
        # u = h * W2 -> (1, vocab_size)
        self.u = np.dot(self.h, self.W2)
        assert self.u.shape == (self.vocab_size,)
        
        # y_pred = softmax(u) = np.exp(x) / np.sum(np.exp(x), axis=0)
        # subtract max(u) for numerical stability, prevent overflow in exp()
        exp_u = np.exp(self.u - np.max(self.u))
        self.y_pred = exp_u / np.sum(exp_u)
        assert self.y_pred.shape == (self.vocab_size,)
        
        return self.y_pred

    def backward(self, target_idx: int, context_idx: int, learning_rate: float):
        """
        Backward pass to compute gradients and adjust weights.
        """
        # e = dL/du = y_pred - y_true (proof: https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
        e = self.y_pred.copy()
        e[context_idx] -= 1.0
        assert e.shape == (self.vocab_size,)

        # du/dW2 = h^T (Since u = h * W2)
        # Gradient of W2: dL/dW2 = dL/du * du/dW2 = e * h^T
        dW2 = np.outer(self.h, e)
        assert dW2.shape == (self.embedding_dim, self.vocab_size)
        
        # du/dh = W2^T (Since u = h * W2)
        # dh/dW1 = one_hot_vector^T (Since h = one_hot_vector * W1)
        # dL/dW1 = dL/du * du/dh * dh/dW1 = e * W2^T * one_hot_vector^T
        # Gradient of W1: dL/dW1 = (one_hot_vector * (W2 * e^T))^T
        dW1_row = np.dot(self.W2, e)
        assert dW1_row.shape == (self.embedding_dim,)
        
        # Update weights (Gradient Descent)
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
