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

    def sigmoid(self, x):
        # Clip x to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def train_step(self, target_idx: int, context_idx: int, label: int, learning_rate: float):
        """
        Computes forward and backward pass for a single target-context pair using Binary Cross Entropy.
        Label: 1 for positive sample, 0 for negative sample.
        Model:
            u * W1 -> h_u
            h_v * W2 -> v
            y = sigmoid(h_u * h_v)
        Loss:
            L = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
        """

        # --- Forward Pass ---

        # h_u = u * W1 (u = one-hot vector of target_idx)
        h_u = self.W1[target_idx].reshape(-1, 1) # (embedding_dim, 1)

        # h_v = v * W2^T (v = one-hot vector of context_idx)
        h_v = self.W2[:, context_idx].reshape(-1, 1) # (embedding_dim, 1)
        
        # z = h_u^T * h_v
        z = np.dot(h_u.T, h_v) # (1, 1)
        pred = self.sigmoid(z).item() # scalar
        
        # --- Backward Pass ---

        # dL/dz = dL/da * da/dz = (a-y)/a(1-a) * a(1-a) = a-y
        dz = pred - label # scalar
        
        # dL/dh_v = dL/dz * dz/dh_v = dz * h_u
        dh_v = dz * h_u # (embedding_dim, 1)
        
        # dL/dh_u = dL/dz * dz/dh_u = dz * h_v
        dh_u = dz * h_v  # (embedding_dim, 1)
        
        # --- Update weights ---

        # W2[:, context_idx] = h_v => update W2 column with gradient dh_v
        self.W2[:, context_idx] -= learning_rate * dh_v.reshape(-1)
        
        # W1[target_idx, :] = h_u => update W1 row with gradient dh_u
        self.W1[target_idx, :] -= learning_rate * dh_u.reshape(-1)
        
        # --- Compute loss ---
        loss = -(label * np.log(pred + 1e-10) + (1 - label) * np.log(1 - pred + 1e-10))
        return loss

    def train(self, training_data, epochs=50, learning_rate=0.01):
        """
        Trains the Word2Vec model using stochastic gradient descent over binary labeled data.
        """
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(training_data)
            
            for target_idx, context_idx, label in training_data:
                loss = self.train_step(target_idx, context_idx, label, learning_rate)
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
