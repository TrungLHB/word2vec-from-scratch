## Installation

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

```bash
pip install -r requirement.txt
```

## Running the Model

```bash
python main.py
```

## Implementation Description

This implementation uses the Skip-gram model with Binary Cross Entropy loss (implicit Negative Sampling).

### Forward Pass

Given a target word index $u$ and a context word index $v$:
- **Target Word Embedding ($h_u$)**: Extracted from the input weight matrix $W_1$.
  $$ h_u = (W_1)_{u, :} $$
- **Context Word Embedding ($h_v$)**: Extracted from the output weight matrix $W_2$.
  $$ h_v = (W_2)_{:, v} $$
- **Dot Product ($z$)**: Measures the similarity between the target and context.
  $$ z = h_u^T h_v $$
- **Prediction ($\hat{y}$)**: Sigmoid function applied to $z$ to get a probability between 0 and 1.
  $$ \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} $$

### Loss Function

**Binary Cross Entropy** loss. $y = 1$ for true context words (positive samples) and $y = 0$ for randomly sampled words (negative samples).
$$ L = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right] $$

### Backward Pass (Gradients)

To update the weights, we compute the partial derivatives of the loss with respect to our parameters.

Derivative of loss with respect to $z$:
$$ \frac{\partial L}{\partial z} = \hat{y} - y $$

Gradients with respect to the embeddings:
$$ \frac{\partial L}{\partial h_v} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial h_v} = (\hat{y} - y) h_u $$
$$ \frac{\partial L}{\partial h_u} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial h_u} = (\hat{y} - y) h_v $$

### Weight Update (Stochastic Gradient Descent)

Using learning rate $\eta$, we update the specific row/column of the weight matrices:

- Update for Context weights ($W_2$):
  $$ (W_2)_{:, v} \leftarrow (W_2)_{:, v} - \eta (\hat{y} - y) h_u $$
- Update for Target weights ($W_1$):
  $$ (W_1)_{u, :} \leftarrow (W_1)_{u, :} - \eta (\hat{y} - y) h_v^T $$

These math formulas directly correspond to the code implemented in `word2vec.py`.