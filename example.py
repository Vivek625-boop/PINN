import numpy as np

# 1. Generate Synthetic Handwritten Digit Data (For pure standalone execution)
# Simulated 28x28 flattened pixel arrays (784 features) for digits 0 to 9
print("Generating training data...")
np.random.seed(42)
num_samples = 1000
num_classes = 10

# Fake images (random data) and labels
X_train = np.random.rand(num_samples, 784)
y_train = np.random.randint(0, num_classes, num_samples)

# Convert integer labels to 1-hot encoded vectors (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
Y_train_hot = np.eye(num_classes)[y_train]

# 2. Define Activation Functions and Derivatives
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # Subtracting max prevents numerical overflow (stability trick)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 3. Initialize Neural Network Parameters From Scratch
# Network Architecture: Input (784 neurons) -> Hidden (64 neurons) -> Output (10 neurons)
input_size = 784
hidden_size = 64
output_size = 10

# Weights initialization using He/Xavier principles
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

# 4. Training Loop (Gradient Descent)
epochs = 50
learning_rate = 0.01
batch_size = 32

print("Training the Neural Network from scratch...")
for epoch in range(epochs):
    # Shuffle dataset every epoch
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train_hot[permutation]
    
    epoch_loss = 0
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        
        # --- FORWARD PROPAGATION ---
        # Layer 1 (Hidden)
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        
        # Layer 2 (Output)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # Compute Cross-Entropy Loss
        loss = -np.sum(Y_batch * np.log(A2 + 1e-15)) / batch_size
        epoch_loss += loss
        
        # --- BACKPROPAGATION (Calculating Gradients) ---
        # Error at output layer
        dZ2 = (A2 - Y_batch) / batch_size
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Error at hidden layer
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # --- PARAMETER UPDATES (Gradient Descent Step) ---
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# 5. Inference / Prediction Verification
print("\nTesting the AI prediction pipeline...")
sample_input = X_train[0:1]  # Fetch the first generated image sample
actual_digit = y_train[0]

# Single forward pass for validation
hidden_pass = relu(np.dot(sample_input, W1) + b1)
output_probabilities = softmax(np.dot(hidden_pass, W2) + b2)
predicted_digit = np.argmax(output_probabilities)

print(f"Predicted Digit Output Matrix: {output_probabilities.round(2)}")
print(f"AI Prediction Choice: {predicted_digit}")
print(f"Ground Truth Label: {actual_digit}")