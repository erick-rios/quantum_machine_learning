import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from typing import Tuple

def load_images_from_directory(directory: str) -> Tuple:
    """Load images from a directory and return the data and labels as numpy arrays

    Args:
        directory (str): Path to the directory containing the images

    Returns:
        Tuple: A tuple containing the image data and labels as numpy arrays
    """
    image_data = []
    image_labels = []
    
    categories = ['dogs', 'cats']  
    
    for category in categories:
        category_path = os.path.join(directory, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((32, 32))  # Resizing the image to 16x16
                image_data.append(np.array(img_resized).flatten())  # Convert the image to a 1D array and append to the list
                image_labels.append(category)
            except Exception as e:
                print(f"Error al cargar la imagen {img_path}: {e}")
    
    return np.array(image_data), np.array(image_labels)

# Variables to store the data and labels
data_dir = '../data/'
image_data, image_labels = load_images_from_directory(data_dir)

# Transform the labels to integers using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(image_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Initialize the quantum device and define the quantum circuit
dev = qml.device('default.qubit', wires=2)

# Define the quantum circuit
@qml.qnode(dev, interface='torch')

def quantum_circuit(weights, X: torch.Tensor) :
    """Quantum circuit for the quantum SVM

    Args:
        weights (_type_): Weights for the quantum circuit
        X (torch.Tensor): Input features for the quantum circuit 

    Returns:
        _type_: _description_
    """
    qml.RX(X[0], wires=0)
    qml.RY(X[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Loss function for the quantum SVM
def quantum_svm_loss(weights, X, y):
    """Compute the loss function for the quantum SVM

    Args:
        weights (_type_): Weights for the quantum circuit
        X (_type_): Data tensor for the input features
        y (_type_): Data tensor for the labels

    Returns:
        _type_: Loss value
    """
    predictions = torch.stack([quantum_circuit(weights, x) for x in X])
    return torch.mean((predictions - y) ** 2)  # Loss function is Mean Squared Error

# Definir el optimizador
weights = torch.randn(2, requires_grad=True)  # Initial Weights for the quantum circuit
optimizer = optim.Adam([weights], lr=0.05)

# Entrenamiento
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss = quantum_svm_loss(weights, X_train_tensor, y_train_tensor)  # Compute the loss
    loss.backward()  # Retropropagation 
    optimizer.step()  # Update the weights
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")


# Predictions and accuracy
def predict(X_test, weights):
    predictions = []
    for i in range(X_test.shape[0]):
        pred = quantum_circuit(weights, torch.tensor(X_test[i], dtype=torch.float32)).item()
        predictions.append(pred)
    return np.array(predictions)

# Make predictions on the test set using the trained weights
y_pred = predict(X_test, weights)

# Evaluate the accuracy
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualizar algunas predicciones
for i in range(5):
    plt.imshow(X_test[i].reshape(32, 32, 3))  # Set the shape of the image
    plt.title(f"Predicción: {'dog' if y_pred[i] > 0.5 else 'cat'} | Real: {label_encoder.inverse_transform([y_test[i]])[0]}")
    plt.show()
