import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from typing import Tuple

def load_images_from_directory(directory: str) -> Tuple:
    image_data = []
    image_labels = []
    categories = ['dogs', 'cats']
    
    for category in categories:
        category_path = os.path.join(directory, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((16, 16))  # Reduciendo el tamaño de imagen para facilitar el entrenamiento
                image_data.append(np.array(img_resized).flatten())
                image_labels.append(category)
            except Exception as e:
                print(f"Error al cargar la imagen {img_path}: {e}")
    
    return np.array(image_data), np.array(image_labels)

# Variables para almacenar los datos y etiquetas
data_dir = '../data/'
image_data, image_labels = load_images_from_directory(data_dir)

# Transformar las etiquetas a enteros
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(image_labels)

# Normalización de los datos de entrada
scaler = StandardScaler()
image_data = scaler.fit_transform(image_data)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, random_state=42)

# Convertir los datos de entrada y las etiquetas a double
X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64)

# Definir el dispositivo cuántico
dev = qml.device('default.qubit', wires=4)  # Incrementar a 4 qubits para una mayor capacidad

# Circuito cuántico con mayor profundidad y codificación de ángulo
@qml.qnode(dev, interface='torch')
def quantum_circuit(weights, X: torch.Tensor):
    # Codificación en ángulo
    for i in range(4):  
        qml.RY(X[i], wires=i)
    
    # Añadir varias capas de operaciones cuánticas para mayor capacidad de representación
    num_layers = len(weights) // 8
    for i in range(num_layers):
        for j in range(4):
            qml.RZ(weights[8 * i + 2 * j], wires=j)
            qml.RX(weights[8 * i + 2 * j + 1], wires=j)
        # Añadir CNOTs para entrelazamiento
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliZ(0))

# Función de pérdida con sigmoid
def quantum_svm_loss(weights, X, y):
    predictions = torch.stack([quantum_circuit(weights, x) for x in X])
    predictions = torch.sigmoid(predictions)  # Usar función sigmoide para valores entre 0 y 1
    return torch.nn.functional.binary_cross_entropy(predictions, y)

# Inicializar pesos con mayor profundidad y configuración del optimizador
weights = torch.randn(8 * 4, dtype=torch.float64, requires_grad=True)  # Ajustar el tamaño de los pesos
optimizer = optim.SGD([weights], lr=0.01, momentum=0.9)  # Cambiar a SGD con momento

# Entrenamiento
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = quantum_svm_loss(weights, X_train_tensor, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Predicción
def predict(X_test, weights):
    predictions = []
    for i in range(X_test.shape[0]):
        pred = quantum_circuit(weights, torch.tensor(X_test[i], dtype=torch.float64)).item()
        predictions.append(pred)
    return np.array(predictions)

# Convertir a etiquetas binarias
y_pred = predict(X_test, weights)
y_pred = (torch.sigmoid(torch.tensor(y_pred)) > 0.5).numpy()

# Evaluación
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

