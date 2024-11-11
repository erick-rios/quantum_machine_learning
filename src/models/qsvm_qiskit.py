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

# Cargar las imágenes y etiquetas (asumiendo que ya tienes las imágenes listas)
def load_images_from_directory(directory):
    image_data = []
    image_labels = []
    
    categories = ['dogs', 'cats']  # Asegúrate de que los nombres de las carpetas sean correctos
    
    for category in categories:
        category_path = os.path.join(directory, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((16, 16))  # Ajusta al tamaño deseado
                image_data.append(np.array(img_resized).flatten())  # Convertir a vector plano
                image_labels.append(category)
            except Exception as e:
                print(f"Error al cargar la imagen {img_path}: {e}")
    
    return np.array(image_data), np.array(image_labels)

# Preprocesar datos
data_dir = '../data/'
image_data, image_labels = load_images_from_directory(data_dir)

# Convertir etiquetas a números
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(image_labels)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, random_state=42)

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Crear un dispositivo cuántico
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, interface='torch')
def quantum_circuit(weights, X):
    qml.RX(X[0], wires=0)
    qml.RY(X[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Función de pérdida para el SVM cuántico
def quantum_svm_loss(weights, X, y):
    predictions = torch.stack([quantum_circuit(weights, x) for x in X])
    return torch.mean((predictions - y) ** 2)  # Función de pérdida MSE como ejemplo

# Definir el optimizador
weights = torch.randn(2, requires_grad=True)  # Pesos iniciales para el circuito cuántico
optimizer = optim.Adam([weights], lr=0.01)

# Entrenamiento
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss = quantum_svm_loss(weights, X_train_tensor, y_train_tensor)  # Suponiendo que ya tienes X_train_tensor y y_train_tensor
    loss.backward()  # Retropropagación
    optimizer.step()  # Actualización de los pesos
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
# Predicciones
def predict(X_test, weights):
    predictions = []
    for i in range(X_test.shape[0]):
        pred = qsvm_circuit(X_test[i], weights).item()
        predictions.append(pred)
    return np.array(predictions)

# Hacer predicciones en el conjunto de prueba
y_pred = predict(X_test, weights)

# Evaluación
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualizar algunas predicciones
for i in range(5):
    plt.imshow(X_test[i].reshape(10, 10, 3))  # Ajusta según el tamaño de la imagen
    plt.title(f"Predicción: {'dog' if y_pred[i] > 0.5 else 'cat'} | Real: {label_encoder.inverse_transform([y_test[i]])[0]}")
    plt.show()