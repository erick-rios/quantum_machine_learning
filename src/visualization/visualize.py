import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Definir las transformaciones
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.485, 0.456, 0.406))  # Normalización
])

# Leer la imagen con OpenCV
img = cv2.imread('../data/dogs_vs_cats/train/cats/cat.0.jpg')

# Convertir de BGR (OpenCV) a RGB (necesario para visualización y transformaciones correctas)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar las transformaciones
img_transformed = transform(img)

# Desnormalizar para visualización
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.485, 0.456, 0.406])
img_np = img_transformed.numpy().transpose((1, 2, 0))  # Convertir de (C, H, W) a (H, W, C)
img_np = std * img_np + mean  # Deshacer la normalización
img_np = np.clip(img_np, 0, 1)  # Asegurarse que los valores están en el rango [0, 1]

# Mostrar la imagen
plt.imshow(img_np)
plt.axis('off')  # Ocultar los ejes
plt.show()
