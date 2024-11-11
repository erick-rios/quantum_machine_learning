import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Definir el directorio donde están las imágenes
data_dir = '../data/dogs_vs_cats/test/'  # Cambia a la ruta exacta si es diferente
categories = ['dogs']  # Puedes añadir otras categorías si tienes más

# Crear una función para cargar las imágenes y visualizar algunas aleatoriamente
def load_images(data_dir, categories, num_samples=5):
    image_data = []
    image_labels = []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                image_data.append(np.array(img))
                image_labels.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return image_data, image_labels

# Cargar las imágenes
image_data, image_labels = load_images(data_dir, categories)
image_data_cat, image_labels_cat = load_images(data_dir,['cats']) 
# Visualizar algunas imágenes de cada categoría
def show_sample_images(image_data, image_labels, categories, num_samples=5):
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(15, 5))

    # Asegurarse de que 'axes' sea una lista bidimensional
    if len(categories) == 1:
        axes = [axes]  # Convertir a una lista de una fila si hay solo una categoría

    for i, category in enumerate(categories):
        category_images = [img for img, label in zip(image_data, image_labels) if label == category]
        
        # Manejar el caso cuando hay menos imágenes que 'num_samples'
        num_to_show = min(num_samples, len(category_images))
        
        for j in range(num_to_show):
            axes[i][j].imshow(category_images[j])
            axes[i][j].axis('off')
            axes[i][j].set_title(category)
    plt.show()

show_sample_images(image_data, image_labels, categories)
show_sample_images(image_data_cat, image_labels_cat, ['cats'])
# Calcular dimensiones promedio de las imágenes
def calculate_image_stats(image_data):
    widths, heights = zip(*(img.shape[:2] for img in image_data))
    avg_width, avg_height = np.mean(widths), np.mean(heights)
    print(f"Average width: {avg_width:.2f}, Average height: {avg_height:.2f}")

calculate_image_stats(image_data)
calculate_image_stats(image_data_cat)
