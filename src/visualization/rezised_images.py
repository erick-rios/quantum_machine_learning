import os
from PIL import Image
import numpy as np

# Definir el directorio donde están las imágenes
data_dir = '../data/dogs_vs_cats'  # Cambia a la ruta exacta si es diferente
output_dir = 'resized_images'  # Carpeta de salida para las imágenes redimensionadas
categories = ['dog']  # Añade más categorías si tienes otras

# Función para calcular dimensiones promedio
def calculate_average_dimensions(data_dir, categories):
    widths, heights = [], []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))
    print(f"Average dimensions: {avg_width}x{avg_height}")
    return avg_width, avg_height

# Llamar a la función para calcular las dimensiones promedio
avg_width, avg_height = calculate_average_dimensions(data_dir, categories)

# Escoge el tamaño final para todas las imágenes (se recomienda 64x64 o 128x128)
target_size = (64, 64)

# Crear una función para redimensionar y guardar imágenes
def resize_and_save_images(data_dir, output_dir, categories, target_size):
    os.makedirs(output_dir, exist_ok=True)
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path)
                img_resized = img.resize(target_size)
                img_resized.save(os.path.join(output_category_path, img_name))
            except Exception as e:
                print(f"Error resizing image {img_path}: {e}")

# Redimensionar y guardar las imágenes
resize_and_save_images(data_dir, output_dir, categories, target_size)
print(f"Images have been resized and saved to '{output_dir}' with target size {target_size}")
