import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Definir el directorio donde están las imágenes
data_dir = '../data/dogs_vs_cats/test/'  
categories = ['dogs']  


def load_images(data_dir, categories, num_samples=5):
    """Load images from a directory and return the data and labels as numpy arrays

    Args:
        data_dir (_type_): Path to the directory containing the images
        categories (_type_): List of categories to load
        num_samples (int, optional): Samples to show  . Defaults to 5.

    Returns:
        _type_: _description_
    """
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
    """Display sample images from each category

    Args:
        image_data (_type_): _description_
        image_labels (_type_): _description_
        categories (_type_): _description_
        num_samples (int, optional): _description_. Defaults to 5.
    """
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(15, 5))

    
    if len(categories) == 1:
        axes = [axes] 

    for i, category in enumerate(categories):
        category_images = [img for img, label in zip(image_data, image_labels) if label == category]
        
        num_to_show = min(num_samples, len(category_images))
        
        for j in range(num_to_show):
            axes[i][j].imshow(category_images[j])
            axes[i][j].axis('off')
            axes[i][j].set_title(category)
    plt.show()

show_sample_images(image_data, image_labels, categories)
show_sample_images(image_data_cat, image_labels_cat, ['cats'])

def calculate_image_stats(image_data):
    """Calculate the average width and height of the images

    Args:
        image_data (_type_): _description_
    """
    widths, heights = zip(*(img.shape[:2] for img in image_data))
    avg_width, avg_height = np.mean(widths), np.mean(heights)
    print(f"Average width: {avg_width:.2f}, Average height: {avg_height:.2f}")

calculate_image_stats(image_data)
calculate_image_stats(image_data_cat)
