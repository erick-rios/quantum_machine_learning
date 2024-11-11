import os
from PIL import Image
import numpy as np


data_dir = '../data/'  
output_dir = '../data/resized_images'  
categories = ['dogs', 'cats']  


def calculate_average_dimensions(data_dir, categories):
    """Calculate the average dimensions of images in a directory

    Args:
        data_dir (_type_): _description_
        categories (_type_): _description_

    Returns:
        _type_: _description_
    """
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


avg_width, avg_height = calculate_average_dimensions(data_dir, categories)


target_size = (64, 64)


def resize_and_save_images(data_dir, output_dir, categories, target_size):
    """Resize images in a directory and save them to a new directory

    Args:
        data_dir (_type_): _description_
        output_dir (_type_): _description_
        categories (_type_): _description_
        target_size (_type_): _description_
    """
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


resize_and_save_images(data_dir, output_dir, categories, target_size)
print(f"Images have been resized and saved to '{output_dir}' with target size {target_size}")
