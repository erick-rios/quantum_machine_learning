import cv2
import os
import numpy as np

# Directorios de entrada y salida
input_dir = '../data/cats/'  # Directorio de imágenes de entrada
output_dir = '../data/cats_png/' # Directorio donde se guardarán las imágenes procesadas

dog_input_dir = '../data/dogs/'  # Directorio de imágenes de entrada
dog_output_dir = '../data/dogs_png/' # Directorio donde se guardarán las imágenes procesadas

# Cargar clasificadores de OpenCV para rostros de perros y gatos
# Nota: Estos clasificadores son aproximados y pueden necesitar ajustes según el dataset.
# Asegúrate de tener los archivos .xml de Haar cascades en la misma carpeta o de definir la ruta completa.
cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
dog_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml') # Para perros podría requerir un XML personalizado

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para procesar imágenes
def process_images():
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue  # Ignorar archivos que no sean imágenes

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detección de rostros de gatos y perros
        cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        dog_faces = dog_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        faces = list(cat_faces) + list(dog_faces)  # Unir detecciones de perros y gatos

        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Recortar el rostro
            face = gray[y:y+h, x:x+w]

            # Redimensionar a 16x16 píxeles
            resized_face = cv2.resize(face, (16, 16))

            # Crear imagen en blanco (fondo blanco)
            output_img = np.ones_like(gray) * 255

            # Pegar el rostro detectado en la posición original, pero en escala de grises
            output_img[y:y+h, x:x+w] = cv2.resize(resized_face, (w, h))

            # Guardar la imagen procesada
            output_path = os.path.join(output_dir, f"processed_{img_name}")
            cv2.imwrite(output_path, output_img)

# Llamada a la función para procesar las imágenes
process_images()

print("Proceso completado. Las imágenes procesadas están en:", output_dir)
