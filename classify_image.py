import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(img_path):
    # Load the trained model
    model = tf.keras.models.load_model('issue_classifier.h5')

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    class_names = ['security', 'electricity', 'water']  # Ensure this matches your dataset
    predicted_class = class_names[np.argmax(predictions)]

    if predicted_class == 'water':
        return "Water-related issue detected"
    elif predicted_class == 'electricity':
        return "Electricity-related issue detected"
    elif predicted_class == 'security':
        return "Security issue detected"
    else:
        return "Issue not recognized"

if __name__ == "__main__":
    # Provide paths to sample images for testing
    test_image_paths = [
        '/home/kundankarn/text_folder/train/security/image(1).png',
        '/home/kundankarn/text_folder/train/electricity/image(6).png',
        '/home/kundankarn/text_folder/train/water/image(4).png'
    ]

    for img_path in test_image_paths:
        result = classify_image(img_path)
        print(f'Image: {img_path}, Prediction: {result}')




