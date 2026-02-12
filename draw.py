import gradio as gr
import numpy as np
import tensorflow as tf
import cv2

# Load the model you saved earlier
model = tf.keras.models.load_model('/Users/alex/Documents/handwriting_data/handwriting.keras')

def predict_digit(sketchpad_data):
    # 1. Get the image from Gradio
    # sketchpad_data['composite'] is an RGBA image
    img = sketchpad_data['composite']
    
    # 2. Convert to Grayscale
    # If the user draws white on black, we use the RGB channels.
    # If it's transparent, we use the Alpha channel.
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # 3. Resize to 28x28 (MNIST size)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 4. Auto-Inversion Logic
    # MNIST data is white (255) on black (0).
    # If the mean is high, the user drew black on white, so we flip it.
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    
    # 5. Normalize (0 to 1)
    img = img.astype('float32') / 255.0
    
    # 6. Reshape for the model: (Batch, Height, Width, Channels)
    img = img.reshape(1, 28, 28, 1)
    
    # 7. Predict
    prediction = model.predict(img)
    
    # Since the model uses 'from_logits=True', we apply softmax to get 0-1 probabilities
    probabilities = tf.nn.softmax(prediction[0]).numpy()
    print(probabilities)
    
    # 8. Return as a dictionary for gr.Label
    class_names = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    return {class_names[i]: float(probabilities[i]) for i in range(10)}

# Launch the interface
interface = gr.Interface(
    fn=predict_digit,
    # 'layer' or 'composite' works, but we use 'composite' for the full drawing
    inputs=gr.Sketchpad(type="numpy", label="Draw a letter"),
    outputs=gr.Label(num_top_classes=3),
    title="handwritting Classifier",
    description="Draw a thick letter in the center. If it fails, try making your lines thicker!"
)

interface.launch()