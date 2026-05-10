import streamlit as st
import PIL.Image as Image
# Import your specific ML frameworks (e.g., torch, tensorflow, ultralytics)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


st.set_page_config(layout="wide", page_title="Multi-Model Detection")

st.title("Multi-Model Image Detection Dashboard")







# 1. Cache the models so they stay in memory
@st.cache_resource
def load_all_models():
    # Load Transfer Learning (manual build + weights)

    # 1. Define the exact same architecture as used in Cell 13/18
    best_params = {'dense_units': 384, 'dropout_rate': 0.4228080027623846, 'lr': 0.00029786720003929296, 'unfreeze_layers': 20}

    # Define Data Augmentation (must match training exactly)
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    # Define Base Model
    base_model = tf.keras.applications.DenseNet121(
        weights=None, # Set to None since we are loading our own weights
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    # Build the Sequential wrapper
    transer_learning = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(best_params['dense_units'], use_bias=False, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(best_params['dropout_rate']),
        layers.Dense(2, activation="softmax")
    ])

    #  Load the weights from the .keras file
    # Keras 3 allows load_weights to read from a full model file
    transer_learning.load_weights("./models/best_crack_model_densenet.keras")

    #  Compile (required if you want to evaluate or continue training)
    transer_learning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    m4 = transer_learning 
    
    # Load others
    m1 = keras.models.load_model('./models/best_cnn_model.keras')
    m2 = keras.models.load_model('./models/best_column-by-column_model.keras')
    m3 = keras.models.load_model('./models/best_MDRNN_model.keras')
    
    return [m1, m2, m3, m4]

models = load_all_models()
model_names = ["CNN", "RNN_COL", "RNN_COL_ROW", "Transfer_learning"]


# inference 
import numpy as np

def run_inference(model, pil_image):
    # 1. Resize image to match model input shape
    img = pil_image.convert("RGB")
    img = img.resize((224, 224))
    
    # 2. Convert to numpy array
    img_array = np.array(img)
    
    # 3. Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    
    # 4. Predict
    prediction = model.predict(img_array)
    
    # 5. Get results (Assuming 2 classes: Crack vs No Crack)
    class_names = ["Crack","No Crack"] 
    result_index = np.argmax(prediction[0])
    confidence = prediction[0][result_index]
    
    return class_names[result_index], confidence


# 2. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.sidebar.image(input_image, caption="Original Image", use_container_width=True)

    st.subheader("Detection Results")
    
    # 3. Create the Grid (2 columns)
    col1, col2 = st.columns(2)
    
    # List of columns to iterate through
    grid = [col1, col2, col1, col2]

    for i, model in enumerate(models):
            with grid[i]:
                st.markdown(f"### {model_names[i]}")
                
                # --- START MODEL INFERENCE ---
                with st.spinner(f'Running {model_names[i]}...'):
                    label, confidence = run_inference(model, input_image)
                
                # Display results
                st.image(input_image, caption=f"Analyzed by {model_names[i]}", use_container_width=True)
                
                # Change color based on detection (optional)
                if "No" in label:
                    st.info(f"Result: {label} ({confidence:.2%})")
                else:
                    st.error(f"Detected: {label} ({confidence:.2%})")
                # --- END MODEL INFERENCE ---
                
                
    # Optional: Add a divider between rows
    if i == 1:
        st.divider()