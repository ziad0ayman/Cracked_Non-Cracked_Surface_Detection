import streamlit as st
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd 


st.set_page_config(layout="wide", page_title="Multi-Model Detection")
st.title("Multi-Model Image Detection Dashboard")

@st.cache_resource
def load_all_models():
    # Model 4: Transfer Learning (Manual Build)
    best_params = {'dense_units': 384, 'dropout_rate': 0.4228080027623846, 'lr': 0.00029786720003929296, 'unfreeze_layers': 20}
    
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    base_model = tf.keras.applications.DenseNet121(
        weights=None, 
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    m4 = tf.keras.Sequential([
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

    m4.load_weights("./models/best_crack_model_densenet.keras")
    
    # Load others
    m1 = keras.models.load_model('./models/best_cnn_model.keras')
    m2 = keras.models.load_model('./models/best_column-by-column_model.keras')
    m3 = keras.models.load_model('./models/best_MDRNN_model.keras')
    
    return [m1, m2, m3, m4]

models = load_all_models()
model_names = ["CNN", "RNN_COL", "RNN_COL_ROW", "Transfer Learning"]

def run_inference(model, pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    prediction = model.predict(img_array)
    
    class_names = ["Crack", "No Crack"] 
    result_index = np.argmax(prediction[0])
    return class_names[result_index], prediction[0][result_index]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.sidebar.image(input_image, caption="Original Image", use_container_width=True)

    st.subheader("Individual Model Results")
    col1, col2 = st.columns(2)
    grid = [col1, col2, col1, col2]
    
    # --- STORAGE FOR VOTES ---
    votes = []
    confidences=[]

    for i, model in enumerate(models):
        with grid[i]:
            st.markdown(f"**{model_names[i]}**")
            with st.spinner(f'Analyzing...'):
                label, confidence = run_inference(model, input_image)
                votes.append(label) # Collect the vote
                confidences.append(confidence)
                if "No" in label:
                    st.info(f"{label} ({confidence:.2%})")
                else:
                    st.error(f"{label} ({confidence:.2%})")
    
            
    st.divider()

    # --- FINAL ANALYSIS SECTION ---
    st.subheader("Result Analysis")

    # 1. Logic for Vote Counting
    # Count the votes
    vote_counts = Counter(votes)
    # Get the label with the most votes
    final_prediction, count = vote_counts.most_common(1)[0]

    # 2. Layout: Three columns for a professional dashboard look
    c1, c2, c3 = st.columns([1.5, 1.5, 1])

    with c1:
        st.markdown("###  Vote distribution")
        vote_df = pd.DataFrame({
            "Label": list(vote_counts.keys()),
            "Count": list(vote_counts.values()),
            "Color": ["#A32D2D" if l == "Crack" else "#185FA5" for l in vote_counts.keys()]
        })
        
        fig1, ax1 = plt.subplots(figsize=(5, 1.6))
        left = 0
        for _, row in vote_df.iterrows():
            ax1.barh(0, row["Count"], left=left, color=row["Color"], height=0.5)
            if row["Count"] > 0:
                ax1.text(left + row["Count"] / 2, 0, f"{row['Label']}\n{row['Count']}",
                        ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            left += row["Count"]
        
        ax1.set_xlim(0, len(models))
        ax1.set_xticks(range(len(models) + 1))
        ax1.set_yticks([])
        ax1.spines[:].set_visible(False)
        fig1.patch.set_alpha(0)
        ax1.set_facecolor("none")
        st.pyplot(fig1)

    with c2:
        
        st.markdown("###  Model confidence")
        bar_colors = ["#FCEBEB" if "Crack" == v and "No" not in v else "#E6F1FB" for v in votes]
        edge_colors = ["#A32D2D" if "Crack" == v and "No" not in v else "#185FA5" for v in votes]
        
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        bars = ax2.barh(model_names, confidences, color=bar_colors,
                        edgecolor=edge_colors, linewidth=1.5, height=0.5)
        ax2.set_xlim(0, 1.05)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.tick_params(axis="y", labelsize=10)
        ax2.tick_params(axis="x", labelsize=9, colors="#888780")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.yaxis.set_tick_params(length=0)
        
        for bar, conf, pred in zip(bars, confidences, votes):
            color = "#A32D2D" if "No" not in pred else "#185FA5"
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{conf:.1%}", va="center", ha="left", fontsize=9, color=color)
        
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        fig2.tight_layout()
        st.pyplot(fig2)

        with c3:
            agreement_count = vote_counts.most_common(1)[0][1]
            
            st.markdown("###  Verdict")
            st.metric("Winner", final_prediction)
            st.metric("Agreement", f"{agreement_count}/4")
            
            if "No" in final_prediction:
                st.success("Structure is likely Stable")
            else:
                st.error("Crack Detected")

    # --- DATA TABLE (Optional) ---
    with st.expander("View Raw Model Output"):
        df_results = pd.DataFrame({
            "Model": model_names,
            "Prediction": votes,
            "Confidence": [f"{c:.2%}" for c in confidences]
        })
        st.table(df_results)
            
            
    st.divider()
            
            
    # --- final VOTING LOGIC ---
    st.subheader(" Final Ensemble Result")
    
    
    # Display the final verdict
    v_col1, v_col2 = st.columns([1, 2])
    
    with v_col1:
        st.metric("Final Verdict", final_prediction)
    
    with v_col2:
        st.write(f"Confidence: **{count} out of {len(models)} models** agreed.")
        # Visual progress bar for the vote share
        progress = count / len(models)
        st.progress(progress)

    if "No" in final_prediction:
        st.success(f"The ensemble concludes there is **{final_prediction}**.")
    else:
        st.warning(f"The ensemble concludes a **{final_prediction}** has been detected.")


