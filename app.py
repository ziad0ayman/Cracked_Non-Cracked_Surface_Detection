import json
import os

import streamlit as st
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from collections import Counter

from src.utils.config import get_model, get_global
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    layout="wide",
    page_title="Crack Detection — Ensemble Dashboard",
    page_icon="🏗️",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    h1 { color: #1A1A2E; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #1A1A2E; font-weight: 600; }
    h3 { color: #16213E; font-weight: 600; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="metric-container"] > label { font-size: 0.85rem !important; color: #6c757d !important; }
    div[data-testid="metric-container"] > div { font-size: 1.8rem !important; font-weight: 700 !important; }
    section[data-testid="stFileUploader"] {
        padding: 1rem; border: 2px dashed #dee2e6; border-radius: 12px; background: #fafbfc;
    }
    section[data-testid="stFileUploader"]:hover { border-color: #0f3460; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #0f3460, #e94560); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Ensemble Detector")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption=None, use_container_width=True)
    st.markdown("---")
    st.markdown("**Models in ensemble:**")
    st.dataframe(
        pd.DataFrame({
            "Model": ["CNN", "Column RNN", "MD-RNN", "DenseNet121"],
            "Type": ["ConvNet", "BiLSTM(Col)", "BiLSTM(Row+Col)", "Transfer"],
        }),
        hide_index=True, use_container_width=True,
    )

# ── Main title ─────────────────────────────────────────────────────────────
st.title("🏗️ Multi-Model Crack Detection")
st.markdown(
    '<p style="color:#6c757d; font-size:1.05rem; margin-top:-8px;">'
    "Upload a concrete surface image — four deep-learning models analyse it and vote on the result."
    "</p>",
    unsafe_allow_html=True,
)

# ── Load evaluation data ──────────────────────────────────────────────────
eval_path = os.path.join(os.path.dirname(__file__), "models", "evaluation_results.json")
with open(eval_path, encoding="utf-8") as f:
    EVAL_DATA = json.load(f)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_detect, tab_perf = st.tabs(["🔍 Detection Dashboard", "📊 Model Performance"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Detection (original functionality)
# ═══════════════════════════════════════════════════════════════════════════
with tab_detect:
    @st.cache_resource(show_spinner="Loading ensemble models...")
    def load_all_models():
        transfer_cfg = get_model("transfer")
        global_cfg = get_global()
        im_size = tuple(global_cfg["image_size"]) + (3,)

        data_augmentation = tf.keras.Sequential([
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="data_augmentation")
        base_model = tf.keras.applications.DenseNet121(
            weights=None, include_top=False, input_shape=im_size
        )
        m4 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=im_size),
            data_augmentation, base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(transfer_cfg["dense_units"], use_bias=False,
                         kernel_regularizer=regularizers.l2(1e-5)),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Dropout(transfer_cfg["dropout_rate"]),
            layers.Dense(global_cfg["num_classes"], activation="softmax"),
        ])
        m4.load_weights("./models/best_crack_model_densenet.keras")
        m1 = keras.models.load_model("./models/best_cnn_model.keras")
        m2 = keras.models.load_model("./models/best_column-by-column_model.keras")
        m3 = keras.models.load_model("./models/best_MDRNN_model.keras")
        return [m1, m2, m3, m4]

    models = load_all_models()
    model_names = ["CNN", "Column RNN", "MD-RNN", "DenseNet121"]
    model_labels = ["ConvNet 6-block", "BiLSTM (Column)", "BiLSTM (Row+Col)", "Transfer DenseNet121"]

    def run_inference(model, pil_image):
        img = pil_image.convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        prediction = model.predict(img_array, verbose=0)
        result_index = np.argmax(prediction[0])
        confidence = prediction[0][result_index]
        label = "Crack" if result_index == 0 else "No Crack"
        return label, confidence, prediction[0]

    if uploaded_file is None:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.info("👈 Upload a JPEG or PNG image from the sidebar to get started.", icon="📸")
    else:
        votes, confidences, probabilities = [], [], []
        with st.status("Running inference on 4 models...", expanded=True) as status:
            for name, model in zip(model_names, models):
                label, conf, probs = run_inference(model, input_image)
                votes.append(label)
                confidences.append(conf)
                probabilities.append(probs)
                st.write(f"✅ **{name}** → {label} ({conf:.1%})")
            status.update(label="All models finished.", state="complete")

        st.subheader("🔍 Individual Predictions")
        cols = st.columns(4)
        colors = {"Crack": "#e94560", "No Crack": "#0f3460"}
        bg_colors = {"Crack": "#fff5f5", "No Crack": "#f0f7ff"}
        for i, (name, label, conf) in enumerate(zip(model_names, votes, confidences)):
            c = colors[label]; bg = bg_colors[label]
            with cols[i]:
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {c}40;border-radius:14px;
                            padding:18px 12px 14px;text-align:center;height:100%;">
                    <div style="font-size:0.8rem;color:#6c757d;margin-bottom:2px;">{model_labels[i]}</div>
                    <div style="font-size:1.1rem;font-weight:700;">{name}</div>
                    <div style="font-size:2.2rem;font-weight:800;color:{c};margin:6px 0 2px;">
                        {"⚠️" if label=="Crack" else "✅"} {label}
                    </div>
                    <div style="font-size:1.3rem;font-weight:600;color:{c};">{conf:.1%}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Ensemble Analysis")
        vote_counts = Counter(votes)
        final_prediction, count = vote_counts.most_common(1)[0]
        col_left, col_right = st.columns([1.1, 2])

        with col_left:
            crack_prob = np.mean([p[0] for p in probabilities])
            st.markdown("### ⚖️ Verdict")
            verdict_color = "#e94560" if final_prediction == "Crack" else "#0f3460"
            st.markdown(f"""
            <div style="background:{bg_colors[final_prediction]};border:2px solid {verdict_color}40;
                        border-radius:16px;padding:24px;text-align:center;">
                <div style="font-size:3rem;">{"⚠️" if final_prediction=="Crack" else "✅"}</div>
                <div style="font-size:1.8rem;font-weight:800;color:{verdict_color};">{final_prediction}</div>
                <div style="font-size:0.95rem;color:#6c757d;margin-top:6px;">{count}/{len(models)} models agree</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Avg Confidence")
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Crack", "No Crack"],
                values=[crack_prob, 1 - crack_prob],
                marker=dict(colors=["#e94560", "#0f3460"]),
                textinfo="label+percent", textposition="outside", hole=0.5, showlegend=False,
            )])
            fig_pie.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown("### 📊 Confidence per Model")
            fig = go.Figure(data=[go.Bar(
                x=confidences, y=model_names, orientation="h",
                marker=dict(color=["#e94560" if v == "Crack" else "#0f3460" for v in votes]),
                text=[f"{c:.1%}" for c in confidences], textposition="outside",
                textfont=dict(size=13, color="#333"),
            )])
            fig.update_layout(height=280, xaxis=dict(range=[0, 1.15], tickformat=".0%", title=None),
                              yaxis=dict(title=None), margin=dict(l=0, r=40, t=0, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(size=12), xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🗳️ Vote Distribution")
            vote_df = pd.DataFrame({"Label": list(vote_counts.keys()), "Count": list(vote_counts.values())})
            fig_vote = px.bar(vote_df, x="Label", y="Count", text="Count",
                              color="Label", color_discrete_map={"Crack": "#e94560", "No Crack": "#0f3460"})
            fig_vote.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(size=12), xaxis_showgrid=False, yaxis_showgrid=False, showlegend=False)
            fig_vote.update_traces(textposition="outside", textfont=dict(size=16))
            st.plotly_chart(fig_vote, use_container_width=True)

        st.divider()
        st.subheader("📋 Full Probability Breakdown")
        rows = []
        for name, probs, vote in zip(model_names, probabilities, votes):
            rows.append({"Model": name, "Crack": f"{probs[0]:.3%}",
                         "No Crack": f"{probs[1]:.3%}", "Prediction": vote})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("📊 Offline Evaluation — Test Set Performance")

    # --- Summary metrics table ---
    summary_rows = []
    for model_name, metrics in EVAL_DATA.items():
        summary_rows.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.2%}",
            "F1 (Cracked)": f"{metrics['f1_cracked']:.2%}",
            "F1 (Non-Cracked)": f"{metrics['f1_non_cracked']:.2%}",
            "F1 (Macro)": f"{metrics['f1_macro']:.2%}",
            "Params": f"{metrics['params']:,}",
        })
    df_summary = pd.DataFrame(summary_rows)

    st.dataframe(
        df_summary,
        column_config={
            "Model": st.column_config.TextColumn("Model"),
            "Accuracy": st.column_config.TextColumn("Accuracy"),
            "F1 (Cracked)": st.column_config.TextColumn("F1 (Cracked)"),
            "F1 (Non-Cracked)": st.column_config.TextColumn("F1 (Non-Cracked)"),
            "F1 (Macro)": st.column_config.TextColumn("F1 (Macro)"),
            "Params": st.column_config.TextColumn("Parameters"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")

    # --- Accuracy bar chart ---
    st.markdown("### 🎯 Accuracy Comparison")
    acc_df = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"]} for k, v in EVAL_DATA.items()
    ])
    fig_acc = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy",
                     color="Model",
                     color_discrete_map={
                         "CNN": "#0f3460", "Column-By-Column RNN": "#e94560",
                         "MD-RNN": "#533483", "DenseNet121": "#0f3460",
                     })
    fig_acc.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig_acc.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0.8, 1])
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")

    # --- Per-class metrics ---
    st.markdown("### 📈 Per-Class Metrics")

    per_class_rows = []
    for model_name, m in EVAL_DATA.items():
        per_class_rows.append({"Model": model_name, "Class": "Cracked",
                               "Precision": m["precision_cracked"], "Recall": m["recall_cracked"],
                               "F1-Score": m["f1_cracked"]})
        per_class_rows.append({"Model": model_name, "Class": "Non-Cracked",
                               "Precision": m["precision_non_cracked"], "Recall": m["recall_non_cracked"],
                               "F1-Score": m["f1_non_cracked"]})
    df_pc = pd.DataFrame(per_class_rows)

    fig_pc = px.bar(df_pc, x="Model", y="F1-Score", color="Class", barmode="group",
                    text_auto=".2%", color_discrete_map={"Cracked": "#e94560", "Non-Cracked": "#0f3460"})
    fig_pc.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         yaxis_tickformat=".0%", yaxis_range=[0.7, 1])
    st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("---")

    # --- Confusion matrices ---
    st.markdown("### 🔢 Confusion Matrices")

    model_keys = list(EVAL_DATA.keys())
    cm_cols = st.columns(2)

    for idx, model_name in enumerate(model_keys):
        cm = EVAL_DATA[model_name]["confusion_matrix"]
        cm_np = np.array(cm)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_np,
            x=["Predicted Cracked", "Predicted Non-Cracked"],
            y=["Actual Cracked", "Actual Non-Cracked"],
            text=cm_np,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#333"},
            colorscale="Blues",
            showscale=False,
            hoverongaps=False,
        ))
        fig_cm.update_layout(
            title=f"{model_name}",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            xaxis_title=None,
            yaxis_title=None,
        )
        with cm_cols[idx % 2]:
            st.plotly_chart(fig_cm, use_container_width=True)

    st.caption("Rows = ground truth, Columns = prediction.  •  Cracked = class 0,  Non-Cracked = class 1")
