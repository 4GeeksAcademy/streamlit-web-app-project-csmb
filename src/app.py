import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# Page configuration for a professional look
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

# Custom CSS to mimic your original wine-themed styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f5f2;
    }
    h1 {
        color: #6a1b2c;
        text-align: center;
    }
    .stButton>button {
        background-color: #6a1b2c;
        color: white;
        width: 100%;
        border-radius: 12px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4a0f1e;
        border: 1px solid #d4a056;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
# Note: Ensure the path is correct for your local environment
model_path = "src/redwine_model_k_neighbors_cluster.sav"

@st.cache_resource
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

class_dict = {
    "0": "Low Quality",
    "1": "Medium Quality",
    "2": "High Quality"
}

# Header
st.title("🍷 Wine Quality Predictor")
st.write("Introduce a wine's chemical characteristics to generate quality prediction")

# Create the form (using columns to mimic your grid)
with st.form("wine_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        val1 = st.slider("Volatile acidity", min_value=0.0, max_value=2.0, step=0.01, format="%.2f")
        val4 = st.slider("Chlorides", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        val7 = st.slider("pH", min_value=2.75, max_value=4.0, step=0.01, format="%.2f")

    with col2:
        val2 = st.slider("Citric acid", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        val5 = st.slider("Total sulfur dioxide", min_value=6.0, max_value=290.0, step=1.0)
        val8 = st.slider("Sulphates", min_value=0.3, max_value=2.0, step=0.01, format="%.2f")

    with col3:
        val3 = st.slider("Residual sugar", min_value=0.0, max_value=16.0, step=0.1, format="%.1f")
        val6 = st.slider("Density", min_value=0.98, max_value=1.1, step=0.01, format="%.3f")
        val9 = st.slider("Alcohol", min_value=8.4, max_value=15.0, step=0.01, format="%.2f")

    # Submit button
    submitted = st.form_submit_button("Predict Quality")
    features = [val1, val2, val3, val4, val5, val6, val7, val8, val9]

# Logic after button click
if submitted:
    data = np.array([features])
    prediction_raw = str(model.predict(data)[0])
    pred_class = class_dict.get(prediction_raw, "Unknown")
    
    # Success message with gold-ish styling
    st.info(f"### Prediction: {pred_class}")

    # Create radar chart

    categories = ['Volatile Acidity', 'Citric Acid', 'Res. Sugar', 
                'Chlorides', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol']
        
    # Simple Min-Max scaling for visualization purposes (based on your input ranges)
    ranges = [2.0, 1.0, 16.0, 1.0, 290.0, 1.1, 4.0, 2.0, 15.0]
    normalized_values = [v / r for v, r in zip(features, ranges)]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=pred_class,
        line_color='#6a1b2c'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f"Chemical Profile: {pred_class.title()}"
        )

    st.plotly_chart(fig)