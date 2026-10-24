import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Prediction Assistant")
st.write("""
This application uses a machine learning model optimized with **5 key features** 
to predict whether a tumor is **Malignant** or **Benign**.
""")

# 2. Load the Saved Model and Features safely
@st.cache_resource
def load_model_assets():
    model = joblib.load('breast_cancer_model.pkl')
    features = joblib.load('selected_features.pkl')
    return model, features

try:
    clf, selected_features = load_model_assets()
    
    # 3. Create User Inputs for the 5 selected features
    st.subheader("📊 Input Patient Clinical Metrics")
    
    # We setup slider ranges based on the standard breast cancer dataset scales
    mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0, step=0.1)
    mean_concave_pts = st.slider("Mean Concave Points", 0.0, 0.2, 0.05, step=0.001, format="%.4f")
    texture_error = st.slider("Texture Error", 0.1, 5.0, 1.2, step=0.05)
    worst_compactness = st.slider("Worst Compactness", 0.01, 1.0, 0.25, step=0.01)
    worst_concavity = st.slider("Worst Concavity", 0.0, 1.2, 0.3, step=0.01)
    
    # 4. Predict Button and Result Processing
    if st.button("Analyze Metrics", type="primary"):
        # Format inputs exactly in the order the model expects
        input_data = np.array([[
            mean_radius, 
            mean_concave_pts, 
            texture_error, 
            worst_compactness, 
            worst_concavity
        ]])
        
        # Get numerical prediction and probabilities
        prediction = clf.predict(input_data)[0]
        probabilities = clf.predict_proba(input_data)[0]
        
        st.write("---")
        st.subheader("🎯 Diagnostic Analysis")
        
        # Breast cancer target convention: 0 = Malignant, 1 = Benign
        if prediction == 0:
            st.error(f"⚠️ **Result:** The model predicts the tumor is **Malignant** ({probabilities[0]*100:.1f}% confidence).")
        else:
            st.success(f"✅ **Result:** The model predicts the tumor is **Benign** ({probabilities[1]*100:.1f}% confidence).")
            
except FileNotFoundError:
    st.error("Error: Could not find `breast_cancer_model.pkl` or `selected_features.pkl`. Make sure you run the saving cells in your notebook first!")
