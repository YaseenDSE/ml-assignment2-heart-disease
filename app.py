"""
Heart Disease Prediction App
Machine Learning Assignment 2
M.Tech (AIML/DSE) - BITS Pilani
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #DC143C;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #2C3E50;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #DC143C;
        margin: 10px 0;
    }
    .positive-prediction {
        background-color: #FFE5E5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #DC143C;
        margin: 15px 0;
    }
    .negative-prediction {
        background-color: #E5FFE5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load all trained models from the model directory"""
    models_dict = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/knn_classifier.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost_classifier.pkl'
    }
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            models_dict[model_name] = joblib.load(file_path)
    
    # Load scaler
    scaler = None
    if os.path.exists('model/scaler.pkl'):
        scaler = joblib.load('model/scaler.pkl')
    
    return models_dict, scaler

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # App Header
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("### Machine Learning Assignment 2 | M.Tech Data Science | BITS Pilani")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading trained models..."):
        models_dict, scaler = load_models()
    
    if not models_dict:
        st.error("❌ No models found! Please ensure model files are in the 'model/' directory.")
        return
    
    st.success(f"✅ Successfully loaded {len(models_dict)} models!")
    
    # ========================================================================
    # SIDEBAR - MODEL SELECTION & FILE UPLOAD
    # ========================================================================
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection dropdown
    selected_model_name = st.sidebar.selectbox(
        "Select ML Model:",
        options=list(models_dict.keys()),
        index=4  # Default to Random Forest (best performer)
    )
    
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (test data)",
        type=['csv'],
        help="Upload heart disease test dataset in CSV format"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Information")
    st.sidebar.info(f"""
    **Selected Model:** {selected_model_name}
    
    **Dataset:** Heart Disease
    
    **Features:** 13 clinical parameters
    
    **Target:** Binary (0=No disease, 1=Disease)
    
    **Classification:** Binary Classification
    """)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    if uploaded_file is not None:
        # Load uploaded data
        try:
            test_data = pd.read_csv(uploaded_file)
            st.markdown('<div class="sub-header">📋 Uploaded Dataset Preview</div>', 
                       unsafe_allow_html=True)
            st.dataframe(test_data.head(10), use_container_width=True)
            
            st.markdown(f"**Dataset Shape:** {test_data.shape[0]} rows × {test_data.shape[1]} columns")
            
            # Check if target column exists (common names: target, output, num, diagnosis)
            target_col = None
            possible_targets = ['target', 'output', 'num', 'diagnosis', 'condition']
            
            for col in possible_targets:
                if col in test_data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                st.warning("⚠️ Target column not found. Showing predictions only (no evaluation metrics).")
                X_test = test_data
                y_test = None
            else:
                X_test = test_data.drop(target_col, axis=1)
                y_test = test_data[target_col]
            
            # ================================================================
            # MAKE PREDICTIONS
            # ================================================================
            st.markdown("---")
            st.markdown('<div class="sub-header">🔮 Model Predictions</div>', 
                       unsafe_allow_html=True)
            
            selected_model = models_dict[selected_model_name]
            
            # Apply scaling for models that need it
            models_needing_scaling = ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']
            
            if selected_model_name in models_needing_scaling and scaler is not None:
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Make predictions
            with st.spinner("Making predictions..."):
                predictions = selected_model.predict(X_test_processed)
                prediction_proba = selected_model.predict_proba(X_test_processed)
            
            # Display predictions with risk levels
            pred_df = pd.DataFrame({
                'Sample Index': range(len(predictions)),
                'Predicted': predictions,
                'Risk Level': ['🔴 High Risk' if p == 1 else '🟢 Low Risk' for p in predictions],
                'Confidence': [f"{max(proba):.2%}" for proba in prediction_proba]
            })
            
            if y_test is not None:
                pred_df['Actual'] = y_test.values
                pred_df['Correct'] = pred_df['Predicted'] == pred_df['Actual']
                pred_df['Match'] = ['✅' if c else '❌' for c in pred_df['Correct']]
            
            st.dataframe(pred_df.head(20), use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            with col1:
                disease_count = sum(predictions == 1)
                st.metric("Patients with Heart Disease", disease_count)
            with col2:
                no_disease_count = sum(predictions == 0)
                st.metric("Patients without Heart Disease", no_disease_count)
            
            # ================================================================
            # EVALUATION METRICS (if ground truth available)
            # ================================================================
            if y_test is not None:
                st.markdown("---")
                st.markdown('<div class="sub-header">📈 Model Evaluation Metrics</div>', 
                           unsafe_allow_html=True)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='binary', zero_division=0)
                recall = recall_score(y_test, predictions, average='binary', zero_division=0)
                f1 = f1_score(y_test, predictions, average='binary', zero_division=0)
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Precision", f"{precision:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Recall", f"{recall:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("F1 Score", f"{f1:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # ============================================================
                # CONFUSION MATRIX
                # ============================================================
                st.markdown("---")
                st.markdown('<div class="sub-header">🎯 Confusion Matrix</div>', 
                           unsafe_allow_html=True)
                
                cm = confusion_matrix(y_test, predictions)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='RdYlGn_r',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    ax=ax,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 16, 'weight': 'bold'}
                )
                ax.set_xlabel('Predicted Diagnosis', fontsize=12, fontweight='bold')
                ax.set_ylabel('Actual Diagnosis', fontsize=12, fontweight='bold')
                ax.set_title(f'Confusion Matrix - {selected_model_name}', 
                           fontsize=14, fontweight='bold', pad=20)
                
                st.pyplot(fig)
                
                # Interpretation
                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                **Interpretation:**
                - **True Negatives (TN):** {tn} - Correctly identified healthy patients
                - **True Positives (TP):** {tp} - Correctly identified patients with heart disease
                - **False Positives (FP):** {fp} - Healthy patients incorrectly flagged (Type I Error)
                - **False Negatives (FN):** {fn} - Patients with disease incorrectly cleared (Type II Error - Critical!)
                """)
                
                # ============================================================
                # CLASSIFICATION REPORT
                # ============================================================
                st.markdown("---")
                st.markdown('<div class="sub-header">📊 Detailed Classification Report</div>', 
                           unsafe_allow_html=True)
                
                report = classification_report(y_test, predictions, 
                                              target_names=['No Disease', 'Disease'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.dataframe(
                    report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']),
                    use_container_width=True
                )
            
            else:
                st.info("ℹ️ No ground truth labels provided. Showing predictions only.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has 13 feature columns matching the heart disease dataset format.")
    
    else:
        # Landing page when no file is uploaded
        st.info("👈 Please upload a CSV file from the sidebar to begin predictions")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use This App:")
        st.markdown("""
        1. **Select a model** from the dropdown in the sidebar
        2. **Upload your test dataset** (CSV format with heart disease features)
        3. **View predictions** with risk levels and confidence scores
        4. **Analyze** evaluation metrics, confusion matrix, and classification report
        
        **Expected CSV Format:**
        - 13 feature columns (clinical parameters)
        - Optional 'target' column (0 or 1) for evaluation
        """)
        
        # Show sample data format
        st.markdown("### 📋 Sample Data Format:")
        sample_data = pd.DataFrame({
            'age': [63, 67],
            'sex': [1, 1],
            'cp': [3, 0],
            'trestbps': [145, 160],
            'chol': [233, 286],
            'fbs': [1, 0],
            'restecg': [0, 0],
            'thalach': [150, 108],
            'exang': [0, 1],
            'oldpeak': [2.3, 1.5],
            'slope': [0, 1],
            'ca': [0, 3],
            'thal': [1, 2],
            'target': [1, 1]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        **Feature Descriptions:**
        - **age:** Age in years
        - **sex:** 1=male, 0=female
        - **cp:** Chest pain type (0-3)
        - **trestbps:** Resting blood pressure
        - **chol:** Serum cholesterol
        - **fbs:** Fasting blood sugar > 120 mg/dl
        - **restecg:** Resting ECG results (0-2)
        - **thalach:** Maximum heart rate achieved
        - **exang:** Exercise induced angina (1=yes, 0=no)
        - **oldpeak:** ST depression
        - **slope:** Slope of peak exercise ST segment
        - **ca:** Number of major vessels (0-3)
        - **thal:** Thalassemia (1-3)
        """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()