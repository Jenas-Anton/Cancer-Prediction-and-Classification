import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ml_tools import run_ML_pipeline
import io

# Set page layout to 'centered'
st.set_page_config(page_title="Cancer Prediction and Classification App", layout="centered")

# Session state to manage the steps and data flow
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None

def input_window():
    # Page title and subtitle
    st.markdown("<h1 style='text-align: center;'>Cancer Prediction and Classification App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Analysis using Random Forest</h3>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload the dataset (should be .csv/.tsv file, the first column must have Gene_ID)", type=["csv", "tsv"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        # Save the uploaded file temporarily
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_path = file_path

        # Move to step 2 (result window)
        if st.button("Run Analysis"):
            st.session_state.step = 2

def result_window():
    # Back button to return to input page
    if st.button("⬅️ Back to Input Window"):
        st.session_state.step = 1

    st.markdown("<h1 style='text-align: center;'>Results </h1>", unsafe_allow_html=True)

    # Make sure a file is uploaded
    if st.session_state.uploaded_file:
        file_path = st.session_state.file_path

        # Fixed sidebar with options
        with st.sidebar:
            st.markdown("<h3>Reports</h3>", unsafe_allow_html=True)
            report_type = st.radio(
                label="Choose a report to view",
                options=["Prediction Result", "Confusion Matrix", "AUC Curve"],
                index=0
            )

        # Display corresponding reports
        if report_type == "Prediction Result":
            result = run_ML_pipeline("prediction_result", file_path, "RF")
            st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
            st.dataframe(result, width=900)  # Keeping the table width as before

            # Download option
            csv = result.to_csv().encode('utf-8')
            st.download_button(label="Download Prediction Result", data=csv, file_name='prediction_results.csv', mime='text/csv')

        elif report_type == "Confusion Matrix":
            st.markdown("<h2 style='text-align: center;'>Confusion Matrix</h2>", unsafe_allow_html=True)
            confusion_matrix_figure = run_ML_pipeline("confusion_matrix", file_path, "RF")
            
            # Display confusion matrix figure directly
            st.pyplot(confusion_matrix_figure)

            # Download option for Confusion Matrix
            buf = io.BytesIO()
            confusion_matrix_figure.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(label="Download Confusion Matrix", data=buf, file_name='confusion_matrix.png', mime="image/png")

        elif report_type == "AUC Curve":
            st.markdown("<h2 style='text-align: center;'>AUC Curve</h2>", unsafe_allow_html=True)
            auc_curve_figure = run_ML_pipeline("roc_auc_curve", file_path, "RF")
            
            # Display AUC curve figure directly
            st.pyplot(auc_curve_figure)

            # Download option for AUC Curve
            buf = io.BytesIO()
            auc_curve_figure.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(label="Download AUC Curve", data=buf, file_name='auc_curve.png', mime="image/png")

def main():
    if st.session_state.step == 1:
        input_window()
    elif st.session_state.step == 2:
        result_window()

if __name__ == "__main__":
    main()