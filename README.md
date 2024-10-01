# 🧬 Cancer Prediction and Classification: Leveraging Long Non-Coding RNAs for Enhanced Detection

## 📚 Table of Contents

1. [The Challenge: Complexity of Cancer](#the-challenge-complexity-of-cancer)
2. [Limitations of Current Methods](#limitations-of-current-methods)
3. [Our Solution: This App Incorporates Long Non-Coding RNAs](#our-solution-this-app-incorporates-long-non-coding-rnas-lncrnas)
4. [Advantages of This App](#advantages-of-this-app)
5. [The Urgency](#the-urgency)
6. [Functions Overview](#functions-overview)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

## 🧩 The Challenge: Complexity of Cancer

Cancer is not a single disease but a spectrum of hundreds of diseases, each with numerous variations driven by genetic, epigenetic, and molecular changes.

### 🚫 Limitations of Current Methods

- **Traditional Methods**: Focus on mutations in protein-coding genes or protein biomarkers.
- **Limitations**: These approaches often detect cancer too late, lack sensitivity, and fail to predict cancer behavior accurately.
- **Missed Indicators**: Critical cancer development signals, particularly in aggressive or early-stage cancers, may be overlooked.

## 💡 Our Solution: This App Incorporates Long Non-Coding RNAs (lncRNAs)

### 🧬 Role of lncRNAs

Long non-coding RNAs (lncRNAs) are essential regulators of gene expression, influencing tumor growth, spread, and resistance to treatment.

### 🔍 Previous Oversight

Traditional methods have largely ignored lncRNAs, missing out on vital data necessary for fully understanding cancer development and progression.

## 🌟 Advantages of This App

1. **🎯 Precision and Sensitivity**
   - Early Detection: This app captures regulatory signals invisible to traditional methods, enhancing early cancer detection.

2. **🔬 Cancer-Type Specificity**
   - Predictive Accuracy: Identifies specific types of cancer based on unique lncRNA signatures, offering high specificity.

3. **🧠 Comprehensive Data Integration**
   - Holistic View: The app combines genomic, expression, epigenetic, and molecular interaction data to provide a detailed cancer molecular landscape.

## ⏰ The Urgency

### 🚀 Critical Juncture
Traditional methods have limitations and can only advance so much further. New data from next-generation sequencing and computational biology offer opportunities to deepen our understanding.

### 🔄 Need for Evolution
- **Adaptation**: This app represents the next frontier in cancer detection, leveraging lncRNAs to advance prediction, diagnosis, and treatment.
- **Impact**: This approach offers potential for earlier detection, more effective treatment, and the possibility of saving countless lives.

## 🛠️ Functions Overview

### get_data()
- **Purpose**: Loads and preprocesses the dataset from feature_set.csv, combining it with positive and negative lncRNA gene lists. It selects important features using ExtraTreesClassifier and returns the feature matrix (X) and labels (y).

```python
X, y = get_data()
```

### feature_select(df)
- **Purpose**: Selects the most important features from the dataframe using ExtraTreesClassifier based on feature importance scores. Returns a dataframe containing only the most relevant features.

```python
selected_features_df = feature_select(input_dataframe)
```

### run_ML_pipeline(report, file_path, model_id)
- **Purpose**: Executes a machine learning pipeline including training, prediction, and evaluation. Supports generating a confusion matrix, ROC AUC curve, or prediction results with the most important features.
- **Parameters**:
  - report: The type of report to generate ('confusion_matrix', 'roc_auc_curve', or 'prediction_result').
  - file_path: Path to the user's data file.
  - model_id: The model to use ('RF' for Random Forest).

```python
run_ML_pipeline(report='confusion_matrix', file_path='path/to/data.csv', model_id='RF')
```

## 🚀 Usage

### 1. Loading Data

To begin using this app, you'll need to load and preprocess your dataset using the `get_data()` function. Ensure your dataset is in the correct format (e.g., feature_set.csv), and that you have both positive and negative lncRNA gene lists available.

```python
X, y = get_data()
```

### 2. Running the Application

The application uses **Streamlit** to provide a user-friendly interface for cancer prediction and classification.

To run the application, open a terminal and type the following command:

```bash
streamlit run app.py
```

### 3. Input File

To make predictions, you will need to upload your data through the app's interface.

1. In the web application, you will see an **upload box**.
2. Upload the input file (input.csv) that contains your data.
   
Ensure your input file is in .csv format and properly structured with the relevant features.

## 📊 Visual Reports: Confusion Matrix and AUC Curve

Once the machine learning pipeline runs, the application generates reports to help visualize the model's performance.

### Confusion Matrix

The **confusion matrix** is a table that summarizes the classification performance, showing the number of:
- True Positives (TP): Correctly predicted positive cases.
- True Negatives (TN): Correctly predicted negative cases.
- False Positives (FP): Incorrectly predicted as positive when it's actually negative.
- False Negatives (FN): Incorrectly predicted as negative when it's actually positive.

This matrix helps you understand the accuracy of the predictions and where the model might be making mistakes.

```python
run_ML_pipeline(report='confusion_matrix', file_path='path_to_data_file', model_id='RF')
```

### AUC ROC (Receiver Operating Characteristic) Curve

The **ROC AUC Curve** (Receiver Operating Characteristic - Area Under the Curve) is a graphical representation of a classification model's performance. It illustrates the relationship between:

- True Positive Rate (Sensitivity): Proportion of actual positives that are correctly identified.
- False Positive Rate (1 - Specificity): Proportion of actual negatives that are incorrectly identified as positives.

The AUC score, ranging from 0.5 to 1, indicates how well the model distinguishes between classes:

- 0.5: Random guessing (the model has no discriminative power).
- 1: Perfect classification (the model distinguishes all classes correctly).

```python
run_ML_pipeline(report='roc_auc_curve', file_path='path_to_data_file', model_id='RF')
```

## 🎉 Conclusion

This app offers a groundbreaking approach by incorporating long non-coding RNAs to enhance cancer detection, providing earlier diagnosis and more targeted treatment strategies. This tool has the potential to revolutionize cancer research and save lives by offering precision and sensitivity beyond the reach of traditional methods.

---

##🚀 Try It Out!
You can test this Cancer Prediction and Classification tool for free on our Hugging Face Space:
Cancer Prediction & Classification Space
Feel free to upload your data and explore the power of lncRNA-based cancer prediction and classification!
