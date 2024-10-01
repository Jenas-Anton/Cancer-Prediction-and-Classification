# Cancer Prediction and Classification: Leveraging Long Non-Coding RNAs for Enhanced Detection

## Table of Contents

1. [The Challenge: Complexity of Cancer](#the-challenge-complexity-of-cancer)
2. [Limitations of Current Methods](#limitations-of-current-methods)
3. [Our Solution: This App Incorporates Long Non-Coding RNAs](#our-solution-this-app-incorporates-long-non-coding-rnas-lncrnas)
    - [Role of lncRNAs](#role-of-lncrnas)
    - [Previous Oversight](#previous-oversight)
4. [Advantages of This App](#advantages-of-this-app)
    - [Precision and Sensitivity](#1-precision-and-sensitivity)
    - [Cancer-Type Specificity](#2-cancer-type-specificity)
    - [Comprehensive Data Integration](#3-comprehensive-data-integration)
5. [The Urgency](#the-urgency)
    - [Critical Juncture](#critical-juncture)
    - [Need for Evolution](#need-for-evolution)
6. [Functions Overview](#functions-overview)
    - [`get_data()`](#get_data)
    - [`feature_select(df)`](#feature_selectdf)
    - [`run_ML_pipeline(report, file_path, model_id)`](#run_ml_pipelinereport-file_path-model_id)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

---

<details>
  <summary><strong>## The Challenge: Complexity of Cancer</strong></summary>

Cancer is not a single disease but a spectrum of hundreds of diseases, each with numerous variations driven by genetic, epigenetic, and molecular changes.

### Limitations of Current Methods

- **Traditional Methods**: Focus on mutations in protein-coding genes or protein biomarkers.
- **Limitations**: These approaches often detect cancer too late, lack sensitivity, and fail to predict cancer behavior accurately.
- **Missed Indicators**: Critical cancer development signals, particularly in aggressive or early-stage cancers, may be overlooked.
</details>

<details>
  <summary><strong>## Our Solution: This App Incorporates Long Non-Coding RNAs (lncRNAs)</strong></summary>

### Role of lncRNAs

Long non-coding RNAs (lncRNAs) are essential regulators of gene expression, influencing tumor growth, spread, and resistance to treatment.

### Previous Oversight

Traditional methods have largely ignored lncRNAs, missing out on vital data necessary for fully understanding cancer development and progression.
</details>

<details>
  <summary><strong>## Advantages of This App</strong></summary>

### 1. Precision and Sensitivity
- **Early Detection**: This app captures regulatory signals invisible to traditional methods, enhancing early cancer detection.

### 2. Cancer-Type Specificity
- **Predictive Accuracy**: Identifies specific types of cancer based on unique lncRNA signatures, offering high specificity.

### 3. Comprehensive Data Integration
- **Holistic View**: The app combines genomic, expression, epigenetic, and molecular interaction data to provide a detailed cancer molecular landscape.
</details>

<details>
  <summary><strong>## The Urgency</strong></summary>

### Critical Juncture
Traditional methods have limitations and can only advance so much further. New data from next-generation sequencing and computational biology offer opportunities to deepen our understanding.

### Need for Evolution
- **Adaptation**: This app represents the next frontier in cancer detection, leveraging lncRNAs to advance prediction, diagnosis, and treatment.
- **Impact**: This approach offers potential for earlier detection, more effective treatment, and the possibility of saving countless lives.
</details>

<details>
  <summary><strong>## Functions Overview</strong></summary>

### `get_data()`
- **Purpose**: Loads and preprocesses the dataset from `feature_set.csv`, combining it with positive and negative lncRNA gene lists. It selects important features using `ExtraTreesClassifier` and returns the feature matrix (`X`) and labels (`y`).

### `feature_select(df)`
- **Purpose**: Selects the most important features from the dataframe using `ExtraTreesClassifier` based on feature importance scores. Returns a dataframe containing only the most relevant features.

### `run_ML_pipeline(report, file_path, model_id)`
- **Purpose**: Executes a machine learning pipeline including training, prediction, and evaluation. Supports generating a confusion matrix, ROC AUC curve, or prediction results with the most important features.
</details>

<details>
  <summary><strong>## Usage</strong></summary>

### 1. Loading Data

To begin using this app, you'll need to load and preprocess your dataset using the `get_data()` function. Ensure your dataset is in the correct format (e.g., `feature_set.csv`), and that you have both positive and negative lncRNA gene lists available.

```python
X, y = get_data()
