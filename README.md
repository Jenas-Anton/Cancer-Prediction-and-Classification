-->The Challenge:
    Complexity of Cancer: Cancer is not a single disease but a spectrum of hundreds of diseases, each with numerous variations driven by genetic, epigenetic, and 
    molecular changes.
--> Limitations of Current Methods:
    Traditional Methods: Focus on mutations in protein-coding genes or protein biomarkers.
-->Limitations: Often detect cancer too late, lack sensitivity, and fail to predict cancer behavior accurately.
                Missed Indicators: Critical cancer development signals, particularly in aggressive or early-stage cancers, may be overlooked.
-->Our Solution: CRlncRC
                Incorporates Long Non-Coding RNAs (lncRNAs):

-->Role of lncRNAs: Essential regulators of gene expression, influencing tumor growth, spread, and resistance to treatment.
-->Previous Oversight: Traditional methods ignore lncRNAs, which are vital for understanding cancer.


-->Advantages of CRlncRC:
1) Precision and Sensitivity:
    Early Detection: Captures regulatory signals invisible to traditional methods, enhancing early cancer detection.
2)Cancer-Type Specificity:
    Predictive Accuracy: Identifies specific types of cancer based on unique lncRNA signatures, offering high specificity.
3)Comprehensive Data Integration:
  Holistic View: Combines genomic, expression, epigenetic, and molecular interaction data to provide a detailed cancer molecular landscape.


-->The Urgency:
    Critical Juncture:Traditional methods have limitations and can only advance so much further.
                      New data from next-generation sequencing and computational biology offer opportunities to deepen our understanding.
    Need for Evolution: 
                        Adaptation: CRlncRC represents the next frontier in cancer detection, leveraging lncRNAs to advance prediction, diagnosis, and treatment.
                        Impact: Offers potential for earlier detection, more effective treatment, and the possibility of saving countless lives.



THE FUNCTIONS USED 

get_data()
      Purpose: Loads and preprocesses the dataset from feature_set.csv, combining it with positive and negative lncRNA gene lists. It selects important features 
       using ExtraTreesClassifier and returns the feature matrix (X) and labels (y).


feature_select(df)
    Purpose: Selects the most important features from the dataframe using ExtraTreesClassifier based on feature importance scores. Returns a dataframe containing 
    only the most relevant features.

run_ML_pipeline(report, file_path, model_id)
    Purpose: Executes a machine learning pipeline including training, prediction, and evaluation. Supports generating a confusion matrix, ROC AUC curve, or 
    prediction results with the most important features.
    Parameters:
    report: The type of report to generate ('confusion_matrix', 'roc_auc_curve', or 'prediction_result').
    file_path: Path to the user's data file.
    model_id: The model to use ('RF' for Random Forest).





