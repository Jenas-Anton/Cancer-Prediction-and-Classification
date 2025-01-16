import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ml_tools import get_data, cancer_mapping

def run_hybrid_pipeline(report, file_path, algorithm=None):
    """
    Implements a hybrid model combining Random Forest and XGBoost using a weighted average of their predictions.

    :param report: Type of report to generate ('confusion_matrix', 'roc_auc_curve', 'prediction_result').
    :param file_path: Path to the user's input data file.
    :param algorithm: (Optional) Algorithm identifier; included for compatibility with app.py.
    :return: The requested report (figure or DataFrame).
    """
    # Load and preprocess the user's data
    user_x = pd.read_csv(file_path, sep=',' if file_path.endswith('.csv') else '\t', index_col=0).fillna(0)
    test_x = user_x.iloc[:, 1:]  # Exclude Gene_ID

    # Get training data
    train_X, train_y = get_data()
    column_names = list(test_x.columns.values)  # Match user input columns with training data
    train_X = train_X[column_names]

    # Convert labels to binary (0 and 1)
    train_y = np.where(train_y == -1, 0, 1)

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.3, random_state=0)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(train_X, train_y)
    rf_pred_proba = rf_model.predict_proba(test_X)[:, 1]

    # Train XGBoost model
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_X, label=test_y)
    xgb_params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': -1
    }
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    xgb_pred_proba = xgb_model.predict(dtest)

    # Combine predictions (weighted average)
    final_pred_proba = 0.6 * rf_pred_proba + 0.4 * xgb_pred_proba
    final_pred = (final_pred_proba > 0.5).astype(int)

    if report == "confusion_matrix":
        # Generate confusion matrix
        cm = confusion_matrix(test_y, final_pred)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')

        return fig

    elif report == "roc_auc_curve":
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(test_y, final_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

        return fig

    elif report == "prediction_result":
        # Predict for the user's data
        dpredict = xgb.DMatrix(test_x)
        rf_user_pred = rf_model.predict(test_x)
        xgb_user_pred = xgb_model.predict(dpredict)
        final_user_pred = (0.6 * rf_user_pred + 0.4 * xgb_user_pred > 0.5).astype(int)

        user_x['predicted_value'] = final_user_pred

        # Feature importance (from Random Forest)
        feature_importances = rf_model.feature_importances_
        important_features = np.argmax(test_x.values * feature_importances, axis=1)
        most_important_feature_names = [test_x.columns[i] for i in important_features]

        # Map cancer types
        cancer_types = []
        for feature, prediction in zip(most_important_feature_names, final_user_pred):
            if prediction == 1:  # Only assign cancer type for positive predictions
                cancer_type = cancer_mapping.get(feature, 'Unknown Cancer')
            else:
                cancer_type = ''
            cancer_types.append(cancer_type)

        user_x['most_important_feature'] = most_important_feature_names
        user_x['cancer_type'] = cancer_types

        return user_x[['Gene_ID', 'predicted_value', 'most_important_feature', 'cancer_type']]

    else:
        raise NotImplementedError(f"The report={report} is not known!")
