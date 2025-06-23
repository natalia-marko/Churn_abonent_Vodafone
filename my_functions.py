import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

def save_model(model=None, features=None, thr=None):
    if features is None:
        features = []
    name = str(model.__class__).split('.')[-1][:-2] + '_' + datetime.today().strftime("%d%m%Y_%H_%M") + '.pickle'
    if model:
        with open(name, 'wb') as file:
            pickle.dump((model, features, thr), file)
            print('Save', name)


def save_table(table, file_path, table_name):
    name = file_path + '\\' + table_name + '.pickle'
    with open(name, 'wb') as file:
        pickle.dump(table, file)
        print('Save', name)


def load_data(file_name):
    with open(file_name, 'rb') as file:
        model_tpl = pickle.load(file)
    return model_tpl


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

from datetime import datetime
def print_current_datetime():
    now = datetime.now()
    print("Date and time calculation: ", now.strftime("%Y-%m-%d %H:%M:%S"))


def get_features(model, X, target_variable, top_n=20):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    top_features = feature_importance_df[:top_n]
    
    features_with_importance_gt_zero = feature_importance_df.loc[feature_importance_df['Importance'] > 0.0, 'Feature'].tolist()

    print("Feature Importance:")
    print(feature_importance_df)
    print("\nTop", top_n, "Features:")
    print(top_features)
    print("\nNumber of Features with Importance > 0.0:", len(features_with_importance_gt_zero))
    
    return top_features['Feature'].tolist(), features_with_importance_gt_zero

import numpy as np
import pandas as pd

def drop_highly_correlated(df, threshold=0.98):
    df_corr = df.corr(method='spearman')
    
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    
    cols_to_drops = []
    for col in df_corr.columns.values:
        if np.in1d([col],cols_to_drops):
            continue
        corr = df_corr[abs(df_corr[col]) > threshold].index
        cols_to_drops = np.union1d(cols_to_drops, corr)

    cols_before_drop = df.columns
    df_new_corr = df.drop(columns=cols_to_drops)
    cols_after_drop = df_new_corr.columns

    dropped_columns = list(set(cols_before_drop) - set(cols_after_drop))
    print("Droped columns:", len(dropped_columns),'nr', dropped_columns)
    
    return df_new_corr, dropped_columns


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def calculate_optimal_threshold(y_true, y_pred_proba, print_results=True):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f1_scores = 2*recall*precision / (recall + precision)
    num_thresholds=100
    new_thresholds = np.linspace(start=min(thresholds), stop=max(thresholds), num=num_thresholds)

    new_f1_scores = np.interp(new_thresholds, thresholds, f1_scores[:-1])

    pr_df = pd.DataFrame({
        'F1_Score': new_f1_scores,
        'Thresholds': new_thresholds
    })

    idx = pr_df['F1_Score'].idxmax()

    optimal_threshold = pr_df.loc[idx, 'Thresholds']
    max_f1 = pr_df.loc[idx, 'F1_Score']

    if print_results:
        print("Optimal threshold value:", round(optimal_threshold, 2))
        print("Maximum F1 Score:", round(max_f1, 2))
        # plot
        plt.figure(figsize=(3, 2))
        plt.plot(pr_df['Thresholds'], pr_df['F1_Score'], label='F1 Score', color='blue')
        plt.axvline(x=optimal_threshold, color='gray', linestyle='--')
        plt.title(f'F1 Score vs Threshold, model CatBoost Classifier\n Optimal threshold: {optimal_threshold:.2f}\n Max F1: {max_f1:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.show()
        now = datetime.now()
        print("Date and time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
        
    return optimal_threshold, max_f1

def plot_learning_curve(model, X_train, y_train, X_valid, y_valid):
    train_auc = []
    valid_auc = []
    n_rounds = model.get_params().get('n_estimators', 200)
    
    for y_pred_train, y_pred_valid in zip(model.staged_predict_proba(X_train), model.staged_predict_proba(X_valid)):
        train_auc.append(roc_auc_score(y_train, y_pred_train[:, 1]))
        valid_auc.append(roc_auc_score(y_valid, y_pred_valid[:, 1]))
    
    plt.figure(figsize=(8,4))
    plt.plot(train_auc, label='Train AUC')
    plt.plot(valid_auc, label='Validation AUC')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('AUC')
    plt.title(f'Learning Curve for Catboost')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_probabilities_hist(train_proba_1, test_proba_1, train_proba_0, test_proba_0):
    plt.figure(figsize=(20,15))

    # histogram for train (class 1)
    plt.subplot(4, 1, 1)
    plt.hist(train_proba_1, bins=30, alpha=0.5, color='blue')
    plt.title('Class 1 (Train)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    # # histogram for test (class 1)
    plt.subplot(4, 1, 2)
    plt.hist(test_proba_1, bins=30, alpha=0.5, color='purple')
    plt.title('Class 1 (Test)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    # # histogram for train (class 0)
    plt.subplot(4, 1, 3)
    plt.hist(train_proba_0, bins=30, alpha=0.5, color='red')
    plt.title('Class 0 (Train)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    # # histogram for test (class 0)
    plt.subplot(4, 1, 4)
    plt.hist(test_proba_0, bins=30, alpha=0.5, color='green')
    plt.title('Class 0 (Test)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score

def plot_roc_pr_curves(y_true, y_scores, title_prefix=''):
    """
    Plots ROC and Precision-Recall curves side by side.
    Parameters:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted probabilities for the positive class.
        title_prefix (str): Optional prefix for plot titles.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC plot
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{title_prefix} ROC Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # PR plot
    axes[1].plot(recall, precision, label=f'AP = {ap:.2f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{title_prefix} Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_roc_auc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plots the ROC AUC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


import time
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier


def plot_cb_model_results(results):

    # Figure out how many epochs there were
    epochs = range(len(results['train_auc']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot AUC
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_auc'], label='train_auc')
    plt.plot(epochs, results['test_auc'], label='test_auc')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_F1'], label='train_F1')
    plt.plot(epochs, results['test_F1'], label='test_F1')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.legend();

import seaborn as sns
def plotting_confusion_matrix(y_true, binary_predictions):
    # Confusion matrix analysis
    print("Generating confusion matrix...")

    cm = confusion_matrix(y_true, binary_predictions)
    cm_normalized = confusion_matrix(y_true, binary_predictions, normalize='true')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Absolute Values)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Retained', 'Churned'])
    ax1.set_yticklabels(['Retained', 'Churned'])

    # # Normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_xticklabels(['Retained', 'Churned'])
    ax2.set_yticklabels(['Retained', 'Churned'])

    plt.tight_layout()
    plt.show()

    # # Business interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nBusiness Impact Analysis:\n")
    print(f"True Negatives (Correctly identified retained): {tn:,}")
    print(f"False Positives (Incorrectly flagged as churn): {fp:,}")
    print(f"False Negatives (Missed churners): {fn:,}")
    print(f"True Positives (Correctly identified churn): {tp:,}")

    print("Confusion matrix analysis completed!")

from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import pandas as pd

def evaluate_classification_model(y_true, y_pred_binary, y_pred_proba, model_name='model'):

    auc_score = roc_auc_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return pd.DataFrame([{
        "Model": model_name,
        "AUC-ROC": auc_score,
        "Recall": recall,
        "F1": f1,
        "Precision": precision
    }])

