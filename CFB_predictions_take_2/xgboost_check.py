#check metrics other than accuracy

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,precision_score, 
                             recall_score, confusion_matrix,
                             f1_score, roc_curve, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)

def metrics(y_test, preds, y_pred_proba):
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    conf_matrix = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {acc}")
    print(f"Recall: {recall}")
    print(f"Precision: {prec}")
    print(f"f1_score: {f1}")
    print(f"Specificity: {specificity}")
    print(f"Roc Auc: {roc_auc}")

    return conf_matrix, fpr, tpr, roc_auc

def plot_ROCAUC_confusion_matrix(conf_matrix, fpr, tpr, roc_auc, num):
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.savefig(f"../CFB_predictions_take_2/preformance_pics/confusion_matrix_XGB_chcek{num}.jpg")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"../CFB_predictions_take_2/preformance_pics/roc_auc_XGB_chcek{num}.jpg")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    drop_cols = ["week", "Year", "team"]
    df = df.drop(columns = drop_cols, axis = 0)

    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)

    #FIRST MODEL    
    scaler = MaxAbsScaler()
    X1 = scaler.fit_transform(X)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X1, y, test_size=0.2, random_state=23, stratify=y
    )

    xgb1 = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,              
        tree_method="hist",
        subsample = .9,
        n_estimators = 600,
        min_child_weight = 1,
        max_depth = 3,
        learning_rate = .05,
        colsample_bytree = .6
    )

    xgb1.fit(
        X_train1, y_train1,
        eval_set=[(X_train1, y_train1), (X_test1, y_test1)],
        verbose=True
    )    
    preds1 = xgb1.predict(X_test1)
    y_pred_proba1 = xgb1.predict_proba(X_test1)[:, 1]

    print("MODEL 1:")
    conf_matrix1, fpr1, tpr1, roc_auc1 = metrics(y_test1, preds1, y_pred_proba1)
    plot_ROCAUC_confusion_matrix(conf_matrix1, fpr1, tpr1, roc_auc1, "1")

    #SECOND MODEL
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X)
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y, test_size=0.2, random_state=23, stratify=y
    )

    xgb2 = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,              
        tree_method="hist",
        subsample = .7,
        n_estimators = 400,
        min_child_weight = 3,
        max_depth = 6,
        learning_rate = .05,
        colsample_bytree = 1.0
    )

    xgb2.fit(X_train2, y_train2)
    preds2 = xgb2.predict(X_test2)
    y_pred_proba2 = xgb2.predict_proba(X_test2)[:, 1]

    print("\nMODEL 2:")
    conf_matrix2, fpr2, tpr2, roc_auc2 = metrics(y_test2, preds2, y_pred_proba2)
    plot_ROCAUC_confusion_matrix(conf_matrix2, fpr2, tpr2, roc_auc2, "2")

    
    # Check for over fitting
    print()
    train_preds = xgb1.predict(X_train1)
    test_preds = xgb1.predict(X_test1)

    print("Train accuracy:", accuracy_score(y_train1, train_preds))
    print("Test accuracy:", accuracy_score(y_test1, test_preds))

    results = xgb1.evals_result()

    train_logloss = results['validation_0']['logloss']
    test_logloss  = results['validation_1']['logloss']

    plt.plot(train_logloss, label='train')
    plt.plot(test_logloss, label='test')
    plt.legend()
    plt.title("XGBoost Learning Curve")
    plt.show()

    
