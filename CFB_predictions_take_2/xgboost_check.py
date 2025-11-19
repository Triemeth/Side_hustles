#check metrics other than accuracy

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler
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

if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")

    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    xgb = XGBClassifier(
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

    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]

    conf_matrix, fpr, tpr, roc_auc = metrics(y_test, preds, y_pred_proba)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.savefig("../CFB_predictions_take_2/preformance_pics/confusion_matrix_XGB_chcek.jpg")
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
    plt.savefig("../CFB_predictions_take_2/preformance_pics/roc_auc_XGB_chcek.jpg")
    plt.show()
