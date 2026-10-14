import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib

from src.config import (
    DATA_DIR, REPORTS_DIR, CLASS_LABEL, ID_COLUMNS,
    BASE_FEATURES, RANDOM_STATE, TEST_SIZE, N_JOBS
)
from src.data.datasets import load_cicids2017_csvs, prepare_dataset
from src.features.feature_engineering import add_engineered_features


def plot_roc(y_true, y_proba, outpath):
    """Save ROC curve to file."""
    plt.figure()
    skplt.metrics.plot_roc(y_true, y_proba)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main(args):
 
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    reports_dir = Path(args.reports_dir) if args.reports_dir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

 
    df = load_cicids2018_csvs(data_dir)
    X, y, _ = prepare_dataset(df, CLASS_LABEL, ID_COLUMNS)

    # Feature selection
    cols = [c for c in BASE_FEATURES if c in X.columns]
    X = X[cols] if cols else X
    X = add_engineered_features(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing
    num_cols = X.columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ])

    # Build pipeline steps
    steps = [("pre", pre)]
    if args.smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    if args.use_rfe:
        steps.append(("rfe", RFE(
            estimator=RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
            n_features_to_select=max(10, int(0.5 * len(num_cols)))
        )))
    if args.use_pca:
        steps.append(("pca", PCA(n_components=min(20, max(2, int(0.3 * len(num_cols)))))))

    # Random Forest classifier
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

    # Final pipeline
    pipe = ImbPipeline(steps + [("clf", rf)])

    print("Pipeline steps:", [name for name, _ in pipe.steps])  # ✅ show steps

    # Hyperparameter grid
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 12, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }

    # Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # GridSearch
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv,
                      n_jobs=N_JOBS, scoring="f1_weighted", verbose=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    # Predictions
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred).tolist()

    # Improved AUC handling
    auc_weighted = None
    try:
        classes = np.array(best.classes_)
        if len(classes) == 2:
            pos_idx = np.where(classes == classes[1])[0][0]
            y_true_bin = (y_test == classes[1]).astype(int)
            auc_weighted = roc_auc_score(y_true_bin, y_proba[:, pos_idx])
        else:
            Y_true = label_binarize(y_test, classes=classes)
            auc_weighted = roc_auc_score(Y_true, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        auc_weighted = None

    metrics = {
        "best_params": gs.best_params_,
        "classification_report": report,
        "confusion_matrix": conf,
        "auc_weighted": auc_weighted,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve
    plot_roc(y_test, y_proba, reports_dir / "roc_ovr.png")

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, models_dir / "rf_classifier.joblib", compress=3)

    # ✅ Pretty print classification report
    from tabulate import tabulate
    rows = []
    for cls, vals in report.items():
        if isinstance(vals, dict) and "precision" in vals:
            rows.append([cls, vals["precision"], vals["recall"], vals["f1-score"], vals["support"]])
    print("\nClassification Report:")
    print(tabulate(rows, headers=["Class", "Precision", "Recall", "F1", "Support"], floatfmt=".4f"))
    print("\nAccuracy:", report.get("accuracy"))
    print("AUC (weighted):", auc_weighted)

    print("\nTraining complete. Reports saved to", reports_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--reports_dir", type=str, default=None)
    ap.add_argument("--use_rfe", action="store_true")
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling on training data")
    args = ap.parse_args()
    main(args)
