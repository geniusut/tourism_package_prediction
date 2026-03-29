
# =========================================
# 1. Imports
# =========================================
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# =========================================
# 2. MLflow Setup 
# =========================================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")


# =========================================
# 3. Load Data from Hugging Face 
# =========================================
repo_id = "geniusut/tourism-package-prediction-project"

train_path = f"hf://datasets/{repo_id}/train.csv"
test_path = f"hf://datasets/{repo_id}/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train/Test loaded from Hugging Face!")


# =========================================
# 4. Split X and y 
# =========================================
target_col = "ProdTaken"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]


# =========================================
# 5. Model + Hyperparameter Grid
# =========================================
rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}


# =========================================
# 6. MLflow Experiment
# =========================================
with mlflow.start_run():

    # Grid Search
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Nested runs (like example)
    results = grid_search.cv_results_

    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    train_preds = best_model.predict(X_train)

    # =========================================
    # 7. Evaluation
    # =========================================
    train_report = classification_report(y_train, train_preds, output_dict=True)
    test_report = classification_report(y_test, y_pred, output_dict=True)

    class_key = '1' if '1' in train_report else 1

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report[class_key]['precision'],
        "train_recall": train_report[class_key]['recall'],
        "train_f1_score": train_report[class_key]['f1-score'],

        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report[class_key]['precision'],
        "test_recall": test_report[class_key]['recall'],
        "test_f1_score": test_report[class_key]['f1-score']
    })

    mlflow.log_params(grid_search.best_params_)


    # =========================================
    # 8. Save Model
    # =========================================
    os.makedirs("model_building/artifacts", exist_ok=True)
    model_path = "model_building/artifacts/model.pkl"

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print("Model saved!")


    # =========================================
    # 9. Upload Model to Hugging Face
    # =========================================
    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Replace with your actual Hugging Face dataset repo
    model_repo_id = "geniusut/tourism-package-prediction-project"
    model_repo_type = "model"


    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pkl",
        repo_id=model_repo_id,
        repo_type=model_repo_type
    )

    print("Model uploaded to Hugging Face!")


    print("Best Parameters:", grid_search.best_params_)
    print("Test Accuracy:", test_report['accuracy'])
