import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cancer_patient_data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Smoking",
    "Air Pollution",
    "Alcohol use",
    "Obesity",
    "Genetic Risk",
]
TARGET = "Level"

# False reproduce la corrida original donde Extra Trees obtuvo 95%.
# True aplica la recomendación metodológica de partición estratificada.
USE_STRATIFIED_SPLIT = False


def train_model():
    df = pd.read_csv(DATA_PATH)

    # Identificadores administrativos: no aportan valor predictivo.
    df = df.drop(columns=["index", "Patient Id"], errors="ignore")

    X = df[FEATURES]
    y = df[TARGET]

    stratify_value = y if USE_STRATIFIED_SPLIT else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    metrics = {
        "model": "ExtraTreesClassifier",
        "features": FEATURES,
        "target": TARGET,
        "classes": labels,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
        "class_distribution": y.value_counts().to_dict(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split": "80/20 stratified" if USE_STRATIFIED_SPLIT else "80/20 random split, random_state=42, no stratify",
        "methodology_note": "If USE_STRATIFIED_SPLIT=True, metrics may change and should be reported again.",
        "scaler": "StandardScaler",
        "hyperparameters": {
            "n_estimators": 50,
            "max_depth": 4,
            "random_state": 42,
        },
    }

    joblib.dump(model, ARTIFACTS_DIR / "extra_trees_lung_risk_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(FEATURES, ARTIFACTS_DIR / "features.pkl")

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Modelo entrenado correctamente. Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred, labels=labels))


if __name__ == "__main__":
    train_model()
