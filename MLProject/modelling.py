import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# ---------- 1. Setup MLflow ----------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("modelling-basic")

# aktifkan autolog (syarat Basic)
mlflow.sklearn.autolog()

# ---------- 2. Load data ter-preprocess ----------

df = pd.read_csv("train_df.csv")
X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 3. Training (tanpa tuning) ----------
with mlflow.start_run(run_name="rf_baseline"):
    model = RandomForestClassifier(
        n_estimators=100,  
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("accuracy", acc)


print("Selesai training baseline, cek MLflow UI untuk melihat run.")
