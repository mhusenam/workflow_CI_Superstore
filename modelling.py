import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub
import shutil

# 1. Setup MLflow & DagsHub
mlflow.set_tracking_uri("https://dagshub.com/mhusenam/Ekperimen_MuhammadHuseinAbdullahMahfud.mlflow")
if "MLFLOW_TRACKING_PASSWORD" in os.environ:
    os.environ["DAGSHUB_CLIENT_TOKEN"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

# 2. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "superstore_preprocessing.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training & Logging
mlflow.set_experiment("Superstore_CI_CD_Final")

with mlflow.start_run(run_name="RandomForest_Final"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    # Tetap log ke DagsHub untuk syarat Dicoding
    mlflow.sklearn.log_model(model, "model")
    
    # KUNCI JALAN PINTAS: Simpan juga modelnya SECARA LOKAL di folder 'local_model'
    local_model_path = "local_model"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    mlflow.sklearn.save_model(model, local_model_path)
    
    print(f"🎉 Sukses! Model tersimpan di folder lokal '{local_model_path}'")