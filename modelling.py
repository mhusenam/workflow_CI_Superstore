import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import dagshub

# 1. Setup MLflow & DagsHub
mlflow.set_tracking_uri("https://dagshub.com/mhusenam/Ekperimen_MuhammadHuseinAbdullahMahfud.mlflow")
if "MLFLOW_TRACKING_PASSWORD" in os.environ:
    os.environ["DAGSHUB_CLIENT_TOKEN"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

# 2. Load Data Preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "superstore_preprocessing.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Buat Eksperimen Baru (Biar Bersih dari Cache Eror Lama)
mlflow.set_experiment("Superstore_CI_CD_Final")

with mlflow.start_run(run_name="RandomForest_Final") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    # KUNCI 1: Log model dengan nama wajib "model"
    mlflow.sklearn.log_model(model, "model")
    
    # KUNCI 2: Tulis RUN_ID ke file teks agar gampang dibaca GitHub
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
        
    print(f"🎉 Sukses! RUN_ID: {run.info.run_id} berhasil dicetak ke run_id.txt")