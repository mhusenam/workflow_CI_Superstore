import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import dagshub

# 1. Inisialisasi DagsHub untuk MLflow Tracking
# (Ubah 'USERNAME_KAMU' dan 'REPOSTORI_KAMU' sesuai akun DagsHub nanti)
dagshub.init(repo_owner='mhusenam', repo_name='Ekperimen_MuhammadHuseinAbdullahMahfud', mlflow=True)

# 2. Load Data Preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "superstore_preprocessing.csv")

print(f"📂 Mencari data di jalur otomatis: {data_path}")
df = pd.read_csv(data_path)

# Pisahkan Fitur dan Target
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MLflow Experiment
mlflow.set_experiment("Superstore_Base_Modelling")

with mlflow.start_run(run_name="RandomForest_Base"):
    # Set Hyperparameters Dasar
    n_estimators = 100
    max_depth = 5
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Training Model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi & Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    # Log Metrics ke MLflow/DagsHub
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
    # Log Model sebagai Artifak
    mlflow.sklearn.log_model(model, "base_random_forest_model")
    
    import shutil
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model") # Hapus yang lama jika ada
    
    mlflow.sklearn.save_model(model, "saved_model")
    
    print(f"🎉 Model berhasil disimpan ke folder 'saved_model'!")