import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow


# Ini akan membuat folder 'mlruns' otomatis di laptopmu
mlflow.set_experiment("Superstore_Basic_Autolog")

# 2. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "superstore_preprocessing.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. KUNCI BASIC: Wajib Autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):
    # 4. KUNCI BASIC: Tanpa Tuning (Kosongkan parameter)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Catat script ini sendiri biar muncul di Artifacts MLflow Lokal
    mlflow.log_artifact(__file__)

    print("✅ Model Basic berhasil dilatih dan dicatat secara LOKAL di folder mlruns!")