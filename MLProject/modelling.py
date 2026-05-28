import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "superstore_preprocessing.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Autolog
mlflow.sklearn.autolog()

# 3. Langsung eksekusi (Otomatis ikut aturan mlflow run GitHub)
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    mlflow.log_artifact(__file__)