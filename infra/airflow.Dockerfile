FROM apache/airflow:3.3.2

# Packages the training DAG imports. Kept minimal; versions are bounded, not pinned to a lock.
RUN pip install --no-cache-dir "scikit-learn>=1.4" "pandas>=2" "mlflow==3.17.2"
