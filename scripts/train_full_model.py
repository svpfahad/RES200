"""Train the full-descriptor XGBoost model on Opt2 acidic data and save predictions."""
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\Fahad\Downloads\RES200\RES 200-20260312T035531Z-1-001\RES 200")
OUT = Path(r"C:\Users\Fahad\Downloads\RES200")

target = 'pKa'
train_data = pd.read_csv(ROOT / 'train_descriptors_op2.csv')
test_data = pd.read_csv(ROOT / 'test_descriptors_op2.csv')

# Sanitize as in the notebook
def sanitize(df):
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, df.notna().all()]
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True)

train_data = sanitize(train_data)
test_data  = sanitize(test_data)

X_train = train_data.drop(columns=[target])
X_test  = test_data.drop(columns=[target])

# Align columns
missing = set(X_train.columns) - set(X_test.columns)
for col in missing:
    X_test[col] = 0
extra = set(X_test.columns) - set(X_train.columns)
if extra:
    X_test = X_test.drop(columns=list(extra))
X_test = X_test[X_train.columns]

y_train = train_data[target]
y_test  = test_data[target]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

model = XGBRegressor(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    reg_lambda=1,
    subsample=0.8,
    objective='reg:squarederror',
    random_state=42,
)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

print(f"Train R2 = {r2_score(y_train, train_pred):.4f}, RMSE = {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
print(f"Test  R2 = {r2_score(y_test,  test_pred ):.4f}, RMSE = {np.sqrt(mean_squared_error(y_test,  test_pred )):.4f}, MAE = {mean_absolute_error(y_test, test_pred):.4f}")

pd.DataFrame({'y_true': y_train.values, 'y_pred': train_pred}).to_csv(OUT / 'predictions_train.csv', index=False)
pd.DataFrame({'y_true': y_test.values,  'y_pred': test_pred }).to_csv(OUT / 'predictions_test.csv',  index=False)

# Save feature importances
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.head(25).to_csv(OUT / 'feature_importances_top25.csv', header=['importance'])
print("Saved predictions_train.csv, predictions_test.csv, feature_importances_top25.csv")
