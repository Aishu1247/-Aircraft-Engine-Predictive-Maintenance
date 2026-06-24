import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

columns = [
    'engine_id',
    'cycle',
    'op_setting_1',
    'op_setting_2',
    'op_setting_3'
]

for i in range(1, 22):
    columns.append(f'sensor_{i}')

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)

df = df.iloc[:, :26]

df.columns = columns

max_cycles = df.groupby(
    'engine_id'
)['cycle'].max()

df['max_cycle'] = df[
    'engine_id'
].map(max_cycles)

df['RUL'] = (
    df['max_cycle']
    -
    df['cycle']
)

features = [
    'op_setting_1',
    'op_setting_2',
    'op_setting_3'
]

for i in range(1, 22):
    features.append(
        f'sensor_{i}'
    )

X = df[features]

y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "engine_model.pkl"
)

joblib.dump(
    features,
    "features.pkl"
)

print(
    "Model Saved Successfully!"
)
