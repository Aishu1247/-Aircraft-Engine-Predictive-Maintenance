import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Aircraft Engine Predictive Maintenance")

st.write(
    """
    AI-powered predictive maintenance system
    built using NASA CMAPSS engine data.
    """
)

# -----------------------------
# Upload Section
# -----------------------------

st.header(
    "📂 Upload Engine Data"
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------
# Load Dataset
# -----------------------------

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

# -----------------------------
# Create RUL
# -----------------------------

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

# -----------------------------
# Features
# -----------------------------

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

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Load Saved Model
# -----------------------------
model = joblib.load(
    "best_engine_model.pkl"
)


predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

# -----------------------------
# Latest Engine Prediction
# -----------------------------

latest_engine = df[
    df['engine_id'] == 50
].iloc[0]


latest_features = latest_engine[
    features
]

predicted_rul = model.predict(
    latest_features.to_frame().T
)[0]



# -----------------------------
# Feature Importance
# -----------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
feature_importance['Importance (%)'] = (
    feature_importance['Importance'] * 100
).round(2)

# -----------------------------
# Dashboard
# -----------------------------

st.success(
    f"""
    Best Model: Extra Trees Regressor

    Model Accuracy (MAE): {round(mae,2)}

    Dataset: NASA CMAPSS FD001
    """
)
model_results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting"
    ],
    "MAE": [
        29.69,
        29.46,
        29.88
    ]
})

st.header(
    "Model Comparison"
)

st.dataframe(
    model_results
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        len(df)
    )

with col2:
    st.metric(
        "Engines",
        df['engine_id'].nunique()
    )

with st.container():

    st.metric(
        "Most Important Sensor",
        feature_importance.iloc[0]["Feature"]
    )

# -----------------------------
# Engine Health Analysis
# -----------------------------

st.header(
    "Engine Health Analysis"
)

col4, col5 = st.columns(2)

with col4:
    st.metric(
        "Predicted RUL",
        round(predicted_rul, 1)
    )

health_score = min(
    100,
    max(
        0,
        predicted_rul / 2
    )
)

with col5:
    st.metric(
        "Health Score",
        round(health_score, 1)
    )
if health_score > 70:

    st.success(
        "🟢 Healthy Engine"
    )

    st.info(
        "Recommendation: Continue normal operation."
    )

elif health_score > 30:

    st.warning(
        "🟡 Maintenance Recommended"
    )

    st.info(
        "Recommendation: Schedule inspection soon."
    )

else:

    st.error(
        "🔴 Critical Condition"
    )

    st.info(
        "Recommendation: Immediate maintenance required."
    )
top_features = (
    feature_importance
    .head(5)
)

st.success(
    f"""
    Main Contributors:

    1. {top_features.iloc[0]['Feature']}
    2. {top_features.iloc[1]['Feature']}
    3. {top_features.iloc[2]['Feature']}
    """
)



# -----------------------------
# Top Sensors
# -----------------------------

st.header(
    "Top Important Sensors"
)


st.dataframe(
    feature_importance[
        ['Feature', 'Importance (%)']
    ].head(10)
)

# -----------------------------
# Feature Importance Graph
# -----------------------------

st.header(
    "Feature Importance Graph"
)
# -----------------------------
# Sensor Trend Analysis
# -----------------------------

st.header(
    "Sensor Trend Analysis"
)

selected_sensor = st.selectbox(
    "Choose Sensor",
    [f"sensor_{i}" for i in range(1, 22)]
)

engine1 = df[
    df['engine_id'] == 1
]

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.plot(
    engine1['cycle'],
    engine1[selected_sensor]
)

ax2.set_xlabel(
    "Cycle"
)

ax2.set_ylabel(
    selected_sensor
)

ax2.set_title(
    f"{selected_sensor} vs Engine Life"
)

st.pyplot(fig2)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    feature_importance['Feature'][:10],
    feature_importance['Importance'][:10]
)

plt.xticks(
    rotation=45
)

st.pyplot(fig)



# -----------------------------
# Upload Prediction Section
# -----------------------------

if uploaded_file is not None:

    uploaded_df = pd.read_csv(
        uploaded_file
    )


    prediction = model.predict(
        uploaded_df[features]
    )

    predicted_rul_upload = prediction[0]
    upload_health_score = min(
    100,
    max(
        0,
        predicted_rul_upload / 2
    )
)
    

    st.metric(
        "Predicted RUL (Uploaded Engine)",
        round(
            predicted_rul_upload,
            1
        )
    )
    st.metric(
    "Uploaded Engine Health Score",
    round(
        upload_health_score,
        1
    )
)
    st.progress(
    upload_health_score / 100
)
    