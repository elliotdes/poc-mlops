from operator import mod

import pandas as pd
import streamlit as st

from ml.model import get_metrics_params, load_model


@st.cache
def get_data(nrows: int):
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";", nrows=nrows)
    return data


@st.cache
def get_stats():
    return get_metrics_params("eed0f563443743bf8649bfa71dc89c22")


@st.cache
def get_model():
    return load_model("runs:/eed0f563443743bf8649bfa71dc89c22/model")


def min_max_avg_col(df: pd.DataFrame, feature: str):
    min_val = df[feature].min()
    max_val = df[feature].max()
    avg_val = (min_val + max_val) / 2
    return float(min_val), float(max_val), float(avg_val)


st.title("Performance Dashboard")

data = get_data(20)
stats = get_stats()
model = get_model()

st.subheader("Sample data")
st.write(data)

st.subheader("Performance")
st.write(stats)

st.subheader("Prediction")
col1, col2 = st.columns(2)
features = []
with col1:
    for feature in model.feature_names_in_:
        number = st.slider(feature, *min_max_avg_col(data, feature))
        features.append(number)
with col2:

    prediction = model.predict(
        pd.DataFrame(
            [features],
            columns=model.feature_names_in_,
        )
    )
    st.metric("Prediction", prediction)
