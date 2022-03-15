import os

import pandas as pd
import pymongo
import streamlit as st

from ml.model import get_metrics_params, load_model


@st.cache
def get_data(nrows: int):
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";", nrows=nrows)
    return data


def get_usage():
    client = pymongo.MongoClient("mongodb://mongo:27017")
    db = client["models"]
    model_collection = db["example-model"]
    return list(model_collection.find({}))


@st.cache
def get_stats(model_id: str):
    if model_id:
        return get_metrics_params(model_id)
    else:
        return None


@st.cache
def get_model(model_id: str):
    if model_id:
        return load_model(model_id)
    else:
        return None


def min_max_avg_col(df: pd.DataFrame, feature: str):
    min_val = df[feature].min()
    max_val = df[feature].max()
    avg_val = (min_val + max_val) / 2
    return float(min_val), float(max_val), float(avg_val)


model_id = os.environ.get("MODEL_ID")

st.title("Performance Dashboard")
st.write(f"Run ID: {model_id}")

data = get_data(20)
usage = get_usage()
stats = get_stats(model_id)
model = get_model(model_id)

st.subheader("Usage")
st.write(usage)

st.subheader("Sample Data")
st.write(data)

st.subheader("Performance")
col_metrics, col_params = st.columns(2)
with col_metrics:
    st.subheader("Metrics")
    for i, e in stats["metrics"].items():
        st.metric(i, e)
with col_params:
    st.subheader("Parameters")
    for i, e in stats["parameters"].items():
        st.metric(i, e)


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
