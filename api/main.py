import datetime
from typing import List
import os

import pandas as pd
from fastapi import FastAPI
import mlflow
import numpy as np
import pymongo

from ml.data import get_data
from ml.evaluate import eval_metrics
from ml.model import Model, get_metrics_params, load_model
from ml.predict import predict
from ml.training import train

app = FastAPI()


def response_to_mongo(r: dict):
    client = pymongo.MongoClient("mongodb://mongo:27017")
    db = client["models"]
    model_collection = db["example-model"]
    model_collection.insert_one(r)
    # Remove ObjectId '_id' field added by insert_one()
    r.pop("_id", None)


@app.on_event("startup")
def load_api_model():
    model_id = os.environ.get("MODEL_ID")
    if model_id:
        global model
        model = load_model(model_id)


@app.get("/")
async def root():
    response = {"message": "Hello World"}
    return response


@app.post("/predict")
async def predict_model(features: List[float]):
    model_id = os.environ.get("MODEL_ID")

    prediction = model.predict(
        pd.DataFrame(
            [features],
            columns=model.feature_names_in_,
        )
    )

    response = {
        "model_id": model_id,
        "features": features,
        "predictions": prediction.tolist(),
        "date": datetime.datetime.utcnow(),
    }
    response_to_mongo(response)
    return response


@app.post("/train")
async def train_model(
    alpha: float = 0.5,
    l1_ratio: float = 0.5,
    experiment_name="Wine Tasting",
    run_name="Linear Regression",
):
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        np.random.seed(1)
        lr = Model(alpha, l1_ratio)

        # data
        train_x, test_x, train_y, test_y = get_data()

        # training
        train(lr, train_x, train_y)

        # predict
        predicted = predict(lr, test_x)

        # evaluate
        rmse, mae, r2 = eval_metrics(test_y, predicted)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr.model, "model")

        response = {"message": "Model trained", "run_id": run_id}
        return response
