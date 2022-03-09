from typing import List

import mlflow
import numpy as np
import pandas as pd
import typer

from ml.data import get_data
from ml.evaluate import eval_metrics
from ml.model import Model, get_metrics_params, load_model
from ml.predict import predict
from ml.training import train

app = typer.Typer()


@app.command()
def train_model(
    alpha: float = 0.5,
    l1_ratio: float = 0.5,
    experiment_name="Wine Tasting",
    run_name="Linear Regression",
):
    """Train a new model.

    Args:
        alpha (float, optional): Defaults to 0.5.
        l1_ratio (float, optional): Defaults to 0.5.
        experiment_name (str, optional): Defaults to "Wine Tasting".
        run_name (str, optional): Defaults to "Linear Regression".
    """
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
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


@app.command()
def predict(model_uri: str, features: List[float]):
    """Use given model to predict the features.

    python cli/main.py predict "runs:/eed0f563443743bf8649bfa71dc89c22/model"
    7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56 9.4

    Args:
        model_uri (str): Model to load.
        features (List[float]): Features to predict.
    """
    loaded_model = load_model(model_uri)

    prediction = loaded_model.predict(
        pd.DataFrame(
            [features],
            columns=loaded_model.feature_names_in_,
        )
    )

    print(prediction)


@app.command()
def stats(run_id: str):
    print(get_metrics_params(run_id))


if __name__ == "__main__":
    app()
