import mlflow
from pyparsing import makeXMLTags
from sklearn.linear_model import ElasticNet


class Model:
    def __init__(self, alpha=0.5, l1_ratio=0.5, state=1) -> None:
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=state)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        predicted = self.model.predict(test_x)
        return predicted


def load_model(run_id: str):
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model


def get_metrics_params(run_id: str):
    model_run = mlflow.get_run(run_id)
    metrics = model_run.data.metrics
    params = model_run.data.params
    return {"metrics": metrics, "parameters": params}
