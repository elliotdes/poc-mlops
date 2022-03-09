import logging
from typing import List

import pandas as pd
from fastapi import FastAPI

from ml.model import load_model

app = FastAPI()


@app.on_event("startup")
def load_api_model():
    global model
    model = load_model("runs:/eed0f563443743bf8649bfa71dc89c22/model")


@app.get("/")
async def root():
    response = {"message": "Hello World"}
    return response


@app.post("/predict")
async def predict(features: List[float]):

    logging.info(type(features))
    prediction = model.predict(
        pd.DataFrame(
            [features],
            columns=model.feature_names_in_,
        )
    )

    response = {"predictions": prediction.tolist()}
    return response
