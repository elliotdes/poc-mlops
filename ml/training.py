from ml.model import Model


def train(model: Model, train_x, train_y):
    model.fit(train_x, train_y)
