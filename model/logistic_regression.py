from sklearn.linear_model import LogisticRegression


def create_model():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    return model
