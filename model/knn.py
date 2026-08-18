from sklearn.neighbors import KNeighborsClassifier


def create_model():
    model = KNeighborsClassifier(
        n_neighbors=5
    )
    return model