from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

def train_elasticnet(input, target):
    model = ElasticNetCV(alphas=[0.01, 0.1, 1, 10], l1_ratio=[0.2, 0.5, 0.8], cv=5)
    model.fit(input, target)
    return model

def train_random_forest(input, target):
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        min_samples_leaf=5,
        random_state=42)
    model.fit(input, target)
    return model
