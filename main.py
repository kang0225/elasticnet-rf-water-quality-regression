import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess
from models import train_elasticnet, train_random_forest
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

train_set, pH = load_and_preprocess('data/water_quality.csv')

train_input, test_input, train_target, test_target = train_test_split(
    train_set, pH, shuffle=False, random_state=42    
)

ss = StandardScaler()
train_scaled = ss.fit_transform(train_input)
test_scaled = ss.transform(test_input)

poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train_scaled)
test_poly = poly.transform(test_scaled)

# 엘라스틱넷과 랜덤 포레스트 모델 생성 및 훈련
elastic_model = train_elasticnet(train_poly, train_target)
rf_model = train_random_forest(train_poly, train_target)

# 성능 결과
print("\nElasticNet 성능:")
evaluate_model(elastic_model, train_poly, train_target, test_poly, test_target)

print("\nRandom Forest 성능:")
evaluate_model(rf_model, train_poly, train_target, test_poly, test_target)