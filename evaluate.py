from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate_model(model, train_input, train_target, test_input, test_target):
    train_pred = model.predict(train_input)
    test_pred = model.predict(test_input)

    mae_train = mean_absolute_error(train_target, train_pred)
    mae_test = mean_absolute_error(test_target, test_pred)

    rmse_train = root_mean_squared_error(train_target, train_pred)
    rmse_test = root_mean_squared_error(test_target, test_pred)

    print("Train MAE:", mean_absolute_error(train_target, train_pred), 
        "Test MAE", mean_absolute_error(test_target, test_pred))
    print("Train RMSE:", root_mean_squared_error(train_target, train_pred),
        "Test RMSE:", root_mean_squared_error(test_target, test_pred))
