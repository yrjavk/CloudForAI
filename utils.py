import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(y,y_pred):
  mse=mean_squared_error(y, y_pred)
  return np.sqrt(mse)

def calculate_relative_error(rmse, range):
    relative_error = float((rmse/range.item()) * 100)

    return relative_error
