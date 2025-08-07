import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import joblib

print("Training model...")


housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y = pd.Series(housing.target)


model = LinearRegression()
model.fit(X, y)
print("Model training complete.")


joblib.dump(model, 'house_price_model.pkl')
print("Model saved as house_price_model.pkl")