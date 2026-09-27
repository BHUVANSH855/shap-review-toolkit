import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

X = pd.DataFrame({"a": pd.Series([1, 2, pd.NA], dtype="Int64"), "b": [0.1, 0.2, 0.3]})
model = RandomForestRegressor(random_state=0).fit(X.fillna(0), [1, 2, 3])
print(shap.TreeExplainer(model, data=X))
