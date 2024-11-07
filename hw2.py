import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data_path = 'I:\\IOT\\HW2\\For Saana\\data.csv'   #data file path
data = pd.read_csv(data_path)


X = data[['DC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]
y = data['AC_POWER']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor()
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R^2 Score": r2
    }



plt.figure(figsize=(14, 6))


for i, (name, model) in enumerate(models.items(), start=1):
    y_pred = model.predict(X_test)

    plt.subplot(1, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # خط ایده‌آل
    plt.xlabel("Actual AC Power")
    plt.ylabel("Predicted AC Power")
    plt.title(f"{name} - Actual vs Predicted")

plt.tight_layout()
plt.show()


metrics_names = ["MAE", "MSE", "R^2 Score"]
metrics_values = {metric: [results[model][metric] for model in results] for metric in metrics_names}

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics_names):
    plt.bar([name for name in results.keys()], metrics_values[metric], alpha=0.6, label=metric)

plt.xlabel("Models")
plt.ylabel("Metrics")
plt.title("Model Evaluation Metrics")
plt.legend()
plt.show()
