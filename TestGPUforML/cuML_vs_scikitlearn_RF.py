import time
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from cuml.ensemble import RandomForestRegressor as rfcu  # type: ignore
import loguru
import matplotlib.pyplot as plt

# 儲存結果的函式
def save_results_to_csv(results, filename="results.csv"):
    df = pd.DataFrame(results, columns=["features_count", "model", "rmse", "r2", "training_time"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# 繪圖的函式
def plot_results(results, test_samples_arr):
    df = pd.DataFrame(results, columns=["features_count", "model", "rmse", "r2", "training_time"])

    models = df['model'].unique()

    for metric in ['rmse', 'r2', 'training_time']:
        plt.figure(figsize=(10, 6))
        for model in models:
            model_data = df[df['model'] == model]
            plt.plot(model_data['features_count'], model_data[metric], label=model)

        plt.title(f"{metric.upper()} vs Features Count")
        plt.xlabel('features Count')
        plt.ylabel(metric.upper())
        plt.legend(title="Model", loc="upper right")
        plt.grid(True)
        #plt.savefig(metric+' cuML_vs_scikitlearn')
        plt.show()





# 儲存結果的列表
results = []


#test_samples_arr = np.arange(1000, 101000, 1000)
test_n_features_arr = np.arange(10, 1034, 10)

for features in test_n_features_arr:
    # 生成數據集
    X, y = make_regression(n_samples=1000, n_features=features, noise=12.63, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    loguru.logger.debug(f"features count: {features} ")




    # 比較 1: 傳統隨機森林 (CPU) 使用 sklearn
    print("=== 傳統隨機森林 (CPU) ===")
    start_time_rf = time.time()

    # 使用 sklearn 的 RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=5000, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    end_time_rf = time.time()

    # 計算 RMSE 和 R^2
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"傳統隨機森林 RMSE: {rmse_rf:.4f}")
    print(f"傳統隨機森林 R²: {r2_rf:.4f}")
    print(f"訓練時間: {end_time_rf - start_time_rf:.2f} 秒")

    # 儲存 RandomForest 的結果
    results.append([features, "RandomForest", rmse_rf, r2_rf, end_time_rf - start_time_rf])

    # 比較 2: 使用 cuML 加速的 Randomforest 
    print("\n=== cuML 加速的 Randomforest ===")
    start_time_cuMLRF = time.time()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')


    # 訓練模型
    n_samples = X_train.shape[0]
    cuMLRF = rfcu(n_estimators=5000, n_streams=1, n_bins=min(32, n_samples) , max_depth=10, random_state=42)
   
    # 訓練模型
    cuMLRF.fit(X_train, y_train)

    # 進行預測
    y_pred_cuMLRF = cuMLRF.predict(X_test)


    end_time_cuMLRF = time.time()

    # 計算 RMSE 和 R^2
    rmse_cuMLRF = np.sqrt(mean_squared_error(y_test, y_pred_cuMLRF))
    r2_cuMLRF = r2_score(y_test, y_pred_cuMLRF)

    print(f"cuML 加速的 Randomforest RMSE: {rmse_cuMLRF:.4f}")
    print(f"cuML 加速的 Randomforest  R²: {r2_cuMLRF:.4f}")
    print(f"訓練時間: {end_time_cuMLRF - start_time_cuMLRF:.2f} 秒")

    # 儲存 cuML 的結果
    results.append([features, "Randomforest(cuML)", rmse_cuMLRF, r2_cuMLRF, end_time_cuMLRF - start_time_cuMLRF])


# 儲存結果到 CSV
save_results_to_csv(results)

# 繪製結果
plot_results(results, test_n_features_arr)
