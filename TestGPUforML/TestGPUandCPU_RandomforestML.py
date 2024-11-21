import time
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb # type: ignore
import loguru
import matplotlib.pyplot as plt

# 儲存結果的函式
def save_results_to_csv(results, filename="results.csv"):
    df = pd.DataFrame(results, columns=["sample_count", "model", "rmse", "r2", "training_time"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# 繪圖的函式
def plot_results(results, test_samples_arr):
    df = pd.DataFrame(results, columns=["sample_count", "model", "rmse", "r2", "training_time"])

    models = df['model'].unique()

    for metric in ['rmse', 'r2', 'training_time']:
        plt.figure(figsize=(10, 6))
        for model in models:
            model_data = df[df['model'] == model]
            plt.plot(model_data['sample_count'], model_data[metric], label=model)

        plt.title(f"{metric.upper()} vs Sample Count")
        plt.xlabel('Sample Count')
        plt.ylabel(metric.upper())
        plt.legend(title="Model", loc="upper right")
        plt.grid(True)
        plt.savefig(metric)
        plt.show()





# 儲存結果的列表
results = []


test_samples_arr = np.arange(1000, 101000, 1000)

for samples in test_samples_arr:
    # 生成數據集
    X, y = make_regression(n_samples=samples, n_features=29, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    loguru.logger.debug(f"sample count: {samples} ")

    # 將數據轉換為 DMatrix (給XGBoost運算的數據格式)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)



    # 比較 1: 傳統隨機森林 (CPU) 使用 sklearn
    print("=== 傳統隨機森林 (CPU) ===")
    start_time_rf = time.time()

    # 使用 sklearn 的 RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=17, random_state=42)
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
    results.append([samples, "RandomForest", rmse_rf, r2_rf, end_time_rf - start_time_rf])

    # 比較 2: 使用 GPU 加速的 XGBoost 
    print("\n=== GPU 加速的 XGBoost ===")
    start_time_xgb = time.time()



    # 訓練模型
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 17,
        'learning_rate': 0.1,
        'tree_method': 'hist',  # 使用 histogram 方法
        'device': 'cuda',  # 使用 GPU
        'random_state': 42,
    }

    # 使用 DMatrix 進行訓練
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # 使用 DMatrix 進行預測
    y_pred_xgb_rf = bst.predict(dtest)


    end_time_xgb = time.time()

    # 計算 RMSE 和 R^2
    rmse_xgb_rf = np.sqrt(mean_squared_error(y_test, y_pred_xgb_rf))
    r2_xgb_rf = r2_score(y_test, y_pred_xgb_rf)

    print(f"GPU 加速的 XGBoost RMSE: {rmse_xgb_rf:.4f}")
    print(f"GPU 加速的 XGBoost  R²: {r2_xgb_rf:.4f}")
    print(f"訓練時間: {end_time_xgb - start_time_xgb:.2f} 秒")

    # 儲存 XGBoost 的結果
    results.append([samples, "XGBoost(GPU)", rmse_xgb_rf, r2_xgb_rf, end_time_xgb - start_time_xgb])


# 儲存結果到 CSV
save_results_to_csv(results)

# 繪製結果
plot_results(results, test_samples_arr)
