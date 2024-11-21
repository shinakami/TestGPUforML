import xgboost as xgb # type: ignore
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor  # type: ignore
import numpy as np

# 生成隨機數據集
X, y = make_regression(n_samples=100000, n_features=50, noise=0.2)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 建立 XGBoost 隨機森林回歸模型
rf_xgb = XGBRFRegressor(
    n_estimators=1000,       # 樹的數量
    subsample=0.8,          # 每棵樹使用的樣本比例
    colsample_bynode=0.8,   # 每個節點使用的特徵比例
    random_state=42,
    device='cuda',  # 使用 GPU 加速
    verbosity=2
)

# 訓練模型
rf_xgb.fit(X_train, y_train)

# 預測
y_pred_rf = rf_xgb.predict(X_test)
y_pred_rf = np.array(y_pred_rf)

print(y_pred_rf)
