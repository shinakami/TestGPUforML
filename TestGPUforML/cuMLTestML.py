import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import loguru

# 生成數據集
X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)

print(X)
print(y)

# 將數據集拆分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = X_train.astype('float32')


# 使用 cuML 訓練 RandomForest
rf = cuRF(n_estimators=1000, n_streams=1, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# 進行預測
y_pred = rf.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)

loguru.logger.debug(f'Accuracy: {accuracy:.4f}')

del rf
