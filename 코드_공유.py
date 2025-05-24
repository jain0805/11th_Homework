# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# feature와 target 분리
X = df.drop('target', axis=1)
y = df['target']

# train/test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
param_grid = {
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200, 300]
}

from xgboost import XGBClassifier
xgb = XGBClassifier()
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters:", best_params)
best_score = grid_search.best_score_
print("Best accuracy:", best_score)

xgb_best = grid_search.best_estimator_
importances = xgb_best.feature_importances_
features = X.columns

plt.figure(figsize=(15,6))
plt.title('Feature Importance')
plt.bar(features, importances, width=0.4)
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("importance")
plt.tight_layout()
plt.show()
