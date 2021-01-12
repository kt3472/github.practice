import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pip install pycaret
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(action='ignore')


# 데이터 로드
df = pd.read_csv("housing.data", delim_whitespace = True, header=None)

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
              'DIS', 'RAD', 'TAX', 'PTRATIO', 'B1000', 'LSTAT', 'MEDV']


#  Null / Na 값 포함 여부 확인
print("*** null 값 포함여부 확인 ***")
print(df.isnull().sum())
print("*** na 값 포함여부 확인 ***")
print(df.isna().sum())

# 중복데이터 삭제
df.drop_duplicates(keep = 'first', inplace = True)



# 각 피쳐별 박스플롯, 히스토그램, 기초통계량 시각화
# RuntimeError: Selected KDE bandwidth is 0. Cannot estiamte density.에러 발생시,
# sns.distributions._has_statsmodels = False 실행

for i in df.columns:
    
    f, axes = plt.subplots(2, 1, figsize=(11, 6))

    sns.distplot(df[i], ax = axes[0]).set_title("Distribution of "+i)
    sns.boxplot(df[i], ax = axes[1])
    plt.show()
    
    print(round(df[i].describe(), 4))
    print('Skewness         ',round(df[i].skew(),4))
    print('Kurtosis         ',round(df[i].kurtosis(),4))



# 각 피쳐간 상관관계 메트릭스 출력
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 10)) 
plt.title("Person Correlation of Features", y = 1, size = 12) 
sns.heatmap(df.astype(float).corr(), linewidths = 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = "white", annot = True, 
            annot_kws = {"size" : 11})



# 각 피쳐와 집값(MEDV)간 산점도 시각화
plt.figure(figsize = (20, 30))

for i, col in enumerate(df.columns):
    
    plt.subplot(5,4, i+1)
    
    x = df[col]
    y = df['MEDV']
    plt.plot(x,y,'o')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x,y, 1))(np.unique(x)))
    plt.xlabel(col, fontsize = 20)
    plt.ylabel('MEDV', fontsize = 20)  


# 모델 알고리즘 선택
X = df.drop(['MEDV'], axis = 1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=2020)
X_train['MEDV'] = y_train


# AutoML setup실행
# 실행 후 입력창이 나오면 엔터클릭

clf = setup(data = X_train, target = 'MEDV')


#AutoML패키지의 campare_model기능을 이용해 MAE기준 상위 2개 알고리즘을 선택

best_two_model = compare_models(sort = 'MAE', n_select = 2, fold= 5)


# Extra Trees 모델 생성 및 예측값과 실제값의 MAE 계산
print("---- create Extra Trees model ----\n")
et_model = create_model('et', fold = 5)
et_model_final = finalize_model(et_model)
et_predictions = predict_model(et_model_final, data = X_test)
et_predictions.head()

print("MAE on test data (Extra Trees) : {}".format(np.mean(abs(et_predictions['Label'] - y_test))))
# MAE on test data (Extra Trees) : 2.233708661417323


# Catboost 모델 생성 및 예측값과 실제값의 MAE 계산
print("---- create CatBoost model ----\n")
cb_model = create_model('catboost', fold = 5)
cb_model_final = finalize_model(cb_model)
cb_predictions = predict_model(cb_model_final, data = X_test)
cb_predictions.head()

print("MAE on test data (CatBoost) : {}".format(np.mean(abs(cb_predictions['Label'] - y_test))))
# MAE on test data (CatBoost) : 2.260665354330709


# Neural Network 모델 생성 및 예측값과 실제값의 MAE 계산

X = df.drop(['MEDV'], axis = 1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=2020)

# 변수 정규화
mean = X.mean(axis=0)
std = X.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# NN 모델 
model = Sequential()
model.add(Dense(128, input_shape = (13, ), activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))   

model.compile(optimizer = 'adam', loss = 'mae', metrics = ['mae'])
model.summary()

history = model.fit(X_train, y_train, epochs = 200, shuffle=True, validation_split= 0.10, verbose=1)

mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('MAE on test data (Neural Network): ', mae_nn)
# MAE on test data (Neural Network):  2.2789268493652344


# 피쳐 선택 및 모델 파라미터 튜닝

df_fe = df.copy()

# 피쳐별 상관관계 메트릭스 출력

colormap = plt.cm.PuBu
plt.figure(figsize=(15, 10)) 
plt.title("Person Correlation of Features", y = 1, size = 12) 
sns.heatmap(df_fe.astype(float).corr(), linewidths = 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = "white", annot = True, 
            annot_kws = {"size" : 11})


# 각 피쳐별 아웃라이어값으로 추정되는 값의 비율 계산

for k, v in df_fe.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df_fe)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))


# MEDV의 아웃라이어 값 제거(MEDV의 값이 50 이상인 행을 삭제)

df_fe = df_fe[~(df_fe['MEDV'] >= 50.0)]


#RAD과 TAX는 서로 상관관계(0.91)가 높아 둘중 한가지 변수만 사용(TAX와 MEDV의 음의 상관관계가 더 높아 RAD피쳐 삭제)
#MEDV와 상관관계가 낮은(0에 가까운) CHAS피쳐 제외

X = df_fe.drop(['MEDV'], axis = 1)
X = X.drop(['CHAS'], axis = 1)
X = X.drop(['RAD'], axis = 1)
y = df_fe['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=2020)
X_train['MEDV'] = y_train


# 피쳐선택 후 데이터로 AutoML 패키지를 이용하여 Catboost, Extra Trees 알고리즘 적용
clf = setup(data = X_train, target = 'MEDV')


# Catboost 모델 생성 및 예측값과 실제값의 MAE 계산
print("---- create catboost model ----\n")
cb_model = create_model('catboost', fold = 5)
cb_model_final = finalize_model(cb_model)
cb_predictions = predict_model(cb_model_final, data = X_test)
cb_predictions.head()

print("MAE on test data (CatBoost) : {}".format(np.mean(abs(cb_predictions['Label'] - y_test))))
# MAE on test data (CatBoost) : 1.7560373983739828


#Extra Trees 모델 생성 및 예측값과 실제값의 MAE 계산
print("---- create Extra trees model ----\n")
et_model = create_model('et', fold = 5)
et_model_final = finalize_model(et_model)
et_predictions = predict_model(et_model_final, data = X_test)
et_predictions.head()
print("MAE on test data (Extra trees Boosting) : {}".format(np.mean(abs(et_predictions['Label'] - y_test))))
# MAE on test data (Extra trees Boosting) : 1.8189674796747968


#Neural Network 모델 생성 및 예측값과 실제값의 MAE 계산

X = df_fe.drop(['MEDV'], axis = 1)
X = X.drop(['CHAS'], axis = 1)
X = X.drop(['RAD'], axis = 1)
y = df_fe['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=2020)

mean = X.mean(axis=0)
std = X.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Neural Network 모델의 최적 파라미터 산출(Gridsearch)

def create_model_for_par(neurons, second_layer=True, input_dims = X_train.shape[1]):    
    
    print(input_dims)
    
    model = Sequential()
    model.add(Dense(neurons, input_shape = (input_dims,), activation = 'relu'))
    model.add(Dense(int(neurons/2), activation = 'relu'))
    if second_layer:
        model.add(Dense(int(neurons/2), activation = 'relu'))
    model.add(Dense(int(neurons/4), activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
    
    return model


model = KerasRegressor(build_fn = create_model_for_par, verbose = 0)


# 파라미터 리스트 

param_grid = dict(
    neurons=[1024,512, 256, 128, 64],
    second_layer=[True, False],
    epochs = [50,100,200,300]
    )


## %%time
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
grid_result = grid.fit(X_train, y_train)


print("Best estimator: " + str(grid.best_params_))
# Best estimator: {'epochs': 100, 'neurons': 256, 'second_layer': True}


# 최적 파라미터를 적용하여 Neural Network 모델 생성 및 예측값과 실제값의 MAE 계산
model = Sequential()
model.add(Dense(256, input_shape = (X_train.shape[1], ), activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))  
model.compile(loss = 'mae', optimizer='adam', metrics = ['mae'])
model.summary()

history = model.fit(X_train, y_train, epochs = 100, shuffle=True, validation_split= 0.10, verbose=1)

nn_predictions = model.predict(X_test)
nn_predictions = pd.DataFrame(nn_predictions)
nn_predictions.index = y_test.index

mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('MAE on test data (Neural Network): ', mae_nn)
# MAE on test data (Neural Network):  1.7830954790115356


# Extra trees 피처별 중요도 시각화

print("*** Extra trees Feature Importance")
plot_model(et_model_final, plot='feature')




# Catboost 피처별 중요도 시각화
cb_feature_importance_df = pd.DataFrame.from_dict(dict(zip(cb_model_final.feature_names_,
                                                        cb_model_final.feature_importances_)),orient="index",columns=["feature_value"]) 

cb_feature_importance_df.sort_values(by=["feature_value"],ascending=True, inplace = True)

cb_feature_importance_df.plot(kind='barh')



#  Neural Network 피처별 중요도 시각화

import shap

shap.initjs()

explainer = shap.DeepExplainer(model, X_train[:100].values)
shap_values = explainer.shap_values(X_test[:100].values)
#shap_values = explainer.shap_values(X_test[:100])
#explainer = shap.DeepExplainer(model, X_train[:100])

shap.summary_plot(shap_values, X_test, plot_type='bar')


# Catboost, Extra Trees, Neural Network 모델의 예측지를 합한 후 평균을 산출
blend_models = (cb_predictions['Label'] + et_predictions['Label'] + nn_predictions[0])/3
print("MAE on test data (average 3 models) : {}".format(np.mean(abs(blend_models - y_test))))