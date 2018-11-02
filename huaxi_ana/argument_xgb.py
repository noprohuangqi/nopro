# n_estimators =30

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
import numpy as np
import gc
from time import sleep
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV



ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


# print(features)


df = ANA_data[features]

X_data = df.as_matrix()
Y_data = ANA_data['result'].as_matrix()

x_train = X_data
y_train = Y_data
# y_train = Y_data[train_index]
# y_test = Y_data[test_index] # , n_estimators=16, max_depth=6
# train


cv_params = {'n_estimators':[10,20,30,50,80,100]}
other_params = {'learing_rate':0.2,'n_estimators':10,'max_depth':5,'min_child_weight':1,'seed':2008,'subsample':0.7,'colsample_bytree':0.8}


model = XGBClassifier(**other_params)


optimized_GBM = GridSearchCV(estimator=model,param_grid=cv_params,scoring='r2',cv = 5,verbose=1)

optimized_GBM.fit(x_train,y_train)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))




# 最佳取值：{'max_depth': 3, 'min_child_weight': 4}

cv_params = {'max_depth':[3,4,5,6,7,8,9,10],'min_child_weight':[1,2,3,4,5,6]}
other_params = {'learing_rate':0.2,'n_estimators':30,'max_depth':5,'min_child_weight':1,'seed':2008,'subsample':0.7,'colsample_bytree':0.8}


# 参数的最佳取值：{'colsample_bytree': 0.8, 'subsample': 0.7}

cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
other_params = {'learing_rate':0.2,'n_estimators':30,'max_depth':3,'min_child_weight':4,'seed':2008,'subsample':0.7,'colsample_bytree':0.8}


# 参数的最佳取值：{'learning_rate': 0.07}
