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
import lightgbm as lgb



# ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
# features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


# # print(features)


# df = ANA_data[features]

# X_data = df.as_matrix()
# Y_data = ANA_data['result'].as_matrix()
# print('prepare to train')
# print(datetime.now())


# iter_test = []
# iter_predictions = []


# for j in range(2):
#     kfold = StratifiedKFold(Y_data,n_folds=5,shuffle=True)
#     i = 0
#     total_tests = []
#     total_predictions = []
#     for train_index,test_index in kfold:
        
#         x_train = X_data[train_index]
#         x_test = X_data[test_index]
#         y_train = Y_data[train_index]
#         y_test = Y_data[test_index] 
        
#         dtrain = lgb.Dataset(data=x_train, 
#                              label=y_train, 
#                              free_raw_data=False, silent=True)
#         dvalid = lgb.Dataset(data=x_test, 
#                              label=y_test, 
#                              free_raw_data=False, silent=True)

        
        
#         params = {
#             'objective': 'binary',
#             'boosting_type': 'gbdt',
#             'nthread': 4,
#             'learning_rate': 0.02,  # 02,
#             'num_leaves': 20,
#             'colsample_bytree': 0.9497036,
#             'subsample': 0.8715623,
#             'subsample_freq': 1,
#             'max_depth': 8,
#             'reg_alpha': 0.041545473,
#             'reg_lambda': 0.0735294,
#             'min_split_gain': 0.0222415,
#             'min_child_weight': 60, # 39.3259775,
#             'seed': 0,
#             'verbose': -1,
#             'metric': 'auc',
#         }
#         clf = lgb.train(
#             params=params,
#             train_set=dtrain,
#             num_boost_round=10000,
#             valid_sets=[dtrain, dvalid],
#             early_stopping_rounds=200,
#             verbose_eval=False
#         )


        

#         #Predict training set: 
# #         lgb_dtest_predictions = clf.predict(x_test)
#         lgb_dtest_predictions = []
#         for i in clf.predict(x_test):
#             if i>0.5:
#                 lgb_dtest_predictions.append(1)
#             else:
#                 lgb_dtest_predictions.append(0)

        
# #         lgb_dtest_predprob = clf.predict_proba(x_test)[:,:]
#         print ("lgbClassifier Accuracy : %.4g" % metrics.accuracy_score(y_test, lgb_dtest_predictions))
        
# #         imp = pd.DataFrame({'feature': features, 'fscore': xgb.feature_importances_}).sort_values(by='fscore', ascending=False)
# #         imp.to_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\feature_importance\\lgb\\muti_xgb_imp_%d_%d.csv'%(j,i), index=None)
        
# #         result = pd.DataFrame()
# #         result['date'] = ANA_data['date'][test_index]
# #         result['id'] = ANA_data['id'][test_index]
# #         result['ISBN'] = ANA_data['ISBN'][test_index]
# #         result['path'] = ANA_data['path'][test_index]
# #         result['predict'] = lgb_dtest_predictions
# #         result['result'] = y_test
# #         result.reset_index()
# #         pred_prob = pd.DataFrame(lgb_dtest_predprob, columns=['NORMAL','BAD'])
# #         result['NORMAL'] = pred_prob['NORMAL'].values
# #         result['BAD'] = pred_prob['BAD'].values
# #         # result['ADOUBT'] = pred_prob['ADOUBT'].values
# #         result = result[(result['predict']!=result['result'])]
# #         result.to_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\result\\lgb\\muti_xgb_result_%d_%d.csv'%(j,i))
        
#         total_tests.extend(list(y_test))
#         total_predictions.extend(list(lgb_dtest_predictions))

#         i += 1
#         gc.collect()
#         sleep(1)
#     print("Total Confusion Matrix: \n", + metrics.confusion_matrix(total_tests, total_predictions))
#     print("----------------")
   
#     print(metrics.classification_report(total_tests, total_predictions))
#     print("----------------")
    
#     print ("Total Accuracy : %.4g" % metrics.accuracy_score(total_tests, total_predictions))
#     print("----------------")
    
#     iter_test.extend(total_tests)
#     iter_predictions.extend(total_predictions)
# print("Total Confusion Matrix:\n ", + metrics.confusion_matrix(iter_test, iter_predictions))
# print("----------------")

# print(metrics.classification_report(iter_test, iter_predictions))
# print("----------------")

# print("Total Accuracy : %.4g" % metrics.accuracy_score(iter_test, iter_predictions))
# print("----------------")
# print(datetime.now())
import pandas as pd
import numpy as np
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


# print(features)


df = ANA_data[features]

train_df = df.as_matrix()
test_df = ANA_data['result'].as_matrix()
print('prepare to train')
print(datetime.now())

num_folds = 5

folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
oof_preds = np.zeros(train_df.shape[0])

sum = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, test_df)):
    train_x, train_y = train_df[train_idx], test_df[train_idx]
    valid_x, valid_y = train_df[valid_idx], test_df[valid_idx]
    clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
    oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    sum += roc_auc_score(valid_y, oof_preds[valid_idx])

print("final score:%.6f" % (sum/5))














