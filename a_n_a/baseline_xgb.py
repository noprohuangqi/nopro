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

ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


# print(features)


df = ANA_data[features]

X_data = df.as_matrix()
Y_data = ANA_data['result'].as_matrix()
print('prepare to train')
print(datetime.now())


iter_test = []
iter_predictions = []


for j in range(2):
    kfold = StratifiedKFold(Y_data,n_folds=7,shuffle=True)
    i = 0
    total_tests = []
    total_predictions = []
    for train_index,test_index in kfold:
        print(datetime.now())
        x_train = X_data[train_index]
        x_test = X_data[test_index]
        y_train = Y_data[train_index]
        y_test = Y_data[test_index] # , n_estimators=16, max_depth=6
        # train
        xgb = XGBClassifier(learning_rate =0.07, n_estimators=30, max_depth=3,min_child_weight=4, # 10 5
            subsample=0.7, colsample_bytree=0.8, seed=2018)
        # =100, max_depth=10,min_child_weight=1 97.30
        
        #Fit the algorithm on the data
        xgb.fit(x_train, y_train)

        #Predict training set: 
        lgb_dtest_predictions = xgb.predict(x_test)
        lgb_dtest_predprob = xgb.predict_proba(x_test)[:,:]
        print ("xgbClassifier Accuracy : %.4g" % metrics.accuracy_score(y_test, lgb_dtest_predictions))
        
        imp = pd.DataFrame({'feature': features, 'fscore': xgb.feature_importances_}).sort_values(by='fscore', ascending=False)
        imp.to_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\feature_importance\\muti_xgb_imp_%d_%d.csv'%(j,i), index=None)
        
        result = pd.DataFrame()
        result['date'] = ANA_data['date'][test_index]
        result['id'] = ANA_data['id'][test_index]
        result['ISBN'] = ANA_data['ISBN'][test_index]
        result['path'] = ANA_data['path'][test_index]
        result['predict'] = lgb_dtest_predictions
        result['result'] = y_test
        result.reset_index()
        pred_prob = pd.DataFrame(lgb_dtest_predprob, columns=['NORMAL','BAD'])
        result['NORMAL'] = pred_prob['NORMAL'].values
        result['BAD'] = pred_prob['BAD'].values
        # result['ADOUBT'] = pred_prob['ADOUBT'].values
        result = result[(result['predict']!=result['result'])]
        result.to_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\result\\muti_xgb_result_%d_%d.csv'%(j,i))
        
        total_tests.extend(list(y_test))
        total_predictions.extend(list(lgb_dtest_predictions))

        i += 1
        gc.collect()
        sleep(1)
    print("Total Confusion Matrix: \n", + metrics.confusion_matrix(total_tests, total_predictions))
    print("----------------")
   
    print(metrics.classification_report(total_tests, total_predictions))
    print("----------------")
    
    print ("Total Accuracy : %.4g" % metrics.accuracy_score(total_tests, total_predictions))
    print("----------------")
    
    iter_test.extend(total_tests)
    iter_predictions.extend(total_predictions)
print("Total Confusion Matrix:\n ", + metrics.confusion_matrix(iter_test, iter_predictions))
print("----------------")

print(metrics.classification_report(iter_test, iter_predictions))
print("----------------")

print("Total Accuracy : %.4g" % metrics.accuracy_score(iter_test, iter_predictions))
print("----------------")
print(datetime.now())