
'''模型融合中使用到的各个单模型'''
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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



clfs = [LGBMClassifier(
            nthread=4,
            n_estimators=1000,
            learning_rate=0.28,
            num_leaves=20,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=5,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, ),
        xgb.XGBClassifier(max_depth=6,n_estimators=1000,num_round = 5),
        RandomForestClassifier(n_estimators=1000,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=1000)]

ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


df = ANA_data[features]

train_df = df.as_matrix()
test_df = ANA_data['result'].as_matrix()
X = train_df[:6800]
# X_test = 
y = test_df[:6800]
X_predict = train_df[6800:]
# X = train_data_X_sd\n",
#     "X_predict = test_data_X_sd\n",
#     "y = train_data_Y\n",

#创建n_folds
from sklearn.cross_validation import StratifiedKFold
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

#创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

#建立模型
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# 用建立第二层模型
clf2 = LogisticRegression(C=0.1,max_iter=100)
clf2.fit(dataset_blend_train, y)
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]


test = pd.read_csv("test.csv")
test["Survived"] = clf2.predict(dataset_blend_test)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('stack3.csv')


from sklearn import svm
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


clfs = [LGBMClassifier(
            nthread=4,
            n_estimators=1000,
            learning_rate=0.28,
            num_leaves=20,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=5,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, ),
        xgb.XGBClassifier(max_depth=6,n_estimators=1000,num_round = 5),
        RandomForestClassifier(n_estimators=1000,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=1000)]

ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\feature_1940_4_fuse.csv') # ANA_data_traditional ANA_data
features = [x for x in ANA_data.columns if x not in ['Unnamed: 0','result','date','id','ISBN','path','project','filename','min_value']]


df = ANA_data[features]

train_df = df.as_matrix()
test_df = ANA_data['result'].as_matrix()
t0 = time.time()
sum = 0
for circle in range(3):
#     X_train,X_test, y_train, y_test =  cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0) 

    X,X_predict,y,X_result = cross_validation.train_test_split(train_df,test_df,test_size=0.15, random_state=circle)
    
#     print(X_predict)
#     X = train_df[:6800]
#     # X_test = 
#     y = test_df[:6800]
#     X_predict = train_df[6800:]

    # X = train_data_X_sd\n",
    #     "X_predict = test_data_X_sd\n",
    #     "y = train_data_Y\n",

    #创建n_folds
    
    
    from sklearn.cross_validation import StratifiedKFold
    n_folds = 5
    skf = list(StratifiedKFold(y, n_folds))

    # #创建零矩阵
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

    #建立模型
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            # print("Fold", i)
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission

            dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:,1]
    #     '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
            print("这是第%d/3轮训练  第%d/4个模型  第%d/5个部分" % (circle+1,j+1,i+1))
            print("耗时 %d s" % (time.time()-t0))
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
#         print("第%d 轮训练的第 %d 个模型训练结束" % (circle+1,j+1))

    # 用建立第二层模型
    clf2 = LogisticRegression(C=0.1,max_iter=100)
    clf2.fit(dataset_blend_train, y)
    y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]
    from sklearn.metrics import roc_auc_score, roc_curve
    
    y_submission= [int(round(num)) for num in y_submission]
    
    sum+=accuracy_score(X_result, y_submission)
    print('Fold %2d AUC : %.6f' % ( circle+1, accuracy_score(X_result, y_submission)))


print("总的精度在 %.6f " % (sum/3))
