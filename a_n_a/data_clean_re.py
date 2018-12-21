import pandas as pd
import numpy as np
import pickle
import os,sys
import shutil

ANA_data = pd.read_csv('data_new.csv')

data = ANA_data[(ANA_data['result']!='可疑（±）') & (~pd.isna(ANA_data['result'])) &  (ANA_data['result']!='未做')]

numerical_series_name = 'result'
def filter_numerical_series(df):
    result = ''
    if ((not pd.isnull(df[numerical_series_name])) and (str(df[numerical_series_name]) == '阴性（－）')):
        result = 0
    else:
        result = 1
    return result


data['result'] = data.apply(filter_numerical_series, axis=1)



def change_file_name(file_dir):
    for root,dirs,files in os.walk(file_dir):
        for file in flies:
            if os.path.splitext(file)[1] =='jpg':
                file_names = file.split('_')
                if len(file_names) ==6:
                    os.rename(os.path.join(root,file),os.path.join(root,file_names[0]+'-'+file_names[1]+'-'+file_names[3]+'.jpg'))

change_file_name("img_path")

dates = data['date'].data
ids = data['id'].data
INSPECTION_IDs = data['ISBN'].data
paths = []
for i in range(len(dates)):
    if len(str(INSPECTION_IDs[i]))>10:
        paths.append('%d-%d-'%(dates[i],ids[i])+str(INSPECTION_IDs[i]).zfill(12)+'.jpg')
    else:
        paths.append('%d-%d-%d.jpg'%(dates[i],ids[i],INSPECTION_IDs[i]))
data['path'] = paths


def is_file_exist(paths):
    if (len(paths)==0):
        return list()
    
    path_exists = list()
    for path in paths:
        if (os.path.exists('imgs/%s'%(path))):
            path_exists.append(1)
        else:
            path_exists.append(0)
    return pd.Series(path_exists)

data = data.reset_index()
data = data[(is_file_exist(data['path'])==1)]
data = data.reset_index()

data.to_csv('data.final.csv')
with open('ANA-inspection-imgpaths.data','wb') as data_file:
    pickle.dump(paths, data_file)

