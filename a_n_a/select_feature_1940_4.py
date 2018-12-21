import pandas as pd
import pickle
import numpy as np
import math
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import time
import datetime

def feature_computer(p):
    idm = 0.0
    hom = 0.0
    cm_energy = 0.0
    cm_contrast = 0.0
    cm_entropy = 0.0
    max_prob = -10000.0
    for i in range(gray_level):
        for j in range(gray_level):
            if p[i][j] > max_prob:
                max_prob = p[i][j]
            cm_energy += p[i][j] * p[i][j]
            cm_contrast += math.pow(i-j,2) * p[i][j]
            if p[i][j]>0:
                cm_entropy -= p[i][j] * math.log(p[i][j])
            idm += p[i][j] / (1+(i-j)*(i-j)) # 均匀性
            hom += p[i][j] / (1+abs(i-j)) # 均匀性
    return idm,hom, cm_energy,cm_contrast, cm_entropy# 均匀性 同质化


def get_region_info(region_img,region_RGB):
    col = region_img.shape[1]
    row = region_img.shape[0]
    region_img_mat = np.array(region_img,dtype=np.int32)
    region_RGB_mat = np.array(region_RGB,dtype=np.int32)
    max_value = 0
    mean_value = 0
    max_value_G = 0
    max_value_R = 0
    max_value_B = 0
    mean_R = 0
    mean_G = 0
    mean_B = 0
    total = 0
    #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    total = np.sum(region_img_mat)
    max_value = np.max(region_img_mat)
    max_value_B = np.max(region_RGB[:,:,0])
    max_value_G = np.max(region_RGB[:,:,1])
    max_value_R = np.max(region_RGB[:,:,2])
#     for i in range(row):
#         for j in range(col):
#             total += int(region_img[i][j])
#             if (int(region_img[i][j]) > max_value):
#                 max_value = int(region_img[i][j])
#             if (int(region_RGB[i][j][0]) > max_value_B):
#                 max_value_B = int(region_RGB[i][j][0])
#             if (int(region_RGB[i][j][1]) > max_value_G):
#                 max_value_G = int(region_RGB[i][j][1])
#             if (int(region_RGB[i][j][2]) > max_value_R):
#                 max_value_R = int(region_RGB[i][j][2])
    
    number = col * row
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    mean_value = np.mean(region_img)
    mean_B = np.mean(region_RGB[:,:,0])
    mean_G = np.mean(region_RGB[:,:,1])
    mean_R = np.mean(region_RGB[:,:,2])
#     for i in range(row):
#         for j in range(col):
#             number += 1
#             mean_value += int(region_img[i][j])
#             mean_B += int(region_RGB[i][j][0])
#             mean_G += int(region_RGB[i][j][1])
#             mean_R += int(region_RGB[i][j][2])

    mean_value = mean_value / number
    mean_B = mean_B / number
    mean_G = mean_G / number
    mean_R = mean_R / number
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #方差
    varis = np.var(region_img)
    varis_B = np.var(region_RGB[:,:,0])
    varis_G = np.var(region_RGB[:,:,1])
    varis_R = np.var(region_RGB[:,:,2])

    cov_gray = np.cov(region_img).mean()
    cov_B = np.cov(region_RGB[:,:,0]).mean()
    cov_G = np.cov(region_RGB[:,:,1]).mean()
    cov_R = np.cov(region_RGB[:,:,2]).mean()
    
    


    # print("xxxx")
    # print(cov_R)
#     varis = 0.0
#     for i in range(row):
#         for j in range(col):
#             varis += (int(region_img[i][j])-mean_value) * (int(region_img[i][j])-mean_value)
#     varis = varis / (col * row)
#     varis_B = 0.0
#     for i in range(row):
#         for j in range(col):
#             varis_B += (int(region_RGB[i][j][0])-mean_value) * (int(region_RGB[i][j][0])-mean_value)
#     varis_B = varis_B / (col * row)
#     varis_G = 0.0
#     for i in range(row):
#         for j in range(col):
#             varis_G += (int(region_RGB[i][j][1])-mean_value) * (int(region_RGB[i][j][1])-mean_value)
#     varis_G = varis_G / (col * row)
#     varis_R = 0.0
#     for i in range(row):
#         for j in range(col):
#             varis_R += (int(region_RGB[i][j][2])-mean_value) * (int(region_RGB[i][j][2])-mean_value)
#     varis_R = varis_R / (col * row)
    
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    glcm_0 = getGlcm(region_img, 0, 1)
    idm, hom, cm_energy,cm_contrast, cm_entropy = feature_computer(glcm_0)

    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    hist = cv2.calcHist([region_img], [0], None, [16], [0.0,255.0])
    hist = np.array(hist).reshape(-1)
    # 获得图像的一致性、熵
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    gray_statis,num = get_img_gray_statistics(region_img)
    gray_prob = get_img_gray_prob(gray_statis,num)

    hist_mean = get_img_mean(gray_prob)
    hist_varis = get_img_varis(gray_prob,gray_statis,hist_mean)



    consist = get_img_consistency(gray_prob)
    entropy = get_img_entropy(gray_prob)

    skewness = get_img_skewness(gray_prob,hist_mean,hist_varis)
    kurtosis = get_img_kurtosis(gray_prob,hist_mean,hist_varis)
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    base_info = np.array([skewness,kurtosis,max_value,max_value_B,max_value_G,max_value_R,mean_value,consist,entropy,mean_R,mean_G,mean_B,idm,hom,cm_energy,cm_contrast, cm_entropy,varis,varis_B,varis_G,varis_R,cov_B,cov_G,cov_R])
    return np.concatenate([base_info,hist])

def get_img_mean(gray_prob):
    mean = 0.0
    for gray_value in gray_prob.keys():
        mean += gray_prob[gray_value]*int(gray_value)
    return mean

def get_img_varis(gray_prob,gray_statis,mean):
    varis = 0.0
    for gray_value in gray_prob.keys():
        varis += gray_prob[gray_value]*( gray_statis[gray_value] - mean)
    return varis



def get_img_entropy(gray_prob):
    entropy = 0.0
    for gray_value in gray_prob.keys():
        entropy -= gray_prob[gray_value] * math.log(gray_prob[gray_value])
    return entropy


def get_img_skewness(gray_prob,mean,varis):
    if varis == 0.0:
        return 0.0
    skewness = 0.0
    for gray_value in gray_prob.keys():
        skewness += math.pow(gray_value-mean, 3) * gray_prob[gray_value]
    return skewness / varis

'''获取图像的峰态，是否存在与均值'''
def get_img_kurtosis(gray_prob,mean,varis):
    if varis == 0.0:
        return 0.0
    kurtosis = 0.0
    for gray_value in gray_prob.keys():
        kurtosis += math.pow(gray_value-mean, 4) / math.pow(gray_prob[gray_value], 3)
    return kurtosis / math.pow(varis, 2)


def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
#     print(height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1




def get_img_consistency(gray_prob):
    consist = 0.0
    for gray_value in gray_prob.keys():
        consist += gray_prob[gray_value] * gray_prob[gray_value]
    return consist


def get_img_gray_prob(gray_statis,num):
    gray_prob = {} # 声明存放凸显灰度值概率的容器
    for gray_value in gray_statis.keys():
        gray_value_num = gray_statis[gray_value]
        gray_prob[gray_value] = gray_value_num / num
    return gray_prob

def get_img_gray_statistics(region_img,max_thresh=255):
    col = region_img.shape[1]
    row = region_img.shape[0]
    gray_statis = {} # 声明存放凸显灰度值统计的容器
    num = 0
    for i in range(row):
        for j in range(col):
            gray_value = int(region_img[i][j])
            if (gray_value <= max_thresh):
                if(gray_value in gray_statis):
                    old_value = gray_statis[gray_value]
                    gray_statis[gray_value] = old_value + 1
                else:
                    gray_statis[gray_value] = 1
                num += 1
    
    return gray_statis,num

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)

    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i+d_x]
            ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)
    return ret



if __name__ == "__main__":
    # with open('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA-inspection-imgpaths.data', 'rb') as path_file:
    #     imgs = pickle.load(path_file)
    gray_level = 256

    batch = []
    t0 = time.time()
    
    i = 0
    for num in range(20):

        with open('C:\\Users\\32002\\Desktop\\ANA_inspection\\ANA-img-candidate\\ANA-img-candidate-%d.mat' % num, 'rb') as path_file:
            imgs = pickle.load(path_file)

        length_imgs = len(imgs)

        for num2 in range(length_imgs):#100


            for num3 in range(len(imgs[num2])):#4

                if len(imgs[num2])<4:
                    print(("此处有毒,只有 %d 个bboxes \n" % num3)*5)
                img_data = imgs[num2][num3]
                img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                img_info = get_region_info(img_gray,img_data)
                batch.append(img_info)
                
            i+=1
#             print("已经处理掉 %d /1940 " % i)
            if i%20 ==0:
                print("处理 %d 张图片耗时 %d" % (i,(time.time()-t0)))
          
    with open('svm/feature_split_img_1940_4','wb') as data_file:
        pickle.dump(batch, data_file)
    print("over!")  


    # 拼接特征与与预测值
    with open('svm/feature_split_img_1940_4','rb') as data_file:
        batch = pickle.load(data_file) 

    feat_column_names = ['skewness','kurtosis','max_value','max_value_B','max_value_G','max_value_R','mean_value','consist','entropy','mean_R','mean_G','mean_B',\
                     'idm','hom',' cm_energy','cm_contrast', 'cm_entropy','varis','varis_B','varis_G','varis_R','hist1','hist2','hist3','hist4','hist5','hist6','hist7','hist8',\
                     'hist9','hist10','hist11','hist12','hist13','hist14','hist15','hist16','cov_B','cov_G','cov_R']

    feat_tradition = pd.DataFrame(batch,columns = feat_column_names)
    ANA_data = pd.read_csv('C:\\Users\\32002\\Desktop\\ANA_inspection\\svm\\data\\ANA_data_traditional.csv') # ANA_data_traditional ANA_data
    temp = []
    k=0
    while(True):
        
        i = 0
        for ii in range(4):
            
            temp.append(ANA_data['result'][k])
            i+=1
        k+=1
        if k == 1940:
            break
        print(k)

    temp = pd.DataFrame(temp,columns = ['result'])
    ANA_feature = pd.concat([feat_tradition,temp],axis=1)
    ANA_feature.to_csv('svm/final_data_1940_4.csv',index = 0)


