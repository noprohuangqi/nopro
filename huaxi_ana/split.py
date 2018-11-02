from __future__ import (
    division,
    print_function,
)
import os
import time
import skimage.data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import selectivesearch
from skimage import io,transform
import cv2
import selectivesearch2
import selectivesearch4
import pickle

def mat_inter(box1,box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
 
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False
    
def mat_inner(box1,box2,thresh):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    if (x01<=x11 and x02>=x12 and y01<=y11 and y02>=y12):
        return True 
        # if float((abs(x11 - x12) * abs(y11 - y12)) / (abs(x01 - x02) * abs(y01 - y02)))>thresh else False
    if (x01>=x11 and x02<=x12 and y01>=y11 and y02<=y12):
        return True 
        # if float((abs(x01 - x02) * abs(y01 - y02)) / (abs(x11 - x12) * abs(y11 - y12)))>thresh else False
    return False
 
def get_coincide_ratio(box1,box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inner(box1,box2,0.6):
        return 1.0
    if mat_inter(box1,box2)==True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=intersection/(area1+area2-intersection)
        return coincide
    else:
        return 0.0

def get_coincide_ratio_V2(box1, box2):
    '''
    计算两个矩形的重合度
    :param box1:
    :param box2:
    :return:
    '''
    if mat_inner(box1, box2, 0.6):
        return 1.0
    if mat_inter(box1, box2) == True:
        y01, x01, y02, x02 = box1
        y11, x11, y12, x12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / area1
        return coincide
    else:
        return 0.0
    

def get_size(box):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    return (box[2]-box[0]) * (box[3]-box[1])

def get_green_pixel_ratio(img_green_np):
    return len(np.where(img_green_np > 128)[0]) / (img_green_np.shape[0] * img_green_np.shape[1])
def get_cell_positions(img_data):
    area = img_data.shape[0] * img_data.shape[1]
    SP_index = 0
    # 图像二值化
    img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    ret1, biImage = cv2.threshold(img_gray,230,255,cv2.THRESH_TRUNC)
    ret1, biImage = cv2.threshold(biImage, 200, 255, cv2.THRESH_OTSU)
    image_pixels = np.array(biImage,dtype=np.int32)
    posi = np.where(image_pixels == 0)
    x_axis_info = posi[0].reshape(-1,1)
    y_axis_info = posi[1].reshape(-1,1)
    return np.concatenate((x_axis_info,y_axis_info),axis=1)

# def kMeansDictionary(training, k):
#     if training.shape[0] > 250000:
#         training = RandomSampling(training, 250000)
#     #K-means algorithm
#     est = KMeans(n_clusters=k,tol=0.06,max_iter=88).fit(training)
#     return est

def get_bboxes(cluster_centers,img_height,img_width):
    candidates_boxes = []
    for cluster_center in cluster_centers:
        # print(cluster_center)
        
        center_y = int((cluster_center[0]+cluster_center[2])/2)
        center_x = int((cluster_center[1]+cluster_center[3])/2)
        if center_y+168>img_height:
            col1 = img_height - 336
            col2 = img_height
        elif center_y-168<0:
            col1 = 0
            col2 = 336
        else:
            col1 = center_y - 168
            col2 = center_y + 168
            
        if center_x+168>img_width:
            row1 = img_width - 336
            row2 = img_width
        elif center_x-168<0:
            row1 = 0
            row2 = 336
        else:
            row1 = center_x - 168
            row2 = center_x + 168

        candidates_boxes.append([col1, row1, col2, row2])
    return candidates_boxes
def filter_intersected_box(candidates):
    '''
    过滤重叠区域过大的 box中的一个【面积较小的那个】
    :param candidates:
    :param width:
    :param height:
    :return:
    '''
    print('第二次过滤重叠区域前box个数为 %d' % (len(candidates)))
    flag = np.zeros((len(candidates)))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            coincide = get_coincide_ratio_V1(candidates[i], candidates[j])
            if (coincide > 0.8):
                if (get_size(candidates[j]) > get_size(candidates[i])):
                    if (flag[j] == 0):
                        flag[i] = 1
                else:
                    if (flag[i] == 0):
                        flag[j] = 1
    for j in range(len(candidates) - 1, -1, -1):
        if (flag[j] == 1):
            candidates.remove(candidates[j])
    print('第二次过滤重叠区域后box个数为 %d' % (len(candidates)))
    return candidates

def get_coincide_ratio_V1(box1, box2):
    '''
    计算两个矩形的重合度
    :param box1:
    :param box2:
    :return:
    '''
    if mat_inner(box1, box2, 0.6):
        return 1.0
    if mat_inter(box1, box2) == True:
        y01, x01, y02, x02 = box1
        y11, x11, y12, x12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return 0.0
def filter_polluted_box(candidates,polluted_boxes):
    '''
    过滤重叠区域过大的 box中的一个【面积较小的那个】
    :param candidates:
    :param width:
    :param height:
    :return:
    '''
    print('过滤污染区域box前box个数为 %d' % (len(candidates)))
    flag = np.zeros((len(candidates)))
    for i in range(len(polluted_boxes)):
        for j in range(len(candidates)):
            coincide = get_coincide_ratio_V2(candidates[j], polluted_boxes[i])
            if (coincide > 0.1):
                flag[j] = 1
                
    for j in range(len(candidates) - 1, -1, -1):
        if (flag[j] == 1):
            candidates.remove(candidates[j])
    print('过滤污染区域box后box个数为 %d' % (len(candidates)))
    return candidates

def get_candidate_region_greenmean(img_green_np, candidates_boxes):
    candidates_boxes_mean = np.zeros(len(candidates_boxes))
    for i in range(len(candidates_boxes)):
        candidates_boxes_mean[i] = np.mean(img_green_np[candidates_boxes[i][0]:candidates_boxes[i][2],candidates_boxes[i][1]:candidates_boxes[i][3]])
    return candidates_boxes_mean

# def get_polluted_regions(img_data,green_ratio):
#     area = img_data.shape[0] * img_data.shape[1]
#     SP_index = 0
#     # 图像二值化
#     img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
#     ret, biImage = cv2.threshold(img_gray, 28+int(pow(green_ratio,1/4)*81), 255, cv2.THRESH_BINARY)
#     print(25+int(pow(green_ratio,1/4)*80))
#     # 对图像进行连通域分割
#     label_img = label(biImage, connectivity = biImage.ndim)
#     props = regionprops(label_img)
#     print('regions number:',len(props))
#     thresh = 20000
#     # 对分割后的连通域进行处理
#     target_regions = list()
#     for prop in props:
#         if prop.area>thresh and prop.area<0.4*area:
#             # y0 x0 y1 x1
#             target_regions.append([prop.bbox[0],prop.bbox[1],prop.bbox[2],prop.bbox[3]])
    
#     # 第一种情况：xxx 第二种情况：
#     print(len(target_regions))
#     if (len(target_regions)>0 and len(props)<1000 and len(target_regions)<4) or (len(target_regions)>0 and len(target_regions)<4 and len(props)<1800 and green_ratio<0.03):
#         print('polluted region')
#         return target_regions
#     else:
#         return []


def main():
    
    time1 = time.time()
    
    # print(len(img_paths))   1940
    with open('ANA-inspection-imgpaths.data', 'rb') as path_file:
        img_paths = pickle.load(path_file)
    # print(len(polluted_regions))  1940
    with open('ANA-img-polluted.bboxes','rb') as ANA_feat_file:
        polluted_regions = pickle.load(ANA_feat_file)

    tt = 0

    for r_img in img_paths:

        img = skimage.io.imread('imgs/'+r_img)
        #     print(img.shape)   (1781, 2375, 3)



        # perform selective search scale 越小区域越多 sigma越小区域越多 
        #     img_lbl, regions = selectivesearch.selective_search(
        #         img, scale=500, sigma=1, min_size=64)
        img_lbl, regions = selectivesearch4.selective_search(
            img, scale=500, sigma=1)
        # img_lbl.shape  (1781, 2375, 4)
        # regions:{'rect': (0, 0, 576, 784), 'size': 291596, 'labels': [0.0]},like

        print("selectivesearch算法划分的区域有: {}".format(len(np.unique(img_lbl[:,:,3]))))

        candidates = set()
        width,height = img.shape[1],img.shape[0]


        for r in regions:

            # regions:{'rect': (0, 0, 576, 784), 'size': 291596, 'labels': [0.0]},like
            if r['rect'] in candidates:
                continue

            # distorted rects
            x, y, w, h = r['rect']
            # excluding regions smaller than 2000 pixels
            if r['size'] < 40000 or r['size'] > 90000:
                continue
            if float(w / h) > 1.5 or float(h / w) > 1.5:
                continue
            candidates.add(r['rect'])

        print("符合尺寸要求的candidates number：: {}".format(len(candidates)))


        candidates = __builtins__.list(candidates)

        # 重新规划candidates，坐标改变
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][1],candidates[i][0],min(candidates[i][1]+candidates[i][3],width),min(candidates[i][0]+candidates[i][2],width))
    
        flag = np.zeros((len(candidates)))
        for i in range(len(candidates)):
            for j_can in range(i+1,len(candidates)):
                coincide = get_coincide_ratio(candidates[i],candidates[j_can])
                if (coincide > 0):
    #                 print(i,j)
                    if (get_size(candidates[j_can]) > get_size(candidates[i])):
                        if (flag[j_can] == 0):
                            flag[i] = 1
                    else:
                        if (flag[i] == 0):
                            flag[j_can] = 1
        for jj in range(len(candidates)-1,-1,-1):
            if (flag[jj] == 1):
                candidates.remove(candidates[jj])
        print(" 第一次符合重叠要求的candidates number：: {}".format(len(candidates)))

        img_np = np.array(img, dtype=np.int32)

        

        candidates_boxes = get_bboxes(candidates,img_np.shape[0],img_np.shape[1])


        # 根据联通域面积计算polluted
        # polluted_regions[tt]:
        # for img in imgs:
        #     print('./imgs/'+img)
        #     img_data = cv2.imread('./imgs/'+img)
        #     img_np = np.array(img_data, dtype=np.int32)
        #     ratio = get_green_pixel_ratio(img_np[:,:,1])
        #     polluted_regions.append(get_polluted_regions(img_data,ratio))



        candidates_boxes = filter_polluted_box(candidates_boxes,polluted_regions[tt])

        candidates_boxes = filter_intersected_box(candidates_boxes)
        # print("取方型区域后符合重叠要求的candidates number：: {}".format(len(candidates_boxes)))



        candidates_boxes_mean = get_candidate_region_greenmean(img_np[:,:,1],candidates_boxes)

        # zip(xzz)可以对zz进行unzip操作
        # (2, 6, 3, 5, 10, 8, 9, 7, 0, 4, 1)
        #  (27.066308815192745, 26.965366354875282, 25.6179935515873, 25.318230938208618, 22.9874929138322, 21.947402919501133, 20.728015164399093, 18.939280399659864, 18.76734339569161, 18.638029691043084, 17.839746315192745)

        sorted_idx,sorted_mean = zip(*sorted(enumerate(candidates_boxes_mean),reverse=True,key=lambda x:x[1]))
        
        candidates_boxes_final = []


        for i in range(min(4,len(sorted_idx))):
            candidates_boxes_final.append(candidates_boxes[sorted_idx[i]])
        
        # 返回
        global img_candidates_region
        img_candidates_region.append(candidates_boxes_final)

        if len(sorted_idx)<4:
            print("得到的box数量小于4，此时的box数量是 %d" % (len(sorted_idx)))

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 9))
        ax.imshow(img)
        i = 0
        for y, x, h, w in candidates_boxes_final:
            if (i % 1)==0:
                # print(x, y, w, h)
                rect = mpatches.Rectangle(
                    (x, y), w-x, h-y, fill=False, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
            i += 1

        print(r_img)
        
        plt.show()
        global j 
        j += 1
        print("这是第 %d 副图" % j)

        # print(img_candidates_region)
        print("耗时 %d" % (time.time()-time1))

        print('\n')
        print('\n')
if __name__ == "__main__":
    img_candidates_region=[]
    j=0
    main()
