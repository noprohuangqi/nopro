import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats.stats as st
from numpy import *



def computer_mi(data,types,project,stand,num):
    list_1 = [i for i in range(len(data)) if i%4==0]
    list_2 = [i+1 for i in list_1]
    list_3 = [i+2 for i in list_1]
    list_4 = [i+3 for i in list_1]
    s1,s2,s3,s4,s5,s6  = stand
    evals = ['优秀','良好','及格','不及格']
    lists = [list_1,list_2,list_3,list_4]
    

    if types =='总体':
        for i in evals:
            if i=='优秀':
                sum1 = np.array(data.loc[list_1+list_2][project]<=s1).sum()
                sum2 = np.array(data.loc[list_3+list_4][project]<=s2).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))

            elif i=='良好':
                sum1 = (np.array(data.loc[list_1+list_2][project]>s1) & np.array(data.loc[list_1+list_2][project]<=s3)).sum()
                sum2 = (np.array(data.loc[list_3+list_4][project]>s2) & np.array(data.loc[list_3+list_4][project]<=s4)).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
            elif i=='及格':
                sum1 = (np.array(data.loc[list_1+list_2][project]>s3) & np.array(data.loc[list_1+list_2][project]<=s5)).sum()
                sum2 = (np.array(data.loc[list_3+list_4][project]>s4) & np.array(data.loc[list_3+list_4][project]<=s6)).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
            else:
                sum1 = np.array(data.loc[list_1+list_2][project]>s5).sum()
                sum2 = np.array(data.loc[list_3+list_4][project]>s6).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
    else:
        for i in evals:
            if i=='优秀':
                sum1 = np.array(data[project]<=s1).sum()
                sum2 = np.array(data[project]<=s2).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))
            elif i=='良好':
                sum1 = (np.array(data[project]>s1) & np.array(data[project]<=s3)).sum()
                sum2 = (np.array(data[project]>s2) & np.array(data[project]<=s4)).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))
            elif i=='及格':
                sum1 = (np.array(data[project]>s3) & np.array(data[project]<=s5)).sum()
                sum2 = (np.array(data[project]>s4) & np.array(data[project]<=s6)).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))

            else:
                sum1 = np.array(data[project]>s5).sum()
                sum2 = np.array(data[project]>s6).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))






def computer_other(data,types,project,stand,num):
    list_1 = [i for i in range(len(data)) if i%4==0]
    list_2 = [i+1 for i in list_1]
    list_3 = [i+2 for i in list_1]
    list_4 = [i+3 for i in list_1]
    s1,s2,s3,s4,s5,s6  = stand
    evals = ['优秀','良好','及格','不及格']
    lists = [list_1,list_2,list_3,list_4]
    

    if types =='总体':
        for i in evals:
            if i=='优秀':
                sum1 = np.array(data.loc[list_1+list_2][project]>=s1).sum()
                sum2 = np.array(data.loc[list_3+list_4][project]>=s2).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))

            elif i=='良好':
                sum1 = (np.array(data.loc[list_1+list_2][project]<s1) & np.array(data.loc[list_1+list_2][project]>=s3)).sum()
                sum2 = (np.array(data.loc[list_3+list_4][project]<s2) & np.array(data.loc[list_3+list_4][project]>=s4)).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
            elif i=='及格':
                sum1 = (np.array(data.loc[list_1+list_2][project]<s3) & np.array(data.loc[list_1+list_2][project]>=s5)).sum()
                sum2 = (np.array(data.loc[list_3+list_4][project]<s4) & np.array(data.loc[list_3+list_4][project]>=s6)).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
            else:
                sum1 = np.array(data.loc[list_1+list_2][project]<s5).sum()
                sum2 = np.array(data.loc[list_3+list_4][project]<s6).sum()
                print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1+sum2,(sum1+sum2)/len(data)))
    else:
        for i in evals:
            if i=='优秀':
                sum1 = np.array(data[project]>=s1).sum()
                sum2 = np.array(data[project]>=s2).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))
            elif i=='良好':
                sum1 = (np.array(data[project]<s1) & np.array(data[project]>=s3)).sum()
                sum2 = (np.array(data[project]<s2) & np.array(data[project]>=s4)).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))
            
            elif i=='及格':
                sum1 = (np.array(data[project]<s3) & np.array(data[project]>=s5)).sum()
                sum2 = (np.array(data[project]<s4) & np.array(data[project]>=s6)).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))
            else:
                sum1 = np.array(data[project]<s5).sum()
                sum2 = np.array(data[project]<s6).sum()
                if num==1 or num==2:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum1,(sum1)/len(data)))
                else:
                    print('{}项目{}{}的人数是:{}，占比是{}'.format(project,types,i,sum2,(sum2)/len(data)))






def computer_project(data,project,stand):
    list_1 = [i for i in range(len(data)) if i%4==0]
    list_2 = [i+1 for i in list_1]
    list_3 = [i+2 for i in list_1]
    list_4 = [i+3 for i in list_1]
    s1,s2,s3,s4,s5,s6  = stand
    evals = ['优秀','良好','及格','不及格']
    lists = [list_1,list_2,list_3,list_4]
    if project.endswith('米'):
        print()
        print('{}项目的总体描述是:'.format(project))
        print(data[project].describe())
        print()
        computer_mi(data,'总体',project,stand,num=1)

        for i in range(4):
            print()
            print('{}项目的第{}次测试的描述是:'.format(project,i+1))
            print(data.iloc[lists[i]][project].describe())
            print()
            types = '第'+str(i+1)+'次测试'
            data_temp = data.iloc[lists[i]]
            computer_mi(data_temp,types,project,stand,i+1)
    else:
        print()
        print('{}项目的总体描述是:'.format(project))
        print(data[project].describe())
        print()
        computer_other(data,'总体',project,stand,num=1)

        for i in range(4):
            print()
            print('{}项目的第{}次测试的描述是:'.format(project,i+1))
            print(data.iloc[lists[i]][project].describe())
            print()
            types = '第'+str(i+1)+'次测试'
            data_temp = data.iloc[lists[i]]
            computer_other(data_temp,types,project,stand,i+1)


def bmi_n(data,n,s1,s2,s3):
    length = len(data)
    print("第{}次共有{}条记录".format(n,length))
    print()
    print('bmi项目第{}次的总体描述是:'.format(n))
    print(data['BMI'].describe())
    print()
    bmi_normal = (array(data['BMI']<=s2) & array(data['BMI']>=s1)).sum()
    bmi_di = (array(data['BMI']<s1)).sum()
    bmi_gao = (array(data['BMI']<=s3) & array(data['BMI']>s2)).sum()
    bmi_hengao = (array(data['BMI']>s3)).sum()
    print("第{}次bmi正常的人数是{},占比{}".format(n,bmi_normal,bmi_normal/length))
    print("第{}次bmi较低的人数是{},占比{}".format(n,bmi_di,bmi_di/length))
    print("第{}次bmi超重的人数是{},占比{}".format(n,bmi_gao,bmi_gao/length))
    print("第{}次bmi肥胖的人数是{},占比{}".format(n,bmi_hengao,bmi_hengao/length))
    print()

def computer_bmi(data,stand):
    list_1 = [i for i in range(len(data)) if i%4==0]
    list_2 = [i+1 for i in list_1]
    list_3 = [i+2 for i in list_1]
    list_4 = [i+3 for i in list_1]
    length = len(data)
    s1,s2,s3 = stand
    print()
    print('bmi项目的总体描述是:')
    print(data['BMI'].describe())
    print()
    bmi_normal = (array(data['BMI']<=23.9) & array(data['BMI']>=17.9)).sum()
    bmi_di = (array(data['BMI']<17.9)).sum()
    bmi_gao = (array(data['BMI']<=28) & array(data['BMI']>23.9)).sum()
    bmi_hengao = (array(data['BMI']>28)).sum()
    print("bmi正常的人数是{},占比{}".format(bmi_normal,bmi_normal/length))
    print("bmi较低的人数是{},占比{}".format(bmi_di,bmi_di/length))
    print("bmi超重的人数是{},占比{}".format(bmi_gao,bmi_gao/length))
    print("bmi肥胖的人数是{},占比{}".format(bmi_hengao,bmi_hengao/length))
    print()
    
    bmi_n(data.loc[list_1],1,s1,s2,s3)
    bmi_n(data.loc[list_2],2,s1,s2,s3)
    bmi_n(data.loc[list_3],3,s1,s2,s3)
    bmi_n(data.loc[list_4],4,s1,s2,s3)




def computer_all(data,sex):
    
    male_stands = {'肺活量':[4800,4900,4300,4400,3100,3200],
    '立定跳远':[2.63,2.65,2.48,2.5,2.08,2.1],
    '坐位体前屈':[21.3,21.5,17.7,18.2,3.7,4.2],
    '50米':[6.9,6.8,7.1,7.0,9.1,9.0],
    '1000米':[207,205,222,220,272,270],
    '引体向上':[17,18,15,16,10,11]}

    female_stands = {'肺活量':[3300,3350,3000,3050,2000,2050],
    '立定跳远':[1.95,1.96,1.81,1.82,1.51,1.52],
    '坐位体前屈':[22.2,22.4,19,19.5,6,6.5],
    '50米':[7.7,7.6,8.3,8.2,10.3,10.2],
    '800米':[210,208,224,222,274,272],
    '仰卧起坐':[52,53,46,47,26,27]}
    female_bmi = [17.2,23.9,27.9]
    male_bmi = [17.9,23.9,27.9]

    if sex == 'male':
        computer_bmi(data,male_bmi)
        for i in male_stands.keys():
            computer_project(data,i,male_stands[i])
            # bmi
            
    else:
        computer_bmi(data,female_bmi)
        for i in female_stands.keys():
            
            computer_project(data,i,female_stands[i])
            


        

       













