engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/user_data?charset=utf8')
data = pd.read_sql('hunyinformodel', engine)

def Jaccard(a,b): #自定义杰卡德相似系数函数，仅对0-1矩阵有效

    return 1.0*(a*b).sum() /(a+b-a*b).sum()
class Recommender():

    sim = None # 相似度矩阵

    def similarity(self, x, distance): # 计算相似度矩阵的函数

        y = np.ones((len(x), len(x)))

        for i in range(len(x)):

            for j in range(len(x)):

                y[i,j] = distance(x[i], x[j])

        return y

    def fit(self, x, distance = Jaccard): # 训练函数

        self.sim = self.similarity(x, distance)

        return self.sim

    def recommend(self, a): # 推荐函数

        return np.dot(self.sim, a) * (1-a)


data.sort_values(by=['realIP','fullURL'],ascending=[True,True],inplace=True)

realIP = data['realIP'].value_counts().index

realIP = np.sort(realIP)

fullURL = data['fullURL'].value_counts().index #

fullURL = np.sort(fullURL)

D = DataFrame([], index = realIP, columns = fullURL )

for i in range(len(data)):

    a = data.iloc[i,0] # 用户名

    b = data.iloc[i,1] # 网址

    D.loc[a,b] = 1 

D.fillna(0,inplace = True)


df = D.copy()

simpler = np.random.permutation(len(df)) 

df = df.take(simpler)# 打乱数据


train = df.iloc[:int(len(df)*0.9), :]

test = df.iloc[int(len(df)*0.9):, :]


df = df.as_matrix()

df_train = df[:int(len(df)*0.9), :]# 前90%为训练集len(df_train) = 9299

df_test = df[int(len(df)*0.9):, :]# 后10%为测试集len(df_test) = 103
df_train = df_train.T

df_test = df_test.T



r = Recommender()

sim = r.fit(df_train)# 计算物品的相似度矩阵



a = DataFrame(sim) # 保存相似度矩阵


a.index = train.columns

a.columns = train.columns

 
a.to_csv('3_1_2similarityMatrix.csv')

result = r.recommend(df_test)


 

result1 = DataFrame(result)

result1.index = test.columns

result1.columns = test.index

result1.to_csv('3_1_3recommedresult.csv')


def xietong_result(K, recomMatrix ): 

    recomMatrix.fillna(0.0,inplace=True)# 将表格中的空值用0填充

    n = range(1,K+1)

    recommends = ['xietong'+str(y) for y in n]

    currentemp = DataFrame([],index = recomMatrix.columns, columns = recommends)

    for i in range(len(recomMatrix.columns)):

        temp = recomMatrix.sort_values(by = recomMatrix.columns[i], ascending = False)

        k = 0 

        while k < K:

            currentemp.iloc[i,k] = temp.index[k]

            if temp.iloc[k,i] == 0.0:

                currentemp.iloc[i,k:K] = np.nan

                break

            k = k+1

 

    return currentemp

xietong_result = xietong_result(3, result1)


xietong_result.to_csv('3_1_4xietong_result.csv')




# 随机推荐
randata = 1 - df_test # df_test是用户浏览过的网页的矩阵形式，randata则表示是用户未浏览过的网页的矩阵形式

randmatrix = DataFrame(randata, index = test.columns,columns=test.index)#这是用户未浏览过(待推荐）的网页的表格形式

def rand_recommd(K, recomMatrix):#　

    import random # 注意：这个random是random模块，

    import numpy as np

    

    recomMatrix.fillna(0.0,inplace=True) # 此处必须先填充空值

    recommends = ['recommed'+str(y) for y in range(1,K+1)]

    currentemp = DataFrame([],index = recomMatrix.columns, columns = recommends)

    

    for i in range(len(recomMatrix.columns)): #len(res.columns)1

        curentcol = recomMatrix.columns[i]

        temp = recomMatrix[curentcol][recomMatrix[curentcol]!=0]

    #     = temp.index[random.randint(0,len(temp))]

        if len(temp) == 0:

            currentemp.iloc[i,:] = np.nan

        elif len(temp) < K:

            r = temp.index.take(np.random.permutation(len(temp))) #注意：这个random是numpy模块的下属模块

            currentemp.iloc[i,:len(r)] = r

        else:

            r = random.sample(temp.index, K)

            currentemp.iloc[i,:] =  r

    return currentemp

 

start4 = time.clock()

random_result = rand_recommd(3, randmatrix) # 调用随机推荐函数

end4 = time.clock()

print '随机为用户推荐3个未浏览过的网址耗时为' + str(end4 - start4)+'s!' # 2.1900423292s!

 

#保存的表名命名格式为“3_1_k此表功能名称”，是本小节生成的第5张表格，功能为random_result：显示随机推荐的结果

random_result.to_csv('random_result.csv')

 

random_result # 结果中出现了全空的行，这是冷启动现象


def popular_recommed(K, recomMatrix):

    recomMatrix.fillna(0.0,inplace=True)

    import numpy as np

    recommends = ['recommed'+str(y) for y in range(1,K+1)]

    currentemp = DataFrame([],index = recomMatrix.columns, columns = recommends)

 

    for i in range(len(recomMatrix.columns)):

        curentcol = recomMatrix.columns[i]

        temp = recomMatrix[curentcol][recomMatrix[curentcol]!=0]

        if len(temp) == 0:

            currentemp.iloc[i,:] = np.nan

        elif len(temp) < K:

            r = temp.index #注意：这个random是numpy模块的下属模块

            currentemp.iloc[i,:len(r)] = r

        else:

            r = temp.index[:K]

            currentemp.iloc[i,:] =  r

 

    return currentemp  

TEST = 1-df_test

test2 = DataFrame(TEST, index = test.columns, columns=test.index)

print test2.head()

print test2.shape 

 

# 确定网页浏览热度排名：

hotPopular = data['fullURL'].value_counts()

hotPopular = pd.DataFrame(hotPopular)

print hotPopular.head()

print hotPopular.shape 

 

# 按照流行度对可推荐的所有网址排序

test3 = test2.reset_index()

list_custom = list(hotPopular.index)

test3['index'] = test3['index'].astype('category')

test3['index'].cat.reorder_categories(list_custom, inplace=True)

test3.sort_values('index',inplace = True)

test3.set_index ('index', inplace = True)

print test3.head()

print test3.shape 

 

 

# 按照流行度为用户推荐3个未浏览过的网址

recomMatrix = test3

start5 = time.clock()

popular_result = popular_recommed(3, recomMatrix)

end5 = time.clock()

print '按照流行度为用户推荐3个未浏览过的网址耗时为' + str(end5 - start5)+'s!'#7.70043007471s!

 

#保存的表名命名格式为“3_1_k此表功能名称”，是本小节生成的第6张表格，功能为popular_result：显示按流行度推荐的结果

popular_result.to_csv('3_1_6popular_result.csv')

 

popular_result

