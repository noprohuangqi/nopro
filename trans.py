import urllib.request

import urllib.parse

import json

content=input("please input:")

url='http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

data={}


data['i']=content
data['from']='AUTO'
data['to']='AUTO'
data['smartresult']='dict'
data['client']= 'fanyideskweb'
data['salt']='1522912490352'
data['sign']= '6f255bb92d8c11ba5514aabb5f921a40'
data['doctype']= 'json'
data['version']= '2.1'
data['keyfrom']= 'fanyi.web'
data['action']= 'FY_BY_REALTIME'
data['typoResult']= 'false'

data=urllib.parse.urlencode(data).encode('utf-8')

response=urllib.request.urlopen(url,data)

html=response.read().decode('utf-8')


target=json.loads(html)



print((target['translateResult'][0][0]['tgt']))
