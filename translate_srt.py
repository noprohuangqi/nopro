'''
该脚本是将一个文件夹（old_path）下的所有src文件，翻译到新的文件夹（new_path）下。

需要自行提供两个文件夹的位置，注意其格式示例，如：
'C:\\Users\\32002\\Desktop\\noprocode\\maomao\\english'

执行完成后会显示‘all is done!’，请耐心等待。


原始文件夹一定要备份！！！！

from translate_srt import translate_all
import hashlib
import random
import urllib.parse
import requests
from concurrent import futures
from io import StringIO
import os

from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

old_path = 'C:\\Users\\32002\\Desktop\\noprocode\\maomao\\english'
new_path = 'C:\\Users\\32002\\Desktop\\noprocode\\maomao\\ans'
translate_all(old_path, new_path)
'''




import hashlib
import random
import urllib.parse
import requests
from concurrent import futures
from io import StringIO
import os

from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

def translate(q):

    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    url = create_url(q, url)
    r = requests.get(url)
    txt = r.json()
    if txt.get('trans_result', -1) == -1:
        print('程序已经出错，请查看报错信息：\n{}'.format(txt))
#         return '这一部分翻译错误\n'
        return '\n'
    return txt['trans_result'][0]['dst']
def create_sign(q, appid, salt, key):
    '''
    制造签名
    '''
    sign = str(appid) + str(q) + str(salt) + str(key)
    md5 = hashlib.md5()
    md5.update(sign.encode('utf-8'))
    return md5.hexdigest()


def create_url(q, url):
    '''
    根据参数构造query字典
    '''
    appid = 20180613000175851
    key = 'XKJebZQ9CW1RnOFwhZVu' 
    fro = 'auto'
    to = 'zh'
    salt = random.randint(32768, 65536)
    sign = create_sign(q, appid, salt, key)
    url = url+'?appid='+str(appid)+'&q='+urllib.parse.quote(q)+'&from='+str(fro)+'&to='+str(to)+'&salt='+str(salt)+'&sign='+str(sign)
    return url



def translate_one(file_name):
    f = open(file_name,'rb')
    f_read = f.read()
    f_read_decode=f_read.decode('utf-8')
    ans = f_read_decode.split('\n')
    temp = []
    
    lists = [str(i)+'\r' for i in range(1000) ]
    
    for i in ans:
        if i.find('-->')!=-1:
            temp.append(i)
            
        elif i in lists :
            temp.append(i)

        elif i=='\r':
            temp.append(i)
        else:
            temp.append(translate(i))
    return temp

def translate_all(old_path,new_path):
    # for root, dirs, files in os.walk(old_path):
    #     for f in files:
    #         old = os.path.join(old_path,f)
    #         new = os.path.join(old_path,f.split('.')[0]+'.'+'txt')
    #         os.rename(old,new)
    for root, dirs, files in os.walk(old_path):
        for f in files:
            if f.endswith('srt'):
                old = os.path.join(old_path,f)
                new = os.path.join(old_path,f.split('.')[0]+'.'+'txt')
                os.rename(old,new)
                file_name = os.path.join(old_path,new)
                ans = translate_one(file_name)
                with open(new_path+'\\'+f,'w',encoding='utf-8') as new_f:
                    for i in ans:
                        new_f.write(i)
                        new_f.write('\n')
                old_ = os.path.join(new_path,f)
                new_ = os.path.join(new_path,f.split('.')[0]+'.'+'srt')
                os.rename(old_,new_)

                print("{} is done!".format(f))
    # for root, dirs, files in os.walk(new_path):
    #     for f in files:
    #         old = os.path.join(old_path,f)
    #         new = os.path.join(old_path,f.split('.')[0]+'.'+'srt')
    #         os.rename(old,new)
    print()
    print('all is done!')

