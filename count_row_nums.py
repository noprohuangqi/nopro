
# -*- coding: utf-8 -*-


import os
import time
import sys


def count_file_row_number(file):
    count = 0
    thefile = open(file, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count



def count_files_or_file_row_number(files):
    abspath = os.getcwd()
    curr_path = os.path.join(abspath,files)
    is_dir = 0
    row_nums = 0
    
    
    if os.path.isdir(curr_path):
        print("您目前查询的是文件夹{}".format(curr_path))
        is_dir = 1
        
    if (is_dir == 0):
        row_nums = count_file_row_number(curr_path)
        print("该文件共有{}行")
        
    if (is_dir == 1):
        for root, dirs, files in os.walk(curr_path):
            for file in files:
                if (file.endswith("py")):
                    complete_path = os.path.join(curr_path,file)
                    row_nums = row_nums + count_file_row_number(complete_path)
                
        print("该文件夹下共有{}行".format(row_nums))
        
    
if __name__ == '__main__':
    start = time.time()
    files = sys.argv[1]
    count_files_or_file_row_number(files)
    end = time.time()
    print("耗时{}".format(start-end))