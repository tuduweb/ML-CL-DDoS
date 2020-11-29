# import os
# predition = [1,2,3,4,5,6,7,8,9,10]
#
# saved_filename = '%d-%d.txt' % (123456, 1)
# with open(saved_filename, 'w') as fout:
#     fout.writelines("\n".join([str(n) for n in predition]))
#     fout.close()
#
# test = "\n".join([str(n) for n in predition])
# print(test)
#
# for t in test:
#     print(bytes(t, encoding = "utf8")  )
#
# print("ok")

import os
import pickle
import pandas as pd
import numpy as np
import re
class ResultTest(object):
    def __init__(self):
        self.hello = 1
        self.result = []

        TESTDATA_PATH = '../../dataset/new_test/1606549610-682356.pkl'


        # 导入测试数据
        self.truth_data = None
        with open(TESTDATA_PATH, 'rb') as fin:
            data = pickle.load(fin)
            self.truth_data = data['Y'].tolist()
            fin.close()

        #print(type(self.truth_data))


    def RunTest(self, resultFileName):
        if os.path.exists(resultFileName) == False:
            print("Result File Cant be found!")
            return

        with open(resultFileName, 'r') as fin:
            result = fin.readlines()
            self.result = [int(n.strip('\n')) for n in result]
            #print(self.result)
            fin.close()

        if len(self.result) != len(self.truth_data):
            print("size not equal")

        correct = 0
        for i in range(len(self.result)):
            if(self.result[i] == self.truth_data[i]):
                correct+=1
            #print("%d %d" %(self.result[i], self.truth_data[i]))

        print("="*20)
        print("File " + resultFileName)
        print(correct / len(self.truth_data))

    def RunTestInDir(self, dir):
        if os.path.exists(dir) == False:
            print("Directory Cant be found!")
            return

        for root, dirs, fnames in os.walk(dir):
            new_fnames = [d for d in fnames if '.txt' in d]
            new_fnames = sorted(new_fnames, key=lambda i: int(re.match(r'\d+-(\d+).txt', i)[1]))
            for fname in new_fnames:
                #if ".txt" in fname:
                self.RunTest(os.path.join(root, fname))
                    #print(fname)


# mytest = ResultTest()
# mytest.RunTestInDir('./result-autotest/1606599656/result/')

# fnames = ['5-1.bat', '10-3.bat' , '11-2.bat']
# new = sorted(fnames, key=lambda i: int(re.match(r'\d+-(\d+).bat', i)[1]))
# print(new)
#
# print(re.match(r'(\d+)', fnames[0]))

import argparse
import globalvar as gl

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--id', '-i', help='测试ID', required=True)
parser.add_argument('--version', '-v', help='样例版本ID', default=2)
#parser.add_argument('--body', '-b', help='body 属性，必要参数', required=False)
global_args = parser.parse_args()

if __name__ == '__main__':
    id = 0
    version = 0
    try:
        id = int(global_args.id)
        version = int(global_args.version)
    except Exception as e:
        print(e)

    if version == 1:
        testPath = 'result'
    else:
        testPath = 'result-autotest'

    testPath = os.path.join(testPath, str(id))
    if os.path.exists(testPath) is False:
        print("ID cant be found")
    else:
        if version == 1:
            mytest = ResultTest()
            mytest.RunTestInDir(testPath)
        else:
            param_file = os.path.join(testPath, 'param.txt')
            param_info = ''
            if os.path.exists(param_file):
                with open(param_file, 'r') as fin:
                    param_info = fin.read()
                    print(param_info)
                    fin.close()
            testPath = os.path.join(testPath, 'result')
            mytest = ResultTest()
            mytest.RunTestInDir(testPath)
            print("@"*20)
            print(param_info)

    print('LOCAL_JUDGE END')
