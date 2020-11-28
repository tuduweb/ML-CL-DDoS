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

        TESTDATA_PATH = 'N:/dataset/media_competetions_manual-uploaded-datasets_train.tar/media_competetions_manual-uploaded-datasets_train/new_test/1606549610-682356.pkl'


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


mytest = ResultTest()
mytest.RunTestInDir('./result/1606568524/')

# fnames = ['5-1.bat', '10-3.bat' , '11-2.bat']
# new = sorted(fnames, key=lambda i: int(re.match(r'\d+-(\d+).bat', i)[1]))
# print(new)
#
# print(re.match(r'(\d+)', fnames[0]))