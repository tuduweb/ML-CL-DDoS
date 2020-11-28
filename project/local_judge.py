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

        print(correct / len(self.truth_data))


mytest = ResultTest()
mytest.RunTest("./result/1606556752/1606556752-14.txt")