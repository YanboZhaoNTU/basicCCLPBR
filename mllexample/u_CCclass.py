from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from CCclass import *
datasnames = ["Yeast"]
num = 0
result = np.array([])
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
w_save = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
save = np.array([])
n = 0
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))
# (1629, 103) (1629, 14) (788, 103) (788, 14)
CCt = CCclass()
for h in range(3):
    X_tr_list = []
    Y_tr_list = []
    X_te_list = []
    Y_te_list = []
    for i in range(1629):
        j = random.randint(0, 1628)

        X_tr_list.append(X_train[j])
        Y_tr_list.append(Y_train[j])
    for i in range(1629):
        j = random.randint(0, 1628)
        X_te_list.append(X_train[j])
        Y_te_list.append(Y_train[j])

    X_tr = np.array(X_tr_list)

    Y_tr = np.array(Y_tr_list)
    X_te = np.array(X_te_list)
    Y_te = np.array(Y_te_list)

    CCt.train(X_tr, Y_tr)
    result = CCt.CC_test(X_te,Y_te)
    print(result.shape)
    # 清空基学习器的训练分类器
    BR_cfl_train = []
    CCt.CCCLF_clear()


    if n == 0:
        save = Y_te
        n = n + 1
    else:
        save = np.hstack([save,Y_te])
print(result.shape)
CCt.CC_train_BR_train(result, save)
###############################################
####训练阶段结束了
###############################################
CCt.CCCLF_clear()
CCt.BRCLF_test_clear()
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
test_clf_num = 0
test_result = np.array([])
for i in range(788):
    j = random.randint(0, 787)

    X_tr_list.append(X_test[j])
    Y_tr_list.append(Y_test[j])


for h in range(3):

    star = test_clf_num
    end = test_clf_num + 14
    test_clf_num = test_clf_num + 14
    X_tr = np.array(X_tr_list)
    test_result = CCt.test_BRC_test(X_tr,star,end)

real_label = np.array(Y_tr_list)
real = np.hstack([real_label,real_label])
real = np.hstack([real,real_label])
final_result = CCt.BR_test_BRC_test(test_result)
correct_matrix = (final_result == real_label)
num_correct = np.sum(correct_matrix)
eva = evaluate(final_result,real)
print(eva)
accuracy = num_correct / (788*14)


