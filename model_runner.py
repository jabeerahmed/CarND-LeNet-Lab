#%%
from ImageClassifier import ImageClassifier as IC
from ImageClassifier import ModelTrainer as MT
import os

def get_string_for_array(ar):
    varstr = '['
    for i in ar: varstr+=(i.__name__ + "-") 

    if (varstr.endswith('-')): varstr = varstr[0:-1]    
    varstr += ']'
    return varstr
    

def printTestParam(EPOCHS, BATCH_SIZE, rate, pre_ops, save_dir='.'):
    print("------------------------------------------")
    print(" EPOCHS          : " + str(EPOCHS))
    print(" BATCH_SIZE      : " + str(BATCH_SIZE))
    print(" Pre-process Ops : " + get_string_for_array(pre_ops))
    print(" Save Dir        : " + save_dir)
    print("")    

def runTest(EPOCHS=10, BATCH_SIZE=128, rate=0.001, pre_ops=[]):
    STR="EP-{}_BS-{}_R-{}_OPS-{}".format(EPOCHS, BATCH_SIZE, rate, get_string_for_array(pre_ops))
    STR=(os.path.join('test_results',STR))
    printTestParam(EPOCHS, BATCH_SIZE, rate, pre_ops, save_dir=STR)

    D = IC()
    D.preprocess_all(pre_ops)
    T = MT(D, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, rate=rate)
    T.train(dirname=STR, pre_ops=pre_ops)
    return D, T



#%%

EPOCHS=25
BATCH_SIZE=64
rate=0.002
#pre_ops = [IC.NormalizeImage, IC.ZeroMeanImage, IC.UnitVarImage]
#pre_ops=[IC.UnitVarImage]
#pre_ops=[IC.ZeroMeanImage, IC.UnitVarImage]
pre_ops = [IC.NormalizeImage, IC.ZeroMeanImage]
Data, Trainer = runTest(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, rate=rate, pre_ops=pre_ops)


#%%

#EPOCHS=10
#BATCH_SIZE=128
#rate=0.001
#pre_ops = [IC.NormalizeImage, IC.ZeroMeanImage]
#
#Data, Trainer = runTest(pre_ops=pre_ops)
#
#---------------------------------
#      | Train  | Test   | Valid  |
#---------------------------------
# (#)  |  34799 |  12630 |   4410 |
# (%)  |   1.00 |   0.36 |   0.13 |
#---------------------------------
#Number Classes  : 43
#Image Dimensions: (32, 32, 3)
#
#Training...
#
#EPOCH 1 ...
#Validation Accuracy = 0.782
#
#EPOCH 2 ...
#Validation Accuracy = 0.860
#
#EPOCH 3 ...
#Validation Accuracy = 0.866
#
#EPOCH 4 ...
#Validation Accuracy = 0.898
#
#EPOCH 5 ...
#Validation Accuracy = 0.909
#
#EPOCH 6 ...
#Validation Accuracy = 0.908
#
#EPOCH 7 ...
#Validation Accuracy = 0.901
#
#EPOCH 8 ...
#Validation Accuracy = 0.923
#
#EPOCH 9 ...
#Validation Accuracy = 0.904
#
#EPOCH 10 ...
#Validation Accuracy = 0.908
#
#Model saved


#%%
#------------------------------------------
# EPOCHS          : 10
# BATCH_SIZE      : 128
# Pre-process Ops : [NormalizeImage-ZeroMeanImage-UnitVarImage]
# Save Dir        : EP-10_BS-128_R-0.001_OPS-[NormalizeImage-ZeroMeanImage-UnitVarImage]
#
#---------------------------------
#      | Train  | Test   | Valid  |
#---------------------------------
# (#)  |  34799 |  12630 |   4410 |
# (%)  |   1.00 |   0.36 |   0.13 |
#---------------------------------
#Number Classes  : 43
#Image Dimensions: (32, 32, 3)
#
#Training...
#
#EPOCH  1 ...
#Validation Accuracy = 0.834
#
#EPOCH  2 ...
#Validation Accuracy = 0.879
#
#EPOCH  3 ...
#Validation Accuracy = 0.895
#
#EPOCH  4 ...
#Validation Accuracy = 0.900
#
#EPOCH  5 ...
#Validation Accuracy = 0.887
#
#EPOCH  6 ...
#Validation Accuracy = 0.915
#
#EPOCH  7 ...
#Validation Accuracy = 0.910
#
#EPOCH  8 ...
#Validation Accuracy = 0.901
#
#EPOCH  9 ...
#Validation Accuracy = 0.897
#
#EPOCH 10 ...
#Validation Accuracy = 0.918
#
#Model saved : EP-10_BS-128_R-0.001_OPS-[NormalizeImage-ZeroMeanImage-UnitVarImage]_003/LeNet

#%%
#------------------------------------------
# EPOCHS          : 10
# BATCH_SIZE      : 128
# Pre-process Ops : [ZeroMeanImage]
# Save Dir        : EP-10_BS-128_R-0.001_OPS-[ZeroMeanImage]
#
#---------------------------------
#      | Train  | Test   | Valid  |
#---------------------------------
# (#)  |  34799 |  12630 |   4410 |
# (%)  |   1.00 |   0.36 |   0.13 |
#---------------------------------
#Number Classes  : 43
#Image Dimensions: (32, 32, 3)
#
#Training...
#
#EPOCH  1 ...
#Validation Accuracy = 0.721
#
#EPOCH  2 ...
#Validation Accuracy = 0.828
#
#EPOCH  3 ...
#Validation Accuracy = 0.857
#
#EPOCH  4 ...
#Validation Accuracy = 0.892
#
#EPOCH  5 ...
#Validation Accuracy = 0.868
#
#EPOCH  6 ...
#Validation Accuracy = 0.899
#
#EPOCH  7 ...
#Validation Accuracy = 0.889
#
#EPOCH  8 ...
#Validation Accuracy = 0.901
#
#EPOCH  9 ...
#Validation Accuracy = 0.910
#
#EPOCH 10 ...
#Validation Accuracy = 0.912
#
#Model saved : EP-10_BS-128_R-0.001_OPS-[ZeroMeanImage]_000/LeNet

#%%
#------------------------------------------
# EPOCHS          : 10
# BATCH_SIZE      : 128
# Pre-process Ops : []
# Save Dir        : EP-10_BS-128_R-0.001_OPS-[]
#
#---------------------------------
#      | Train  | Test   | Valid  |
#---------------------------------
# (#)  |  34799 |  12630 |   4410 |
# (%)  |   1.00 |   0.36 |   0.13 |
#---------------------------------
#Number Classes  : 43
#Image Dimensions: (32, 32, 3)
#
#Training...
#
#EPOCH  1 ...
#Validation Accuracy = 0.616
#
#EPOCH  2 ...
#Validation Accuracy = 0.755
#
#EPOCH  3 ...
#Validation Accuracy = 0.817
#
#EPOCH  4 ...
#Validation Accuracy = 0.828
#
#EPOCH  5 ...
#Validation Accuracy = 0.845
#
#EPOCH  6 ...
#Validation Accuracy = 0.850
#
#EPOCH  7 ...
#Validation Accuracy = 0.872
#
#EPOCH  8 ...
#Validation Accuracy = 0.873
#
#EPOCH  9 ...
#Validation Accuracy = 0.870
#
#EPOCH 10 ...
#Validation Accuracy = 0.877
#
#Model saved : EP-10_BS-128_R-0.001_OPS-[]_000/LeNet