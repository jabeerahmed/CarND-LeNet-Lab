from ImageClassifier import ImageClassifier as IC
from ImageClassifier import ModelTrainer as MT
from Timer import Timer
from model_runner import *
from model_runner import DataModifier, Utils
from matplotlib import pyplot
import numpy as np
import ConvNet as CNN
import os
import pickle
import pandas as pd
from pandas import Categorical as cat

def loadData(dirpath='traffic-signs-data'):
    org  = IC(data_dir=dirpath)
    data = DataModifier(org)
    return org, data

def saveData(org, dirpath):
    dirpath = Utils.find_an_empty_dir(dirpath)
    print("Saving data to: " + dirpath)
    if (os.path.exists(dirpath) == False): os.makedirs(dirpath)
    with open(os.path.join(dirpath, 'train.p'), "wb") as fp:
        pickle.dump(org.train, fp)
    with open(os.path.join(dirpath, 'test.p'), "wb" ) as fp:
        pickle.dump(org.test, fp)
    with open(os.path.join(dirpath, 'valid.p'), "wb") as fp:
        pickle.dump(org.valid, fp)

def augmentData(org=None, data=None, save_dir=None, num_total=1000):
#     augmentDataSet(data.train)
#     augmentDataSet(data.test)
    n_train,n_test,n_valid = num_total, np.int(num_total*0.3), np.int(num_total*0.1)
    augmentDatasetPerspective(data.train, num_total=n_train, sz=26, delta=3, t_rng=range(-3, 3), perspTrans=True)
    augmentDatasetPerspective(data.test,  num_total=n_test , sz=26, delta=3, t_rng=range(-3, 3), perspTrans=True)
    augmentDatasetPerspective(data.valid, num_total=n_valid, sz=26, delta=3, t_rng=range(-3, 3), perspTrans=True)
    DataModifier.updateDataSet(org, data.train, org.train)
    DataModifier.updateDataSet(org, data.test,  org.test)
    DataModifier.updateDataSet(org, data.valid, org.valid)    
    if (save_dir is not None): saveData(org, save_dir)
    return org, data

def basicNorm(img):
    shape = img.shape
    dst = np.float32(img)
    if (len(shape) == 2):
            p  = dst
            p -= p.min()
            p /= p.max()        
    else:
        for i in range(shape[2]): 
            p  = dst[:, :, i]
            p -= p.min()
            p /= p.max()

    return dst
            
def visualize(data, setname='train'):
    
    dataset = data.dmap[setname]
    signnames = data.signs
    col_width = max(len(name) for name in signnames)
    class_counts = data.get_data_dist(setname)
    
    for c in range(len(signnames)):
        X_train = dataset[c]
        c_count = len(X_train)
        c_index = 0
        
        print("Class %i: %-*s  %s samples" % (c, col_width, signnames[c], str(c_count)))
        fig = pyplot.figure(figsize = (6, 1))
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
        random_indices = random.sample(range(c_index, c_index + c_count), 10)
        for i in range(10):
            axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
            axis.imshow(basicNorm(X_train[random_indices[i]]))
                
        pyplot.show()
        print("--------------------------------------------------------------------------------------\n")

    pyplot.bar( np.arange( 43 ), class_counts, align='center' )
    pyplot.xlabel('Class')
    pyplot.ylabel('Number of training examples')
    pyplot.xlim([-1, 43])
    pyplot.show()


def pandasData(org):
    n_train, n_test, n_valid = len(org.train['labels']), len(org.test['labels']), len(org.valid['labels'])
    train = pd.DataFrame({'label': cat(org.train['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.train['labels']), categories=org.signs), 'image': list(org.train['features']), 'type': cat(['train'] * n_train, categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_train}, copy=False)
    valid = pd.DataFrame({'label': cat(org.valid['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.valid['labels']), categories=org.signs), 'image': list(org.valid['features']), 'type': cat(['valid'] * n_valid, categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_valid}, copy=False)
    test  = pd.DataFrame({'label': cat(org.test ['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.test ['labels']), categories=org.signs), 'image': list(org.test ['features']), 'type': cat(['test' ] * n_test , categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_test }, copy=False)

    train_test_valid_set = pd.Series([train, test, valid], index=['train', 'test', 'valid'], copy=False)
    full_set = pd.concat([train, test, valid], ignore_index=True, copy=False)    
    class_set = pd.Series([full_set[full_set.label == i] for i in range(org.num_classes)], copy=False)
    return full_set, class_set, train_test_valid_set

#%% 
#==============================================================================
#   Load Data 
#==============================================================================

data_dir = os.path.join('aug1K_sz26_d3_rng3_persp', '000')
org, data = Timer.run(loadData, args={'dirpath': data_dir})
full_set, class_set, ttv_set = pandasData(org)

#%%
#==============================================================================
#   Pre-Process 
#==============================================================================

def preprocess(org=None, data=None, pre_ops=[]):
    if (len(pre_ops) > 0): org.preprocess_all(pre_ops)
    return org, data


def Center_1to1(src, i, dst):
    dst[i] = (src[i] - 128.0)/128.0

kEPOCHS, kBATCH_SIZE, kRATE, kNETWORK, kNETWORK_ARGS = 'EPOCHS', 'BATCH_SIZE', 'RATE', 'NETWORK', 'NETWORK_ARGS'

def train(D=None, EPOCHS=10, BATCH_SIZE=128, RATE=0.001, NETWORK=MT.LeNet, NETWORK_ARGS={}):
    if D is None: raise Exception('D in None')
    T = MT(D, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, rate=RATE, network=NETWORK, network_args=NETWORK_ARGS)
    T.train()
    return D, T

# pre_ops = [Center_1to1]
pre_ops = [IC.Norm, IC.ZMean]
org, data = Timer.run(preprocess,  args={'org': org, 'data': data, 'pre_ops':pre_ops})

# , 'dropouts':{0: 0.9, 1:0.6}
args = {'D': org, 'EPOCHS': 25, 'BATCH_SIZE': 64, 'RATE': 0.0001, 'NETWORK': IC.LeNet, 'NETWORK_ARGS': {'sigma':0.001}} 

_, _ = Timer.run(train, args=args)