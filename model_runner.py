#%%
from ImageClassifier import ImageClassifier as IC
from ImageClassifier import ModelTrainer as MT
import numpy as np
import matplotlib.pyplot as plt
from Timer import Timer
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

def runTest(D=None, EPOCHS=10, BATCH_SIZE=128, rate=0.001, pre_ops=[],network=MT.LeNet, network_args={}):
    if D is None: D = IC()
    STR="EP-{}_BS-{}_R-{}_OPS-{}-NET-{}".format(EPOCHS, BATCH_SIZE, rate, get_string_for_array(pre_ops), network.__name__)
    STR=(os.path.join('test_results',STR))
    printTestParam(EPOCHS, BATCH_SIZE, rate, pre_ops, save_dir=STR)
    
    D.reset_data()
    D.preprocess_all(pre_ops)
    T = MT(D, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, rate=rate, network=network, network_args=network_args)
    T.train(dirname=STR, pre_ops=pre_ops)
    return D, T

#%%
from matplotlib.colors import hsv_to_rgb
import cv2

def get_rand(rng, num):
    return np.random.randint(low=rng.start, high=rng.stop, size=num)

def plot_grid_subplot(nrows, ncols, plot_func, func_args={}, col_labels=[]):
    fig,axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, min(100000, 20*(nrows / ncols))))
    for c in range(ncols):

        if (c < len(col_labels)): axs[0][c].set_title('{}'.format(col_labels[c]))
        for r in range(nrows):
            plot_func(r, c, fig, axs, **func_args)
    plt.show()  


def boost_hsv(img):
    h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)/255.0
    #hsv_mean = [np.copy(h) -127.0 for h in hsv]
#    for i in range(3): h[:,:,i] /= (np.max(h[:,:,i]))
#    h[:,:,0] /= (np.mean(h[:,:,0]) + np.std(h[:,:,0]))
#    h[:,:,1] /= (np.mean(h[:,:,1]) + np.std(h[:,:,1]))
#    h[:,:,2] /= (np.mean(h[:,:,2]) + np.std(h[:,:,2]))
#    h = h.astype(np.float32)
#    h[:,:,0] /= (np.max(h[:,:,0]))
#    h[:,:,1] /= (np.max(h[:,:,1]))
    h[:,:,2] /= (np.max(h[:,:,2]))
#    h*=255.0
#    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return hsv_to_rgb(np.clip(h, 0.0, 1.0))


def boost_eq(img):
    h = np.copy(img)
    h = cv2.cvtColor(h, cv2.COLOR_RGB2YCrCb)
    h[:,:,0] = (cv2.equalizeHist(h[:,:,0]))
#    h[:,:,1] = (cv2.equalizeHist(h[:,:,1]))
#    h[:,:,2] = (cv2.equalizeHist(h[:,:,2]))
    
    return cv2.cvtColor(h, cv2.COLOR_YCrCb2RGB)


def boost_eq_gray(img):
    h = np.copy(img)
    h = cv2.cvtColor(h, cv2.COLOR_RGB2GRAY)

    h = (cv2.equalizeHist(h))
#    h[:,:,1] = (cv2.equalizeHist(h[:,:,1]))
#    h[:,:,2] = (cv2.equalizeHist(h[:,:,2]))
    
    return h
    

def boost_eq_gray_zmean_uvar(img):
    h = np.copy(img)
    h = cv2.cvtColor(h, cv2.COLOR_RGB2GRAY)
    h = h.astype(np.float32)
    
    h /= h.mean()
    h /= h.std()
    
    return h



def plot_data(data, classes=[], n_per_class=0, signs=None):
    n_classes = len(classes)        
    im_grid = {}
    for j, k in enumerate(classes):
        images = data[k]
        im_idx = np.random.randint(low=0, high=len(images), size=n_per_class)
        pstr = "Class: {:2d} | ".format(k)
        for i in im_idx: pstr+=" {:5d}".format(i)
        print(pstr)
        im_grid[j] = [images[im_idx[i]] for i in range(n_per_class)]         

    
    func = lambda r, c, fig, axs, ims: [axs[r][c].imshow(ims[c][r]), axs[r][c].axis('off')]
    plot_grid_subplot(n_per_class, n_classes, func, {'ims': im_grid})       
    

def show_image(r, c, fig, axs, ims):
    cmap, norm = None, None
    if (len(ims[c][r].shape) == 2): 
        cmap='gray'
#        norm = Normalize()
        
    axs[r][c].imshow(ims[c][r], cmap=cmap, norm=norm) 
    axs[r][c].axis('off')


def plot_data_2(data, classes=[], n_per_class=0, signs=None):
    n_classes = len(classes)        
    n_per_class
    im_grid = {}
    for j, k in enumerate(classes):
        images = data[k]
        im_idx = np.random.randint(low=0, high=len(images), size=n_per_class)
        print("Class: {} | {}".format(j, im_idx))
#        im_grid[j] = [np.hstack((images[im_idx[i]], boost_hsv(images[im_idx[i]]))) for i in range(n_per_class)] 
        im_grid[j] = [] 
        for i in range(n_per_class): 
            im_grid[j].append(np.copy(images[im_idx[i]]))
            im_grid[j].append(boost_hsv(images[im_idx[i]]))
            im_grid[j].append(boost_eq(images[im_idx[i]]))
            im_grid[j].append(cv2.cvtColor(images[im_idx[i]], cv2.COLOR_RGB2GRAY))
            im_grid[j].append(boost_eq_gray(images[im_idx[i]]))

    
    plot_grid_subplot(len(im_grid[0]), n_classes, show_image, {'ims': im_grid})       


def translate(t=[0,0]): return np.float32([[1, 0, t[0]], [0, 1, t[1]]])

def rotate(center, angle): return cv2.getRotationMatrix2D(center, angle, 1)
    
def warpImage(im_src, mat):
    sz = im_src.shape
    im_dst = np.zeros_like(im_src)
    
#    src = np.float32([ [5,5],  [5, 25], [25, 5]])
#    dst = (np.random.rand(3,2) * 32).astype(np.float32)
#    dst = np.float32([[13,3], [10, 25], [25, 5]])
#    mat = cv2.getAffineTransform(src, dst)
    for i in range(3):
        if (mat.shape == (3,3)): 
            im_dst[:, :, i] = cv2.warpPerspective(im_src[:,:,i], mat, (sz[0], sz[1]), borderMode=cv2.BORDER_REPLICATE)
        else:
            im_dst[:, :, i] = cv2.warpAffine(im_src[:,:,i], mat, (sz[0], sz[1]), borderMode=cv2.BORDER_REPLICATE)
    return im_dst


def augmentDataSet(dataset):
    M = []
    M.append(translate(t=[5, -5]))
    M.append(translate(t=[5,  5]))
    M.append(translate(t=[-5, 5]))
    M.append(translate(t=[-5,-5]))
    M.append(translate(t=[-5, 0]))
    M.append(translate(t=[ 5, 0]))
    M.append(translate(t=[ 0,-5]))
    M.append(translate(t=[ 0, 5]))
    for i in [-30, -20, -10, 10, 20, 30]: M.append(rotate((16,16), i))
    
    hist = [len(dataset[i]) for i in dataset]    
    for v in (dataset):
        m_max = max(hist)
        ims = dataset[v]
        n = len(ims)
        delta = m_max - n
        
        im_idx = np.random.randint(0, high = n, size = (delta))
        mt_idx = np.random.randint(0, high = len(M), size = (delta))    
        
        for i, idx in enumerate(im_idx):
            img = ims[idx]
            mat = M[mt_idx[i]]
            ims.append(warpImage(img, mat))
    
    
def augmentPerspective(img, sz=22, delta=3, t_rng=range(-10, 10)):
    rand = lambda rng, size=None: np.random.randint(rng.start, rng.stop, size=size)
    pt2Rng  = lambda pt, d=5: [range(i-d, i+d) for i in pt]
    getRect = lambda c=16, sz=22: [[c - sz/2, c - sz/2], 
                                   [c + sz/2, c - sz/2],
                                   [c + sz/2, c + sz/2],
                                   [c - sz/2, c + sz/2]] 
    

    pts = np.int32(getRect(sz=22))
    dst = [[rand(pt2Rng(pt, d=delta)[0]), rand(pt2Rng(pt, d=delta)[1])] for pt in pts]
    pts, dst = np.float32(pts), np.float32(dst) + np.float32([rand(t_rng), rand(t_rng)])   
    mat = cv2.getPerspectiveTransform(pts, dst)

    dst = np.zeros_like(img)        
    if (len(img.shape) == 2): 
        return cv2.warpPerspective(img, mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE)    
    
    if (len(img.shape) == 3):     
        for i in range(img.shape[2]):
            dst[:, :, i] = cv2.warpPerspective(img[:,:,i], mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE)
        return dst    

    return img


def augmentDatasetPerspective(dataset, num_total=2100):
    
    for v in (dataset):
        ims = dataset[v]
        n = len(ims)
        delta = num_total - n
        if (delta < 0): continue

        for idx in np.random.randint(low=0, high=n, size=delta):
            ims.append(augmentPerspective(ims[idx], sz=24, delta=3, t_rng=range(-7,7)))

class DataModifier:
    
    def __init__(self, data):
        self.signs = data.signs
        self.num_classes = data.num_classes
        self.train = DataModifier.split_data_into_classes(data.train['features'], data.train['labels'], self.num_classes)
        self.valid = DataModifier.split_data_into_classes(data.valid['features'], data.valid['labels'], self.num_classes)
        self.test  = DataModifier.split_data_into_classes(data.test ['features'], data.test ['labels'], self.num_classes)
        self.dmap  = { 'train': self.train, 'test': self.test, 'valid': self.valid }
    
    def split_data_into_classes(images, labels, num_classes):
        l_data = {i:[] for i in range(num_classes)}
        for idx, label in enumerate(labels): l_data[label].append(images[idx])
        return l_data    
    

    def print_data_stats(l_data, data_info, signs):
        print("-----------------------------------------------------------------------|")
        print("| Data : {:62.62}|".format(data_info))
        print("|                                                                      |")
        print("| Class |  Num Imgs  | Label                                           |")
        print("|----------------------------------------------------------------------|")
        for key, imgs in l_data.items():
            print("|  {:>3d}  |  {:>6d}  |  {}".format(key, len(imgs), signs[key] ))
            print("|----------------------------------------------------------------------|")
            
    def get_data_dist(self, key): return [len(self.dmap[key][ar]) for ar in self.dmap[key]]
        

    def updateDataSet(self, org):
        new_ims = np.empty((0, 32, 32, 3), dtype=np.float32)
        new_lbl = np.empty((0), dtype=np.float32)
        for i in self.train:
            new_ims = np.vstack((new_ims, self.train[i]))
            new_lbl = np.append(new_lbl, np.repeat(i, len(self.train[i])))
        org.train['features'], org.train['labels'] = new_ims, new_lbl 

        new_ims = np.empty((0, 32, 32, 3), dtype=np.float32)
        new_lbl = np.empty((0), dtype=np.float32)
        for i in self.test:
            new_ims = np.vstack((new_ims, self.test[i]))
            new_lbl = np.append(new_lbl, np.repeat(i, len(self.test[i])))
        org.test['features'], org.test['labels'] = new_ims, new_lbl

        org.reset_data()
        for i in range(3): 
            org.shuffle_training_data()
            org.shuffle_test_data()
        
        org.print_data_info()
        
        return new_lbl, new_ims
        

#%%

def load():
    org  = IC()
    data = DataModifier(org)
    #augmentDataSet(data.train)
    #augmentDataSet(data.test)
    augmentDatasetPerspective(data.train, num_total=3000)
    #augmentDataSet(data.test)
    l,im = data.updateDataSet(org)
    return org, data


def train():

    kEPOCHS, kBATCH_SIZE, kRATE, kPRE_OPS, = 'EPOCHS', 'BATCH_SIZE', 'rate', 'pre_ops'
    kNETWORK, kNETWORK_ARGS = 'network', 'network_args'
    
    tests  =[
    #            {kEPOCHS: 25, kBATCH_SIZE: 64, kRATE: 0.001, kNETWORK: MT.LeNetWithDropOut, kNETWORK_ARGS: {'dropouts': {0:0.5, 1:0.5}}},
    #            {kEPOCHS: 25, kBATCH_SIZE:  64, kRATE: 0.002, kNETWORK: MT.LeNetWithDropOut, kNETWORK_ARGS: {'dropouts': {2:0.5}}},
    #            {kEPOCHS: 25, kBATCH_SIZE: 128, kRATE: 0.002, kNETWORK: MT.LeNetWithDropOut, kNETWORK_ARGS: {'dropouts': {2:0.5, 3:0.5}}},
    #            {kEPOCHS: 25, kBATCH_SIZE: 128, kRATE: 0.002, kNETWORK: MT.LeNetWithDropOut, kNETWORK_ARGS: {'dropouts': {2:0.5}}},
                {kEPOCHS:  15, kBATCH_SIZE: 64, kRATE: 0.002, kNETWORK: MT.LeNetWithDropOut, kNETWORK_ARGS: {'dropouts': {}}}     
            ]
    
    
    for p in tests:
        p[kPRE_OPS] = [IC.ZShift]
        Data, Trainer = runTest(D=org, **p)


#%%
#
#org, data = Timer.run(load)
#
#%%
#
#Timer.run(train)
#
#%%

    
#pre_ops = [IC.NormalizeImage, IC.ZeroMeanImage, IC.UnitVarImage]
#pre_ops=[IC.UnitVarImage]
#pre_ops=[IC.ZeroMeanImage, IC.UnitVarImage]


##%%
#
#plot_data(data.train, classes=get_rand(range(0,42), 5), n_per_class=30, signs=data.signs)
#
#
#
#
#
##%%
#
#count=0
#
#
#
#src1 = np.float32([ [ 6,6],  [6, 26], [26, 6]])
#dst1 = np.float32([[13,5], [10, 25], [25, 5]])
#dst2 = np.float32([[10,3], [10, 27], [22, 8]])
#
#
#
#def translate(t=[0,0]): return np.float32([[1, 0, t[0]], [0, 1, t[1]]])
#def rotate(center, angle): return cv2.getRotationMatrix2D(center, angle, 1)
#    
#    
#    
#def warpImage(im_src, mat):
#    sz = im_src.shape
#    im_dst = np.zeros_like(im_src)
#    
##    src = np.float32([ [5,5],  [5, 25], [25, 5]])
##    dst = (np.random.rand(3,2) * 32).astype(np.float32)
##    dst = np.float32([[13,3], [10, 25], [25, 5]])
##    mat = cv2.getAffineTransform(src, dst)
#    for i in range(3):
#        if (mat.shape == (3,3)): 
#            im_dst[:, :, i] = cv2.warpPerspective(im_src[:,:,i], mat, (sz[0], sz[1]), borderMode=cv2.BORDER_REPLICATE)
#        else:
#            im_dst[:, :, i] = cv2.warpAffine(im_src[:,:,i], mat, (sz[0], sz[1]), borderMode=cv2.BORDER_REPLICATE)
#    return im_dst
#
#def plotImage(im_src, im_dst):    
#    im_comp = np.hstack((im_src, im_dst))    
#    plt.imshow(im_comp)
#    plt.show()
#    
#im = data.train[40][112]
#im_center = (im.shape[0]/2, im.shape[1]/2)
##plotImage(im, warpImage(im, translate(t=[5, -5])))
##plotImage(im, warpImage(im, translate(t=[5,  5])))
##plotImage(im, warpImage(im, translate(t=[-5, 5])))
##plotImage(im, warpImage(im, translate(t=[-5,-5])))
##plotImage(im, warpImage(im, translate(t=[-5, 0])))
##plotImage(im, warpImage(im, translate(t=[ 5, 0])))
##plotImage(im, warpImage(im, translate(t=[ 0,-5])))
##plotImage(im, warpImage(im, translate(t=[ 0, 5])))
#
#for i in [-60, -30, 30, 60]: plotImage(im, warpImage(im, rotate(im_center, i)))
#
##%%
#
#hist = [len(data.train[i]) for i in range(len(data.train))]
#plt.plot(hist)
#
##%%        
#        
##    new_ims = np.int(max(0, np.floor(m_max/n)-1)*n)
##    new_total = np.int(n + new_ims)
##    new_delta = m_max - new_total
##    new_hist.append(new_total)
##    print("delta = {:5}   |   num new = {:5}   |   new total = {:5}   |   new_del = {:5}".format(delta, new_ims, new_total, new_delta))    
#
##%%
#
#
##%%
#
#
##%%
#
#
#
#
##%%
#
##EPOCHS=25
##BATCH_SIZE=64
##rate=0.004
#
##tests = [ 
##            [25, 64, 0.002],
##            [30, 64, 0.002],
##            [35, 64, 0.002],
##            [1, 64, 0.001]
##         ]
##
##for p in tests:
##    pre_ops = [IC.Norm, IC.ZMean]
##    Data, Trainer = runTest(EPOCHS=p[0], BATCH_SIZE=p[1], rate=p[2], pre_ops=pre_ops, network=MT.LeNetWithDropOut, network_args={'dropouts': {0: 0.5}})
##%%
#
#
#
#
##%%
#
#import numpy as np
#import matplotlib.pyplot as plt
#
#a = IC()
#
#def split_data_into_classes(images, labels):
#    l_data = {i:[] for i in range(43)}
#    for idx, label in enumerate(labels): l_data[label].append(images[idx])
#    return l_data    
#    
#
#def print_data_stats(l_data, data_info, signs):
#    print("-----------------------------------------------------------------------|")
#    print("| Data : {:62.62}|".format(data_info))
#    print("|                                                                      |")
#    print("| Class |  Num Imgs  | Label                                           |")
#    print("|----------------------------------------------------------------------|")
#    for key, imgs in l_data.items():
#        print("|  {:>3d}  |  {:>6d}  |  {}".format(key, len(imgs), signs[key] ))
#        print("|----------------------------------------------------------------------|")
#
#
#train = split_data_into_classes(a.train['features'], a.train['labels'])
#valid = split_data_into_classes(a.valid['features'], a.valid['labels'])
#test  = split_data_into_classes( a.test['features'],  a.test['labels'])
#
##print_data_stats(train, "Traing Set Only: Total Images = {}".format(len(a.X_train)))  
##print("\n\n\n\n\n\n\n\n\n\n")  
##print_data_stats(valid, "Validation Set Only: Total Images = {}".format(len(a.X_valid)))    
##print("\n\n\n\n\n\n\n\n\n\n")  
##print_data_stats(test , "Testing Set Only: Total Images = {}".format(len(a.X_test)))    
#
#
#
##%%
#
#
##cols = ['{}'.format(col) for col in range(1, 4)]
##rows = ['Row {}'.format(row) for row in ['A', 'B', 'C', 'D']]
##
##fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
##
##for ax, col in zip(axes[0], cols):
##    ax.set_title(col)
#
#
##%%
#plot_data(train, classes=range(6), n_per_class=5, signs=a.signs)
#
##%%
##EPOCHS=10
##BATCH_SIZE=128
##rate=0.001
##pre_ops = [IC.NormalizeImage, IC.ZeroMeanImage]
##
##Data, Trainer = runTest(pre_ops=pre_ops)
##
##---------------------------------
##      | Train  | Test   | Valid  |
##---------------------------------
## (#)  |  34799 |  12630 |   4410 |
## (%)  |   1.00 |   0.36 |   0.13 |
##---------------------------------
##Number Classes  : 43
##Image Dimensions: (32, 32, 3)
##
##Training...
##
##EPOCH 1 ...
##Validation Accuracy = 0.782
##
##EPOCH 2 ...
##Validation Accuracy = 0.860
##
##EPOCH 3 ...
##Validation Accuracy = 0.866
##
##EPOCH 4 ...
##Validation Accuracy = 0.898
##
##EPOCH 5 ...
##Validation Accuracy = 0.909
##
##EPOCH 6 ...
##Validation Accuracy = 0.908
##
##EPOCH 7 ...
##Validation Accuracy = 0.901
##
##EPOCH 8 ...
##Validation Accuracy = 0.923
##
##EPOCH 9 ...
##Validation Accuracy = 0.904
##
##EPOCH 10 ...
##Validation Accuracy = 0.908
##
##Model saved
#
#
##%%
##------------------------------------------
## EPOCHS          : 10
## BATCH_SIZE      : 128
## Pre-process Ops : [NormalizeImage-ZeroMeanImage-UnitVarImage]
## Save Dir        : EP-10_BS-128_R-0.001_OPS-[NormalizeImage-ZeroMeanImage-UnitVarImage]
##
##---------------------------------
##      | Train  | Test   | Valid  |
##---------------------------------
## (#)  |  34799 |  12630 |   4410 |
## (%)  |   1.00 |   0.36 |   0.13 |
##---------------------------------
##Number Classes  : 43
##Image Dimensions: (32, 32, 3)
##
##Training...
##
##EPOCH  1 ...
##Validation Accuracy = 0.834
##
##EPOCH  2 ...
##Validation Accuracy = 0.879
##
##EPOCH  3 ...
##Validation Accuracy = 0.895
##
##EPOCH  4 ...
##Validation Accuracy = 0.900
##
##EPOCH  5 ...
##Validation Accuracy = 0.887
##
##EPOCH  6 ...
##Validation Accuracy = 0.915
##
##EPOCH  7 ...
##Validation Accuracy = 0.910
##
##EPOCH  8 ...
##Validation Accuracy = 0.901
##
##EPOCH  9 ...
##Validation Accuracy = 0.897
##
##EPOCH 10 ...
##Validation Accuracy = 0.918
##
##Model saved : EP-10_BS-128_R-0.001_OPS-[NormalizeImage-ZeroMeanImage-UnitVarImage]_003/LeNet
#
##%%
##------------------------------------------
## EPOCHS          : 10
## BATCH_SIZE      : 128
## Pre-process Ops : [ZeroMeanImage]
## Save Dir        : EP-10_BS-128_R-0.001_OPS-[ZeroMeanImage]
##
##---------------------------------
##      | Train  | Test   | Valid  |
##---------------------------------
## (#)  |  34799 |  12630 |   4410 |
## (%)  |   1.00 |   0.36 |   0.13 |
##---------------------------------
##Number Classes  : 43
##Image Dimensions: (32, 32, 3)
##
##Training...
##
##EPOCH  1 ...
##Validation Accuracy = 0.721
##
##EPOCH  2 ...
##Validation Accuracy = 0.828
##
##EPOCH  3 ...
##Validation Accuracy = 0.857
##
##EPOCH  4 ...
##Validation Accuracy = 0.892
##
##EPOCH  5 ...
##Validation Accuracy = 0.868
##
##EPOCH  6 ...
##Validation Accuracy = 0.899
##
##EPOCH  7 ...
##Validation Accuracy = 0.889
##
##EPOCH  8 ...
##Validation Accuracy = 0.901
##
##EPOCH  9 ...
##Validation Accuracy = 0.910
##
##EPOCH 10 ...
##Validation Accuracy = 0.912
##
##Model saved : EP-10_BS-128_R-0.001_OPS-[ZeroMeanImage]_000/LeNet
#
##%%
##------------------------------------------
## EPOCHS          : 10
## BATCH_SIZE      : 128
## Pre-process Ops : []
## Save Dir        : EP-10_BS-128_R-0.001_OPS-[]
##
##---------------------------------
##      | Train  | Test   | Valid  |
##---------------------------------
## (#)  |  34799 |  12630 |   4410 |
## (%)  |   1.00 |   0.36 |   0.13 |
##---------------------------------
##Number Classes  : 43
##Image Dimensions: (32, 32, 3)
##
##Training...
##
##EPOCH  1 ...
##Validation Accuracy = 0.616
##
##EPOCH  2 ...
##Validation Accuracy = 0.755
##
##EPOCH  3 ...
##Validation Accuracy = 0.817
##
##EPOCH  4 ...
##Validation Accuracy = 0.828
##
##EPOCH  5 ...
##Validation Accuracy = 0.845
##
##EPOCH  6 ...
##Validation Accuracy = 0.850
##
##EPOCH  7 ...
##Validation Accuracy = 0.872
##
##EPOCH  8 ...
##Validation Accuracy = 0.873
##
##EPOCH  9 ...
##Validation Accuracy = 0.870
##
##EPOCH 10 ...
##Validation Accuracy = 0.877
##
##Model saved : EP-10_BS-128_R-0.001_OPS-[]_000/LeNet
#
##%%
#
#
##Image Dimensions: (32, 32, 3)
##
##
##Class: 0 | [103  45  54  61 177]
##Class: 1 | [1423  622 1425  811 1318]
##Class: 2 | [1336 1193  726   19 1463]
##Class: 3 | [ 355  800 1234  422  860]
##Class: 4 | [1410  176 1131 1616 1030]
#
##%%
#
#
#
#from matplotlib.colors import hsv_to_rgb
#import cv2
#
#ims = [data.train[1][811], data.train[3][422], data.train[4][1131] ]
#hsv = [cv2.cvtColor(im, cv2.COLOR_RGB2HSV) for im in ims]
#
#hsv_boost = [np.copy(h) for h in hsv]
#for i in hsv_boost: i[:,:,2]=i[:,:,2]*1
#
#ims2 = [hsv,hsv_boost]
#
#plot_grid_subplot(2, 3, lambda r,c,fig,axs: axs[r][c].imshow(((cv2.cvtColor(ims2[r][c],cv2.COLOR_HSV2RGB)))))
#
##%%
#
#hsv_norm = [np.copy(h)/255.0 for h in hsv]
##hsv_mean = [np.copy(h) -127.0 for h in hsv]
#
#hsv_uvar = [np.copy(h) for h in hsv_norm]
#for h in hsv_uvar:
#    h[:,:,0] /= np.max(h[:,:,0])
#    h[:,:,1] /= np.max(h[:,:,1])
#    h[:,:,2] /= np.max(h[:,:,2])
#    
#p = hsv_uvar
#plot_grid_subplot(1, 3, lambda r,c,fig,axs: axs[c].imshow( hsv_to_rgb(p[c]) ))
#plot_grid_subplot(1, 3, lambda r,c,fig,axs: axs[c].imshow( ims[c]))
#
##plot_grid_subplot(2, 3, lambda r,c,fig,axs: axs[r][c].imshow(((cv2.cvtColor(ims2[r][c],cv2.COLOR_HSV2RGB)))))
#
#
#
#
