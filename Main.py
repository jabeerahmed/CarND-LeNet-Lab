#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:45:00 2017

@author: jabeerah
"""

%load_ext autoreload
%autoreload 1
%aimport model_runner
%aimport ConvNet
%aimport Timer
%matplotlib inline
##%%

from Timer import Timer
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import tensorflow as tf 
from tensorflow.contrib.layers import flatten
import csv

import sys
import os
import shutil
from sklearn.utils import shuffle
import json
import cv2


#==============================================================================
# Utils
#==============================================================================

FID = None
def setFileID(new_file_id):
    global FID
    FID = new_file_id

def print(msg=""):
    global FID    
    if (type(msg) != str): msg = str(msg)
    pstr = msg if msg.endswith("\n") else msg+"\n"
    
    if (FID is not None): 
        FID.write(pstr)
        FID.flush()

    sys.stdout.write(msg if msg.endswith("\n") else msg+"\n")
    sys.stdout.flush()
    
class Utils:    

    def get_string_for_array(ar):
        varstr = '['
        for i in ar: varstr+=(i.__name__ + "-") 
    
        if (varstr.endswith('-')): varstr = varstr[0:-1]    
        varstr += ']'
        return varstr
        
    def printTestParam(EPOCHS, BATCH_SIZE, rate, save_dir='.'):
        print("------------------------------------------\n")
        print("-EPOCHS          : " + str(EPOCHS) + "\n")
        print("-BATCH_SIZE      : " + str(BATCH_SIZE) + "\n")
        print("-Save Dir        : " + save_dir + "\n")
        print("------------------------------------------\n")
        print("")    


    def find_an_empty_dir(dirname):
        v = 0
        while (True):
            tmp_name = os.path.join(dirname, "%03d"%(v))
            if (os.path.isdir(tmp_name) == False): return tmp_name
            v+=1
            
    def progressString(pcent, space=50):
        bar = '[{:' + str(space) + '}]'
        msg = "{} {:1.2f}".format(bar.format("#"* int(space * pcent)), pcent)
        return msg
    
    
    def progressPrint( epoch, valid, train=None, loss=None):
        msg = "{:3} | V {}".format(epoch, valid)
        if (train is not None): msg += " | T {}".format(train)
        if (loss  is not None): msg += " | L {}".format(loss)
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()       

##%% 
#==============================================================================
# IMAGE CONTAINER
#==============================================================================


class ImageLoader:
    
    def Load_Image_Data(data_dir):
        training_file =  os.path.join(data_dir, 'train.p')
        validation_file= os.path.join(data_dir, 'valid.p')
        testing_file =   os.path.join(data_dir, 'test.p')
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        return train, test, valid


    def Read_Sign_Names(signs_file_name):
        names =[]
        if None is not signs_file_name: 
            with open(signs_file_name) as f:
                a = csv.reader(f);
                for i in a: names.append(i[1])
            
            if (len(names) > 0): names.pop(0)

        return names
    

    def __init__(self, data_dir='traffic-signs-data', label_csv_file='signnames.csv'):
        self.data_descriptor = {}
        self.data_dir=data_dir
        self.label_csv_file=label_csv_file
        self.data_descriptor.update({'data_dir': self.data_dir, 'label_file': label_csv_file})
                
        # Load data set
        train, test, valid = ImageLoader.Load_Image_Data(self.data_dir)          
        # Load Image Signs  
        self.signs = ImageLoader.Read_Sign_Names(self.label_csv_file)
        self.num_classes=len(self.signs)
                
        self.train = ImageLoader.split_data_into_classes(train['features'], train['labels'], self.num_classes)
        self.valid = ImageLoader.split_data_into_classes(valid['features'], valid['labels'], self.num_classes)
        self.test  = ImageLoader.split_data_into_classes( test['features'],  test['labels'], self.num_classes)
        self.dmap  = { 'train': self.train, 'test': self.test, 'valid': self.valid }
            
    
    def getParamDescriptor(self, desc={}): 
        desc.update({'ImageLoaderParams': self.data_descriptor})
        return desc
        
    
    def split_data_into_classes(images, labels, num_classes):
        l_data = {i:[] for i in range(num_classes)}
        for idx, label in enumerate(labels): l_data[label].append(images[idx])
        return l_data    
    
    
    def print_data_stats(self, data_type='train'):
        l_data = self.dmap[data_type]
        signs = self.signs
        print("-----------------------------------------------------------------------|")
        print("| Data : {:62.62}|".format(data_type))
        print("|                                                                      |")
        print("| Class |  Num Imgs  | Label                                           |")
        print("|----------------------------------------------------------------------|")
        for key, imgs in l_data.items():
            print("|  {:>3d}  |  {:>6d}  |  {}".format(key, len(imgs), signs[key] ))
            print("|----------------------------------------------------------------------|")
            

    def get_data_dist(self, key): return [len(self.dmap[key][ar]) for ar in self.dmap[key]]
            

    def augmentDatasetPerspective(self, dataType, num_total=2100, sz=22, delta=3, t_rng=range(-10, 10), perspTrans=False):    
        dataset = self.dmap[dataType]    
        self.data_descriptor['aug_'+dataType] = {'total_images': num_total, 'sz': sz, 'delta': delta, 't_rng':str(t_rng), 'perspTrans':perspTrans} 

        for v in (dataset):
            ims = dataset[v]
            n = len(ims)
            num_aug = num_total - n
            if (num_aug < 0): continue
    
            for idx in np.random.randint(low=0, high=n, size=num_aug):
                ims.append(ImageLoader.AugmentPerspective(ims[idx], sz=sz, delta=delta, t_rng=t_rng, perspTrans=perspTrans))
        return dataset


    def AugmentPerspective(img, sz=22, delta=3, t_rng=range(-10, 10), perspTrans=True):
        rand = lambda rng, size=None: np.random.randint(rng.start, rng.stop, size=size)
        pt2Rng  = lambda pt, d=delta: [range(i-d, i+d) for i in pt]
        getRect = lambda c=16, sz=sz: [[c - sz/2, c - sz/2], 
                                       [c + sz/2, c - sz/2],
                                       [c + sz/2, c + sz/2],
                                       [c - sz/2, c + sz/2]] 
        
    
        pts = np.int32(getRect(sz=sz))
        dst = [[rand(pt2Rng(pt, d=delta)[0]), rand(pt2Rng(pt, d=delta)[1])] for pt in pts]
        pts, dst = np.float32(pts), np.float32(dst) + np.float32([rand(t_rng), rand(t_rng)])   
        mat = cv2.getPerspectiveTransform(pts, dst) if perspTrans else cv2.getAffineTransform(pts[:3,:], dst[:3,:])
    
        dst = np.zeros_like(img)        
        if (len(img.shape) == 2): 
            if (perspTrans): 
                return cv2.warpPerspective(img, mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE) 
            else: 
                return cv2.warpAffine(img, mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE) 
        if (len(img.shape) == 3):     
            for i in range(img.shape[2]):
                if (perspTrans): 
                    dst[:, :, i] = cv2.warpPerspective(img[:,:,i], mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE)
                else: 
                    dst[:, :, i] = cv2.warpAffine(img[:,:,i], mat, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REPLICATE)
            return dst    
        return img    
        
    
    def plot_grid_subplot(nrows, ncols, plot_func, func_args={}, col_labels=[]):
        fig,axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, min(100000, 20*(nrows / ncols))))
        for c in range(ncols):
    
            if (c < len(col_labels)): axs[0][c].set_title('{}'.format(col_labels[c]))
            for r in range(nrows):
                plot_func(r, c, fig, axs, **func_args)
        plt.show()  
    

    def plot_data(self, dataType='train', classes=None, n_per_class=5):
        if (classes is None): classes = random.sample(range(0,self.num_classes), 5)    
        data = self.dmap[dataType]
        n_classes = len(classes)        
        im_grid = {}
        for j, k in enumerate(classes):
            images = data[k]
            im_idx = np.random.randint(low=0, high=len(images), size=n_per_class)
            pstr = "Class: {:2d} | ".format(k)
            for i in im_idx: pstr+=" {:5d}".format(i)
            print(pstr)
            im_grid[j] = [images[im_idx[i]] for i in range(n_per_class)]         
    
        
        func = lambda r, c, fig, axs, ims: [axs[r][c].imshow(basicNorm(ims[c][r])), axs[r][c].axis('off')]
        ImageLoader.plot_grid_subplot(n_per_class, n_classes, func, {'ims': im_grid})       
        
        
    def visualize(self, setname='train'):        
        data = self        
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
    
    
    def plot_histogram(self, setname='train'):
        class_counts = self.get_data_dist(setname)
        pyplot.bar( np.arange( self.num_classes ), class_counts, align='center' )
        pyplot.xlabel('Class')
        pyplot.ylabel('Number of {} examples'.format(setname))
        pyplot.xlim([-1, self.num_classes])
        pyplot.show()

        
    
    def CreateDataSet(self, pcent_train=1.0, pcent_test=1.0, pcent_valid=1.0):
        
        train, test, valid = {'features': [], 'labels': []}, {'features': [], 'labels': []}, {'features': [], 'labels': []}
        
        dst =   [train,       test,       valid]
        src =   [self.train,  self.test,  self.valid]
        pcent = [pcent_train, pcent_test, pcent_valid]

        for j in range(3):        
            dataDst, dataSrc = dst[j], src[j]
            for i in range(self.num_classes):
                dataDst['features'] += dataSrc[i]
                dataDst['labels']   += [i]* len(dataSrc[i])
            
            n_sample = len(dataDst['features'])
            pcent_tk = pcent[j]            
            idx = range(0, n_sample) if (pcent_tk == 1.0) else random.sample(range(0, n_sample), int(n_sample * pcent_tk))
                        
            dataDst['features'] = np.copy(np.take(dataDst['features'], idx, axis=0))
            dataDst['labels'  ] = np.copy(np.take(dataDst['labels'],   idx, axis=0))
            
        return train, test, valid

        
#%%

#==============================================================================
# Networks and Layers
#==============================================================================

def CalcOutputSize(in_x, k_x, s_x, valid=True):
    m = (k_x - 1) if valid else 0
    return math.ceil(float(in_x - m) / float(s_x))

def GetOutputSize(inSize, kSize, stride, paddingValid=True):
    w = CalcOutputSize(inSize[1], kSize[1], stride[1], paddingValid)
    h = CalcOutputSize(inSize[2], kSize[2], stride[2], paddingValid)
    c = kSize[0]
    return (w, h, c)


class Network(object):
    
    def __init__(self, name = "Default"):
        self.name = name
        self.descriptor = {}      
        self.train_feed, self.eval_feed = {}, {}
        self.layers = []
        
    def getParamDescriptor(self, desc={}): 
        desc.update({'Network_'+self.name : self.descriptor})
        return desc
    
    def getSaveDir(self): return None
    def getLogFilePath(self): return None

    def fully_connected(x, num_outs, mu=0.0, sigma=1.0, name=None, dtype=np.float32):
        num_ins = x.shape.as_list()[-1]        
        Ws = tf.Variable(tf.truncated_normal((num_ins, num_outs), mean=mu, stddev=sigma, name=name, dtype=dtype))
        Bs = tf.Variable(np.zeros((num_outs,)), dtype=dtype)
        return tf.matmul(x, Ws) + Bs
        
    def conv2d(x, fh, fw, fd, strides=[1, 1, 1, 1], padding='VALID', mu=0.0, sigma=1.0, name=None, dtype=np.float32):
        (_, xh, xw, xc) = x.shape.as_list()
        F_W = tf.Variable(tf.truncated_normal(shape=(fh, fw, xc, fd), mean=mu, stddev=sigma, name=None), dtype=dtype)
        F_b = tf.Variable(np.zeros((fd, )), dtype=dtype)
        return tf.nn.conv2d(x, F_W, strides, padding) + F_b
    
    def addDropOut(layer, keep_prob, train_feed, eval_feed, name=None):
        if ( 0 < keep_prob < 1 ):
            kp = tf.placeholder(tf.float32, name=name)
            train_feed.update({kp: keep_prob})
            eval_feed.update({kp: 1.0})
            return tf.nn.dropout(layer, keep_prob=kp)
        return layer
    
    def add(self, layer):
        lastLayer = self.layers[-1] if len(self.layers) > 0 else None
        if (layer is not None and layer != lastLayer): self.layers.append(layer)
        return layer

    def strTensor(b): return "Tensor {:20s} shape={}".format(b.name, str(b.get_shape()))


class LeNetWithDropOut(Network):
    
    def __init__(self, x, num_outputs, mu=0.0, sigma=0.1, dropouts={}):        
        Network.__init__(self, "LeNetWithDropOut" )
        train_feed, eval_feed = {}, {}
        
        tensor = None
        
        self.add(x)
        
        Layer = 0       # Convolutional. Input = 32x32x3. Output = 28x28x6.
        tensor = self.add(Network.conv2d(x, 5, 5, 6, strides = [1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma))
        tensor = self.add(tf.nn.relu(tensor))
        if (Layer in dropouts): tensor = self.add(Network.addDropOut(tensor, dropouts[Layer], train_feed, eval_feed))
        tensor = self.add(tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
            
        Layer = 1       # Convolutional. Input = 32x32x3. Output = 28x28x6
        tensor = self.add(Network.conv2d(tensor, 5, 5, 16, strides=[1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma))
        tensor = self.add(tf.nn.relu(tensor))
        if (Layer in dropouts): tensor = self.add(Network.addDropOut(tensor, dropouts[Layer], train_feed, eval_feed))
        tensor = self.add(tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
    
        Layer = 2       # Flatten. Input = 5x5x16. Output = 400.
        tensor = self.add(flatten(tensor))
        if (Layer in dropouts): tensor = self.add(Network.addDropOut(tensor, dropouts[Layer], train_feed, eval_feed))
    
        Layer = 3       # Layer 3: Fully Connected. Input = 400. Output = 120.
        tensor = self.add(Network.fully_connected(tensor, 120, mu, sigma))
        tensor = self.add(tf.nn.relu(tensor))
        if (Layer in dropouts): tensor = self.add(Network.addDropOut(tensor, dropouts[Layer], train_feed, eval_feed))
    
        Layer = 4       # Fully Connected. Input = 120. Output = 84
        tensor = self.add(Network.fully_connected(tensor, 84, mu, sigma))
        tensor = self.add(tf.nn.relu(tensor))
        if (Layer in dropouts): tensor = self.add(Network.addDropOut(tensor, dropouts[Layer], train_feed, eval_feed))
    
        Layer = 5     # Fully Connected. Input = 84. Output = 43.
        tensor = self.add(Network.fully_connected(tensor, num_outputs, mu, sigma))

        self.train_feed, self.eval_feed = train_feed, eval_feed
    
        self.updateDescriptor(args={'mu': mu, 'sigma': sigma, 'dropouts': dropouts})


    def updateDescriptor(self, args={}):
        self.descriptor['LeNetParams'] = args
        tstring = {}
        for i, t in enumerate(self.layers): tstring["{:2}".format(i)] = Network.strTensor(t)
        self.descriptor['LeNetArch'] = tstring                               

#%%


#==============================================================================
# 
#==============================================================================
 
class Trainer(object):

    
    def __init__(self, train, test, valid, signs, num_classes, network, args={}, rate=0.001):
        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test,  self.y_test  =  test['features'],  test['labels']
        self.image_shape = self.X_train[0].squeeze().shape
        self.signs = signs
        self.num_classes = num_classes

        self.x  = tf.placeholder(tf.float32, (None,) + self.image_shape)
        self.y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(self.y, self.num_classes)
        
        # init network
        self.network = network(self.x, self.num_classes, **args)    
        logits = self.network.layers[-1]
        self.train_feed, self.eval_feed = self.network.train_feed, self.network.eval_feed
        # Init Train Pipeline
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)        

        # Init Eval Pipeline
        self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

        self.one_hot_y, self.logits, self.rate = one_hot_y, logits, rate
        self.print_data_info()

        self.descriptor = {}

        
    def print_data_info(self):
        num_data = {'train': len(self.y_train), 'test': len(self.y_test), 'valid': len(self.y_valid)}
        ratio_data={k:num_data[k]/num_data['train'] for k in ['train', 'test', 'valid']}
        print('---------------------------------')
        print('      '+'| Train  | Test   | Valid  |')
        print('---------------------------------')
        print(' (#)  '+"| %6d | %6d | %6d |"% (num_data['train'], num_data['test'], num_data['valid']))
        print(' (%)  '+"| %6.2f | %6.2f | %6.2f |"% (ratio_data['train'], ratio_data['test'], ratio_data['valid']))
        print('---------------------------------')
        print('Number Classes  : ' + str(self.num_classes))
        print('Image Dimensions: ' + str(self.image_shape))
        print()


    def getParamDescriptor(self, desc={}):
        desc['trainer'] = self.descriptor
        return desc

    def train(self, sess, dirname='results', EPOCHS=10, BATCH_SIZE=256, stat_freq=30, rfunc=None):

        self.descriptor.update({'learning_rate': self.rate, 'EPOCHS': EPOCHS, 'BATCH_SIZE':BATCH_SIZE})
        
        dirname = Utils.find_an_empty_dir(dirname)
        if (os.path.isdir(dirname) == False): os.makedirs(dirname)
        outpath = os.path.join(dirname, "SavedModel")

        run_incomplete = True
        stats = []

        print("TrainFeed = " + str(self.train_feed))
        print("EvalFeed  = " + str(self.eval_feed))
        
        msg = lambda pcent, sp=30: Utils.progressString(pcent, space=sp)
        bar = lambda ep, valid, t=None, l=None: Utils.progressPrint(ep, valid, train=t, loss=l)

        try:            
            print("Training...")
            print()
            
            sess.run(tf.global_variables_initializer())
            num_examples = len(self.X_train)
            validation_accuracy = 0
            
            for i in range(EPOCHS):
                tot_acc, tot_loss, tot_n = 0,0,0
                self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
                for n, offset in enumerate(range(0, num_examples, BATCH_SIZE)):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = self.X_train[offset:end], self.y_train[offset:end]
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    feed_dict.update(self.train_feed)
            
                    if (n % stat_freq == (stat_freq-1)):
                        train_acc, loss_r = self.calcAccAndLoss(sess, batch_x, batch_y)
                        bar(i, msg(validation_accuracy,sp=20), t=msg(train_acc,sp=20), l=msg(100*loss_r,sp=30))
                        stats.append([validation_accuracy, n, train_acc, i, loss_r])
                        tot_acc, tot_loss, tot_n = tot_acc+train_acc, tot_loss+loss_r, tot_n + 1
                        if (rfunc is not None): rfunc(stats)
            
            
                    sess.run(self.training_operation, feed_dict=feed_dict)
            
                self.X_test, self.y_test = shuffle(self.X_test, self.y_test)
                validation_accuracy = self.evaluate(sess, self.X_test, self.y_test)
                bar(i, msg(validation_accuracy,sp=20), t=msg(tot_acc/tot_n,sp=20), l=msg(100*(tot_loss/tot_n),sp=30))
                print()
            
                
            self.saver.save(sess, outpath)
            print("Model saved : " + outpath)
            run_incomplete = False
        finally:
            if (run_incomplete): shutil.rmtree(dirname)

        return stats


    def evaluate(self, sess, x_data, y_data, BATCH_SIZE=128):
        
        num_examples = len(x_data)
        BATCH_SIZE = min(BATCH_SIZE, num_examples)
        total_accuracy = 0

        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={self.x: batch_x, self.y: batch_y}
            feed_dict.update(self.eval_feed)
            accuracy = sess.run(self.accuracy_operation, feed_dict=feed_dict)
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
                
    
    def calcLoss(self, sess, x_data, y_data, BATCH_SIZE=128):
        num_examples = len(x_data)
        BATCH_SIZE = min(BATCH_SIZE, num_examples)
        total_loss = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={self.x: batch_x, self.y: batch_y}
            feed_dict.update(self.eval_feed)
            loss = sess.run(self.loss_operation, feed_dict=feed_dict)
            total_loss += (loss * len(batch_x))
        return loss / num_examples


    def calcAccAndLoss(self, sess, X_data, y_data):
        return self.evaluate(sess, X_data, y_data), self.calcLoss(sess, X_data, y_data)


#%%
def loadData(dirpath='traffic-signs-data'):
    data  = ImageLoader(data_dir=dirpath)
    return data

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

def augmentData(data, num_total=1000, save_dir=None, sz=26, delta=3, t_rng=range(-3, 3), perspTrans=True):
    n_train,n_test,n_valid = num_total, np.int(num_total*0.3), np.int(num_total*0.1)
    data.augmentDatasetPerspective('train', num_total=n_train, sz=sz, delta=delta, t_rng=t_rng, perspTrans=perspTrans)
    data.augmentDatasetPerspective('test',  num_total=n_test , sz=sz, delta=delta, t_rng=t_rng, perspTrans=perspTrans)
    data.augmentDatasetPerspective('valid', num_total=n_valid, sz=sz, delta=delta, t_rng=t_rng, perspTrans=perspTrans)

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
            

#from pandas import Categorical as cat
#
#def pandasData(org, isOrginialSet=False):
#    n_train, n_test, n_valid = len(org.train['labels']), len(org.test['labels']), len(org.valid['labels'])
#    train = pd.DataFrame({'label': cat(org.train['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.train['labels']), categories=org.signs), 'image': list(org.train['features']), 'type': cat(['train'] * n_train, categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_train}, copy=False)
#    valid = pd.DataFrame({'label': cat(org.valid['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.valid['labels']), categories=org.signs), 'image': list(org.valid['features']), 'type': cat(['valid'] * n_valid, categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_valid}, copy=False)
#    test  = pd.DataFrame({'label': cat(org.test ['labels'], categories=range(org.num_classes)), 'sign': cat(np.take(org.signs, org.test ['labels']), categories=org.signs), 'image': list(org.test ['features']), 'type': cat(['test' ] * n_test , categories=['train', 'test', 'valid']), 'isAugmented': [False] * n_test }, copy=False)
#
#    train_test_valid_set = pd.Series([train, test, valid], index=['train', 'test', 'valid'], copy=False)
#    full_set = pd.concat([train, test, valid], ignore_index=True, copy=False)    
#    class_set = pd.Series([full_set[full_set.label == i] for i in range(org.num_classes)], copy=False)
#    return full_set, class_set, train_test_valid_set
    

#%% LOAD THE DATA

data_dir = os.path.join("aug1K_sz26_d3_rng3_persp", "000")
training_file = os.path.join(data_dir, "train.p")
validation_file = os.path.join(data_dir, "valid.p")
testing_file = os.path.join(data_dir, "test.p")





#org, data = Timer.run(loadData)
#full_set, class_set, ttv_set = pandasData(org)
#
#org, data = augmentData(org, data, num_total=1000)


#%%

print('ConvNet main loop')

#org_data = ImageLoader()
#augmentData(data=org_data, num_total=1000, save_dir=None, sz=26, delta=3, t_rng=range(-3, 3), perspTrans=True)

network_args = {'mu': 0.0, 'sigma':0.1, 'dropouts':{}}

train, test, valid = org_data.CreateDataSet()
model = Trainer(train, test, valid, org_data.signs, org_data.num_classes, LeNetWithDropOut, args=network_args, rate=0.001)

global FID


try :
    print("Start")
    # Set file logging path
    logpath = Utils.find_an_empty_dir(os.path.join('results', model.network.name))
    os.makedirs(logpath, mode=511, exist_ok=False)
    log_file = os.path.join(logpath, 'log.txt')
    FID = open(log_file, "w")
    
    # Print descriptions
    model.print_data_info()
    desc = org_data.getParamDescriptor()
    desc = model.network.getParamDescriptor(desc)
    print(json.dumps(desc, indent=4, sort_keys=True))
    
    with tf.Session() as sess:        
        model.train(sess)
    #    x  = tf.placeholder(tf.float32, (None, 32, 32, 3))
    #    num_classes = 43
    #    mu, sigma = 0.1, 0.1
    #    
    #    net = LeNetWithDropOut(x, num_classes,dropouts={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5})
    #    net.layers

finally:
    try:
        FID.close()
    finally:
        FID = None
    pass        
