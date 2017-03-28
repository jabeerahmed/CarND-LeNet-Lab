#!/usr/bin/env python3
import os
import sys
import csv
import pickle
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class ImageClassifier:
    
    print("ImageClassifier Loaded!!!")    
    def __init__(self, data_dir='.', label_csv_file='signnames.csv'):
        self.data_dir=data_dir
        self.label_csv_file=label_csv_file
        # Load data set
        train, test, valid = self.load_image_data()          
        self.X_train, self.y_train = np.array(train['features'], dtype=np.float32), train['labels']
        self.X_valid, self.y_valid = np.array(valid['features'], dtype=np.float32), valid['labels']
        self.X_test,  self.y_test  = np.array( test['features'], dtype=np.float32),  test['labels']
 
#        self.X_train, self.y_train = np.array(train['features']), train['labels']
#        self.X_valid, self.y_valid = np.array(valid['features']), valid['labels']
#        self.X_test,  self.y_test  = np.array( test['features']),  test['labels']
       
        # Load Image Signs  
        self.signs = self.read_signnames()
        self.num_classes=len(self.signs)
        self.image_shape = self.X_train[0].squeeze().shape
        self.train, self.test, self.valid = (train, test, valid) 
        # print info
        self.print_data_info()

        self.__data = {'train': self.train, 'test': self.test, 'valid': self.valid}
        self.__data_types = ('train', 'test', 'valid')
        

    def load_image_data(self):
        return ImageClassifier.Load_Image_Data(self.data_dir)


    def Load_Image_Data(data_dir):
        training_file = 'traffic-signs-data/train.p'
        validation_file='traffic-signs-data/valid.p'
        testing_file = 'traffic-signs-data/test.p'
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        return train, test, valid


    def read_signnames(self):
        return ImageClassifier.Read_Sign_Names(self.label_csv_file)
            
    
    def Read_Sign_Names(signs_file_name):
        names =[]
        if None is not signs_file_name: 
            with open(signs_file_name) as f:
                a = csv.reader(f);
                for i in a: names.append(i[1])
            
            if (len(names) > 0): names.pop(0)

        return names
                
    
    def print_data_info(self):
        num_data = {'train': len(self.train['labels']), 'test': len(self.test['labels']), 'valid': len(self.valid['labels'])}
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


    def plot_random_train_image(self):
        index = random.randint(0, len(self.X_train))
        self.plot_train_image(index)

    
    def plot_train_image(self, index):
        image = self.X_train[index].squeeze()
        label = "Index = " + str(index) + " | Label = " + str(self.y_train[index]) + " | " + str(self.signs[self.y_train[index]])
        ImageClassifier.Plot_Image(image, label)        
    
    
    def Plot_Image(image, label='N/A'):
        plt.figure(figsize=(1,1))
        plt.imshow(image)
        print(label)
            

    def zero_mean_all_data(self):
        ops = [ImageClassifier.ZeroMeanImage];
        self.preprocess(ops, self.X_train)
        self.preprocess(ops, self.X_test)
        self.preprocess(ops, self.X_valid)
        
        
    def preprocess(self, operations, data, dst_buf=None):
        if (dst_buf is None): dst_buf = data
        for i in range(len(data)):
            for op in operations:
                op(data, i, dst=dst_buf)

    def preprocess_all(self, operations):
        self.preprocess(operations, self.X_train)
        self.preprocess(operations, self.X_test )
        self.preprocess(operations, self.X_valid)
        
                
    def ZeroMeanImage(src, i, dst):  
#        if (dst is None): dst = src
        dst[i] = src[i] - np.mean(src[i])

    def NormalizeImage(src, i, dst): 
#        if (dst is None): dst = src
        dst[i] = src[i]/255.0

    def UnitVarImage(src, i, dst): 
#        if (dst is None): dst = src
        dst[i] = src[i]/(np.std(src[i]))
    
    
    def shuffle_training_data(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        

class ModelTrainer:
    
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
    
    def LeNet(x, num_outputs, mu=0.0, sigma=0.1):    
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1 = ModelTrainer.conv2d(x, 5, 5, 6, strides = [1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)
        # TODO: Activation.
        relu1 = tf.nn.relu(conv1)
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2 = ModelTrainer.conv2d(pool1, 5, 5, 16, strides=[1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)        
        # TODO: Activation.
        relu2 = tf.nn.relu(conv2)
        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # TODO: Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(pool2)   
        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = ModelTrainer.fully_connected(fc0, 120, mu, sigma)
        # TODO: Activation.
        fc1_relu = tf.nn.relu(fc1)
        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2 = ModelTrainer.fully_connected(fc1_relu, 84, mu, sigma)    
        # TODO: Activation.
        fc2_relu = tf.nn.relu(fc2)
        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        return ModelTrainer.fully_connected(fc2_relu, num_outputs, mu, sigma)   
    

    def __init__(self, data, network=LeNet, EPOCHS=10, BATCH_SIZE=128, rate=0.001):
        # Hyper-Parameters
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.rate = rate
        self.initialModel(data, network)
        

    def initialModel(self, data, network):
        self.initialize_data_container(data)
        self.setModelNet(network)
        self.initTrainingPipeline()
        self.initEvalPipeline()  
    
        
    def initialize_data_container(self, data):
        self.data = data
        self.data.shuffle_training_data()
        x = tf.placeholder(tf.float32, (None,) + self.data.image_shape)
        y = tf.placeholder(tf.int32, (None))
        l = tf.one_hot(y, self.data.num_classes)
        self.x, self.y, self.one_hot_y = x, y, l
        

    def setModelNet(self, network):
        self.network = network
        

    def initTrainingPipeline(self):
        self.logits, self.training_operation = ModelTrainer.InitTrainPipeline(self.x, self.data.num_classes, self.one_hot_y, network=self.network, rate=self.rate)        
    
    
    def initEvalPipeline(self):        
        cp, ac, sv = ModelTrainer.InitEvalPipeline(self.logits, self.one_hot_y)
        self.correct_prediction, self.accuracy_operation, self.saver = cp, ac, sv
        
        
    def evaluate(self, X_data, y_data):
        return ModelTrainer.Evaluate(X_data, y_data, self.x, self.y, self.accuracy_operation, self.BATCH_SIZE)

        
    def train(self, dirname='./', pre_ops=[]):
        
        if (os.path.isdir(dirname) == False): os.mkdir(dirname)
        dirname = Utils.find_an_empty_dir(dirname)
        os.mkdir(dirname)
        outpath = os.path.join(dirname, self.network.__name__)
        logfile = os.path.join(dirname, self.network.__name__ + '_log.txt')
        fid = open(logfile, 'w')
        run_incomplete = True

        try:            
            with tf.Session() as sess:
                print("Training...")
                print()
                Utils.printTestParam(self.EPOCHS, self.BATCH_SIZE, self.rate, pre_ops, save_dir=dirname, fid=fid)
                
                sess.run(tf.global_variables_initializer())
                num_examples = len(self.data.X_train)
                            
                for i in range(self.EPOCHS):
                    self.data.shuffle_training_data()
                    for offset in range(0, num_examples, self.BATCH_SIZE):
                        end = offset + self.BATCH_SIZE
                        batch_x, batch_y = self.data.X_train[offset:end], self.data.y_train[offset:end]
                        sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})
                        
                    validation_accuracy = self.evaluate(self.data.X_valid, self.data.y_valid)
                    print("EPOCH {:2} ...".format(i+1))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print()                
    
                    fid.write("EPOCH {:2} ...".format(i+1))
                    fid.write("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    fid.write('\n')                
    
                self.saver.save(sess, outpath)
                print("Model saved : " + outpath)
                run_incomplete = False
            fid.close()
        finally:
            if (run_incomplete): shutil.rmtree(dirname)    


    def InitTrainPipeline(x, num_classes, one_hot_y, network=(LeNet), rate=0.001):
        logits = network(x, num_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        return (logits, training_operation)
            
    
    def InitEvalPipeline(logits, one_hot_y):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        return correct_prediction, accuracy_operation, saver       


    def Evaluate(X_data, y_data, x, y, accuracy_operation, BATCH_SIZE=128):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
                

class Utils:    

    def get_string_for_array(ar):
        varstr = '['
        for i in ar: varstr+=(i.__name__ + "-") 
    
        if (varstr.endswith('-')): varstr = varstr[0:-1]    
        varstr += ']'
        return varstr
        
    def printTestParam(EPOCHS, BATCH_SIZE, rate, pre_ops, save_dir='.', fid=sys.stdout):
        fid.write("------------------------------------------\n")
        fid.write("-EPOCHS          : " + str(EPOCHS) + "\n")
        fid.write("-BATCH_SIZE      : " + str(BATCH_SIZE) + "\n")
        fid.write("-Pre-process Ops : " + Utils.get_string_for_array(pre_ops) + "\n")
        fid.write("-Save Dir        : " + save_dir + "\n")
        fid.write("------------------------------------------\n")
        fid.write("")    


    def find_an_empty_dir(dirname):
        v = 0
        while (True):
            tmp_name = os.path.join(dirname, "%03d"%(v))
            if (os.path.isdir(tmp_name) == False): return tmp_name
            v+=1

    