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
    def __init__(self, data_dir='traffic-signs-data', label_csv_file='signnames.csv'):
        self.data_dir=data_dir
        self.label_csv_file=label_csv_file
        
        # Load data set
        train, test, valid = self.load_image_data()          
 
        # Load Image Signs  
        self.signs = self.read_signnames()
        self.num_classes=len(self.signs)

        # Set Data
        self.set_data(train['features'], test['features'], valid['features'],
                      train['labels']  , test['labels'],   valid['labels'])
        self.train, self.test, self.valid = (train, test, valid) 
        
        # print info
        self.print_data_info()

    
    def reset_data(self):
        # Set Data
        self.set_data(self.train['features'], self.test['features'], self.valid['features'],
                      self.train['labels']  , self.test['labels'],   self.valid['labels'])
        
    
    def set_data(self, X_train, X_test, X_valid, y_train, y_test, y_valid):
        self.X_train, self.y_train = np.array(X_train, dtype=np.float32), y_train
        self.X_valid, self.y_valid = np.array(X_valid, dtype=np.float32), y_valid
        self.X_test,  self.y_test  = np.array( X_test, dtype=np.float32), y_test
        self.image_shape = self.X_train[0].squeeze().shape
        
        

    def load_image_data(self):
        return ImageClassifier.Load_Image_Data(self.data_dir)


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
        self.reset_data()
        pp_train = np.zeros_like(self.X_train)
        pp_test  = np.zeros_like(self.X_test)
        pp_valid = np.zeros_like(self.X_valid)
        self.preprocess(operations, self.X_train, dst_buf=pp_train)
        self.preprocess(operations, self.X_test , dst_buf=pp_test )
        self.preprocess(operations, self.X_valid, dst_buf=pp_valid)
        self.set_data(pp_train, pp_test, pp_valid, self.train['labels']  , self.test['labels'],   self.valid['labels'])
                
                
    def ZMean(src, i, dst):  
#        if (dst is None): dst = src
        dst[i] = src[i] - np.mean(src[i])

    def Norm(src, i, dst): 
#        if (dst is None): dst = src
        dst[i] = src[i]/255.0

    def UnitVar(src, i, dst): 
#        if (dst is None): dst = src
        dst[i] = src[i]/(np.std(src[i]))


    def ZShift(src, i, dst): 
#        if (dst is None): dst = src
        dst[i] = (src[i] - 128.0)
    
    
    def shuffle_training_data(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        

    def shuffle_test_data(self):
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)

    def shuffle_valid_data(self):
        self.X_valid, self.y_valid = shuffle(self.X_valid, self.y_valid)

    def createModelTrainer(self, args={}):
        return ModelTrainer(self.X_train, self.y_train, self.X_valid, self.y_valid, **args)

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
    
    def addDropOut(layer, keep_prob, train_feed, eval_feed, name=None):
        if ( 0 < keep_prob < 1 ):
            kp = tf.placeholder(tf.float32, name=name)
            train_feed.update({kp: keep_prob})
            eval_feed.update({kp: 1.0})
            return tf.nn.dropout(layer, keep_prob=kp)
        return layer

    def LeNetWithDropOut(x, num_outputs, mu=0.0, sigma=0.1, dropouts={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}):    
        train_feed, eval_feed = {}, {}

        Layer = 0       # Convolutional. Input = 32x32x3. Output = 28x28x6.
        conv1 = ModelTrainer.conv2d(x, 5, 5, 6, strides = [1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)
        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        if (Layer in dropouts): pool1 = ModelTrainer.addDropOut(pool1, dropouts[Layer], train_feed, eval_feed)
            
        Layer = 1   #Convolutional. Input = 32x32x3. Output = 28x28x6
        conv2 = ModelTrainer.conv2d(pool1, 5, 5, 16, strides=[1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        if (Layer in dropouts): pool2 = ModelTrainer.addDropOut(pool2, dropouts[Layer], train_feed, eval_feed)

        Layer = 2       # Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(pool2)
        if (Layer in dropouts): fc0 = ModelTrainer.addDropOut(fc0, dropouts[Layer], train_feed, eval_feed)

        Layer = 3       # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = ModelTrainer.fully_connected(fc0, 120, mu, sigma)
        fc1_relu = tf.nn.relu(fc1)
        if (Layer in dropouts): fc1_relu = ModelTrainer.addDropOut(fc1_relu, dropouts[Layer], train_feed, eval_feed)

        Layer = 4       # Fully Connected. Input = 120. Output = 84
        fc2 = ModelTrainer.fully_connected(fc1_relu, 84, mu, sigma)
        fc2_relu = tf.nn.relu(fc2)
        if (Layer in dropouts): fc2_relu = ModelTrainer.addDropOut(fc2_relu, dropouts[Layer], train_feed, eval_feed)

        Layer = 5     # Fully Connected. Input = 84. Output = 43.
        return ModelTrainer.fully_connected(fc2_relu, num_outputs, mu, sigma), train_feed, eval_feed
    
    
    def LeNet(x, num_outputs, mu=0.0, sigma=0.1):
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
        return ModelTrainer.fully_connected(fc2_relu, num_outputs, mu, sigma), {}, {}
    

#    def __init__(self, data, network=LeNet, network_args={}, EPOCHS=10, BATCH_SIZE=128, rate=0.001):
    def __init__(self, X_train, y_train, X_test, y_test, network=LeNet, network_args={}, EPOCHS=10, BATCH_SIZE=128, rate=0.001):
        # Hyper-Parameters
        self.EPOCHS=EPOCHS 
        self.BATCH_SIZE=BATCH_SIZE
        self.rate=rate
        self.initialModel(X_train, y_train, X_test, y_test, network, network_args)
        

    def initialModel(self, X_train, y_train, X_test, y_test, network, network_args={}):
        self.initialize_data_container(X_train, y_train, X_test, y_test)
        self.setModelNet(network, network_args)
        self.initTrainingPipeline()
        self.initEvalPipeline()  
    
        
    def initialize_data_container(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = shuffle(X_train, y_train)
        self.X_test,  self.y_test  = shuffle(X_test,  y_test)
        self.num_classes = len(np.unique(y_train))
        self.image_shape = X_train[0].shape
        x = tf.placeholder(tf.float32, (None,) + self.image_shape)
        y = tf.placeholder(tf.int32, (None))
        l = tf.one_hot(y, self.num_classes)
        self.x, self.y, self.one_hot_y = x, y, l
        

    def setModelNet(self, network, network_args={}):
        self.network = network
        self.network_args = network_args
        

    def initTrainingPipeline(self):
        ret = ModelTrainer.InitTrainPipeline(self.x, self.num_classes, self.one_hot_y, rate=self.rate,
                                             network=self.network, network_args=self.network_args)
        self.logits, self.training_operation, self.loss_operation, self.train_feed, self.eval_feed = ret

    def initEvalPipeline(self):
        cp, ac, sv = ModelTrainer.InitEvalPipeline(self.logits, self.one_hot_y)
        self.correct_prediction, self.accuracy_operation, self.saver = cp, ac, sv
        
        
    def evaluate(self, sess, X_data, y_data):
        return ModelTrainer.Evaluate(sess, X_data, y_data, self.x, self.y, self.accuracy_operation, self.eval_feed, BATCH_SIZE=self.BATCH_SIZE)


    def calcLoss(self, sess, X_data, y_data):
        return ModelTrainer.CalcLoss(sess, X_data, y_data, self.x, self.y, self.loss_operation, self.eval_feed, BATCH_SIZE=self.BATCH_SIZE)


    def calcAccAndLoss(self, sess, X_data, y_data):
        return self.evaluate(sess, X_data, y_data), self.calcLoss(sess, X_data, y_data)


    def progressString(pcent, space=50):
        bar = '[{:' + str(space) + '}]'
        msg = "{} {:1.2f}".format(bar.format("#"*np.int(space * pcent)), pcent)
        return msg
    
    
    def progressPrint( epoch, valid, train=None, loss=None):
        msg = "{:3} | V {}".format(epoch, valid)
        if (train is not None): msg += " | T {}".format(train)
        if (loss  is not None): msg += " | L {}".format(loss)
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()        
    
    
    def train(self, dirname='./', pre_ops=[], stat_freq=30, rfunc=None):
        if (os.path.isdir(dirname) == False): os.makedirs(dirname)
        dirname = Utils.find_an_empty_dir(dirname)
        os.mkdir(dirname)
        outpath = os.path.join(dirname, self.network.__name__)
        logfile = os.path.join(dirname, self.network.__name__ + '_log.txt')
        fid = open(logfile, 'w')
        run_incomplete = True
        stats = []

        print("TrainFeed = " + str(self.train_feed))
        print("EvalFeed  = " + str(self.eval_feed))
        
        msg = lambda pcent, sp=30: ModelTrainer.progressString(pcent, space=sp)
        bar = lambda ep, valid, t=None, l=None: ModelTrainer.progressPrint(ep, valid, train=t, loss=l)
        try:            
            with tf.Session() as sess:
                print("Training...")
                print()
                Utils.printTestParam(self.EPOCHS, self.BATCH_SIZE, self.rate, pre_ops, save_dir=dirname, fid=fid)
                
                sess.run(tf.global_variables_initializer())
                num_examples = len(self.X_train)
                validation_accuracy = 0
                
                for i in range(self.EPOCHS):
                    tot_acc, tot_loss, tot_n = 0,0,0
                    self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
                    for n, offset in enumerate(range(0, num_examples, self.BATCH_SIZE)):
                        end = offset + self.BATCH_SIZE
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

#                    fid.write("EPOCH {:2} ...".format(i+1))
#                    fid.write("Validation Accuracy = {:.3f}".format(validation_accuracy))
#                    fid.write('\n')                
                    
                self.saver.save(sess, outpath)
                print("Model saved : " + outpath)
                run_incomplete = False
            fid.close()
        finally:
            if (run_incomplete): shutil.rmtree(dirname)

        return stats


    def InitTrainPipeline(x, num_classes, one_hot_y, network=(LeNet), network_args={}, rate=0.001):
        logits, train_feed, eval_feed = network(x, num_classes, **network_args)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        training_operation = optimizer.minimize(loss_operation)
        return (logits, training_operation, loss_operation, train_feed, eval_feed)
            
    
    def InitEvalPipeline(logits, one_hot_y):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        return correct_prediction, accuracy_operation, saver       

    def Evaluate(sess, x_data, y_data, x, y, accuracy_operation, eval_feed={}, BATCH_SIZE=128):
        num_examples = len(x_data)
        total_accuracy = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={x: batch_x, y: batch_y}
            feed_dict.update(eval_feed)
            accuracy = sess.run(accuracy_operation, feed_dict=feed_dict)
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
                

    def CalcLoss(sess, x_data, y_data, x, y, loss_op, eval_feed={}, BATCH_SIZE=128):
        num_examples = len(x_data)
        total_loss = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={x: batch_x, y: batch_y}
            feed_dict.update(eval_feed)
            loss = sess.run(loss_op, feed_dict=feed_dict)
            total_loss += (loss * len(batch_x))
        return loss / num_examples
                



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

    