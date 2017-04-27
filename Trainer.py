#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:18:17 2017

@author: jabeerah
"""

import tensorflow as tf
import sys
import os
import shutil
from sklearn.utils import shuffle


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


class Network(object):
    
    def __init__(self):
        self.name = "Default" 
        self.description = "None"        
        self.X_train, self.y_train = None, None
        self.X_valid, self.y_valid = None, None
        self.X_test,  self.y_test  = None, None
        self.x_placeholder, self.y_placeholder = None, None
        self.train_feed, self.test_feed = {}, {}


    def getDescription(self): self.description
    def getSaveDir(self): return None
    def getLogFilePath(self): return None
    
    
class Trainer(object):

    def __initTrainPipeline(self, logits, one_hot_y, rate=0.001):
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)        
            
    def __initEvalPipeline(self, logits, one_hot_y):
        self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()


    def __init__(self, logits, one_hot_y, rate=0.001):
        self.__initTrainPipeline(logits, one_hot_y, rate=rate)
        self.__initEvalPipeline(logits, one_hot_y)
        self.one_hot_y, self.logits, self.rate = one_hot_y, logits, rate


    def evaluate(self, sess, x_data, y_data, x, y, accuracy_operation, eval_feed={}, BATCH_SIZE=128):

        num_examples = len(x_data)
        total_accuracy = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={x: batch_x, y: batch_y}
            feed_dict.update(eval_feed)
            accuracy = sess.run(accuracy_operation, feed_dict=feed_dict)
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
                
    
    def calcLoss(self, sess, x_data, y_data, x, y, loss_op, eval_feed={}, BATCH_SIZE=128):
        num_examples = len(x_data)
        total_loss = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            feed_dict={x: batch_x, y: batch_y}
            feed_dict.update(eval_feed)
            loss = sess.run(loss_op, feed_dict=feed_dict)
            total_loss += (loss * len(batch_x))
        return loss / num_examples


    def calcAccAndLoss(self, sess, X_data, y_data):
        return self.evaluate(sess, X_data, y_data), self.calcLoss(sess, X_data, y_data)

      
    def train(self, sess, dirname='./', pre_ops=[], stat_freq=30, rfunc=None):

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
        
        msg = lambda pcent, sp=30: Utils.progressString(pcent, space=sp)
        bar = lambda ep, valid, t=None, l=None: Utils.progressPrint(ep, valid, train=t, loss=l)
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
                
