#%%
################################################################################
## Load Data
################################################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

#%%
################################################################################
## 
##  The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
##  However, the LeNet architecture only accepts 32x32xC images, where C is the 
##  number of color channels. In order to reformat the MNIST data into a shape that 
##  LeNet will accept, we pad the data with two rows of zeros on the top and bottom, 
##  and two columns of zeros on the left and right (28+2+2 = 32). 
##  
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
##
################################################################################

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

#%%
################################################################################
##  Visualize Data
##
##  View a sample from the dataset.
##  You do not need to modify this section.
##  
################################################################################
import random
import numpy as np
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

#%%
################################################################################
##  Preprocess Data
##  
##  Shuffle the training data.
##  You do not need to modify this section.
################################################################################
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

#%%
################################################################################
## Setup TensorFlow
## 
## The EPOCH and BATCH_SIZE values affect the training speed and model accuracy.
## You do not need to modify this section.
################################################################################
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

#%%
################################################################################
##  TODO: IMPLEMENT LENET-5
##  
##  Implement the LeNet-5 neural network architecture.
##  This is the only cell you need to edit.
##  
##  INPUT
##  
##  The LeNet architecture accepts a 32x32xC image as input, where C is the number 
##  of color channels. Since MNIST images are grayscale, C is 1 in this case.
##  Architecture
##  
##  * Layer 1:      Convolutional. The output shape should be 28x28x6.
##  * Activation:   Your choice of activation function.
##  * Pooling:      The output shape should be 14x14x6.
##  * Layer 2:      Convolutional. The output shape should be 10x10x16.
##  * Activation:   Your choice of activation function.
##  * Pooling:      The output shape should be 5x5x16.
##  * Flatten.      Flatten the output shape of the final pooling layer such 
##                  that it's 1D instead of 3D. The easiest way to do is by using 
##                  tf.contrib.layers.flatten, which is already imported for you.
##  * Layer 3:      Fully Connected. This should have 120 outputs.
##  * Activation:   Your choice of activation function.
##  * Layer 4:      Fully Connected. This should have 84 outputs.
##  * Activation:   Your choice of activation function.
##  * Layer 5:      Fully Connected (Logits). This should have 10 outputs.
##  
##  OUTPUT
##  
##  Return the result of the 2nd fully connected layer.
################################################################################

def print_output_info(input_size, filter_size, padding, stride):
    print("Input Size : " + str(input_size))
    print("Filter Size: " + str(filter_size))
    print("Padding    : " + str(padding))
    print("Stride     : " + str(stride))
    print("Output size: " + str((input_size - filter_size + 2*padding)/stride + 1))


from tensorflow.contrib.layers import flatten

def fully_connected(x, num_outs, mu=0.0, sigma=1.0, dtype=np.float32, name=None):
    num_ins = x.shape.as_list()[-1]
    
    Ws = tf.Variable(tf.truncated_normal((num_ins, num_outs), mean=mu, stddev=sigma, dtype=dtype, name=name))
    Bs = tf.Variable(np.zeros((num_outs,), dtype=dtype))
    return tf.matmul(x, Ws) + Bs
    
def conv2d(x, fh, fw, fd, strides=[1, 1, 1, 1], padding='VALID', mu=0.0, sigma=1.0, dtype=np.float32, name=None):
    
    (_, xh, xw, xc) = x.shape.as_list()
    
    F_W = tf.Variable(tf.truncated_normal(shape=(fh, fw, xc, fd), mean=mu, stddev=sigma, dtype=dtype, name=None))
    F_b = tf.Variable(np.zeros((fd, ), dtype=dtype))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'SAME'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(x, F_W, strides, padding) + F_b

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = conv2d(x, 5, 5, 6, strides = [1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)
    # TODO: Activation.
    relu1 = tf.nn.relu(conv1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(pool1, 5, 5, 16, strides=[1, 1, 1, 1], padding='VALID', mu=mu, sigma=sigma)        
    # TODO: Activation.
    relu2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(pool2)   
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = fully_connected(fc0, 120, mu, sigma)
    # TODO: Activation.
    fc1_relu = tf.nn.relu(fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = fully_connected(fc1_relu, 84, mu, sigma)    
    # TODO: Activation.
    fc2_relu = tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    return fully_connected(fc2_relu, 10, mu, sigma)   


#%%
################################################################################
##  FEATURES AND LABELS
##
##  Train LeNet to classify MNIST data.
##  x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
################################################################################
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


#%%
################################################################################
##  TRAINING PIPELINE
##
##  Create a training pipeline that uses the model to classify MNIST data.
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
################################################################################
rate = 0.001

logits = LeNet(x)
#%%
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


#%%
################################################################################
##  MODEL EVALUATION
##  
##  Evaluate how well the loss and accuracy of the model for a given dataset.
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
################################################################################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


#%%
################################################################################
##  TRAIN THE MODEL
##  
##  Run the training data through the training pipeline to train the model.
##  Before each epoch, shuffle the training set.
##  After each epoch, measure the loss and accuracy of the validation set.
##  Save the model after training.
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
################################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


#%%
################################################################################    
##  EVALUATE THE MODEL
##  
##  Once you are completely satisfied with your model, evaluate the performance of 
##  the model on the test set.
##  
##  Be sure to only do this once!
##  If you were to measure the performance of your trained model on the test set, 
##  then improve your model, and then measure the performance of your model on the 
##  test set again, that would invalidate your test results. You wouldn't get a 
##  true measure of how well your model would perform against real data.
##  
##  YOU DO NOT NEED TO MODIFY THIS SECTION.
################################################################################
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
