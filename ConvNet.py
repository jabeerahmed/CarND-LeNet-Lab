import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import math

def CalcOutputSize(in_x, k_x, s_x, valid=True):
    m = (k_x - 1) if valid else 0
    return math.ceil(float(in_x - m) / float(s_x))

def GetOutputSize(inSize, kSize, stride, paddingValid=True):
    w = CalcOutputSize(inSize[1], kSize[1], stride[1], paddingValid)
    h = CalcOutputSize(inSize[2], kSize[2], stride[2], paddingValid)
    c = kSize[0]
    return (w, h, c)

class Layer(object):
    
    def __init__(self, ltype, inLayer, name=None):
        self.__inLayer = inLayer 
        self.__layerType = ltype
        self.__name = name if name is not None else ltype
        self.__nextLayer = None
        self.__tensor = None
        self.__layerContainer = None
        if (self.inLayer() is not None): self.inLayer().setNextLayer(self)

    def __str__(self):      return self.name()
    def name(self):         return self.__name
    def tensor(self):       return self.__tensor
    def inSize(self):       return self.inTensor().get_shape().as_list()
    def outSize(self):      return self.tensor().get_shape().as_list()
    def inLayer(self):      return self.__inLayer
    def inTensor(self):     return self.inLayer().tensor()
    def layerType(self):    return self.__layerType
    def nextLayer(self):    return self.__nextLayer
    def getGroup(self):     return self.__layerContainer    
    
    def setNextLayer(self, layer):  self.__nextLayer = layer
    def setTensor(self, tensor):    self.__tensor = tensor
    def setGroup(self, group):      self.__layerContainer = group

    def setInLayer(self, layer): 
        self.__inLayer = layer
        self.createLayer()
        self.updateLayer()

    def updateLayer(self): 
        print("update layer: name: {:20} | type: {:20}".format(self.tensor().name, self.layerType()))
        if self.nextLayer() is not None: self.nextLayer().updateLayer()            

    def createLayer(self): raise NotImplementedError('{} needs to implement method'.format(self.layerType()))

    def insertLayer(self, layer):   raise NotImplementedError('{} needs to implement method'.format(self.layerType()))
    def swapLayer(self, layer):     raise NotImplementedError('{} needs to implement method'.format(self.layerType()))       
    def addLayer(self, layer):      raise NotImplementedError('{} needs to implement method'.format(self.layerType())) 
    def removeLayer(self):          self.swapLayer(None)
    
    
class BranchOutLayer(Layer):
        
    def __init__(self, inLayer, name=None):
        Layer.__init__(self, 'branch_out', inLayer, name=name)
        self.__next_layers = {}       

    def tensor(self): return self.inTensor()

    def setNextLayer(self, layer):  
        if hash(layer) not in self.__next_layers.keys():
            self.__next_layers[hash(layer)] = layer

    def updateLayer(self): 
        for k in self.__next_layers:  
            layer = self.__next_layers[k]
            layer.updateLayer()            
            
    def nextLayer(self): return [self.__next_layers[i] for i in self.__next_layers]

    
class ConvLayerBase(Layer):
        
    def __init__(self, layerType, inLayer, kSize, strides, padding, name=None):
        Layer.__init__(self, layerType, inLayer, name=name)
        self.__kSize = kSize
        self.__strides = strides
        self.__padding = padding
    
    def kSize(self):        return self.__kSize
    def strides(self):      return self.__strides
    def padding(self):      return self.__padding

    def calcOutputSize(self):
        nDims = GetOutputSize(self.inSize(), self.kSize(), self.strides(), self.padding() == 'VALID')
        return (self.inSize()[0], ) + nDims


class ConvLayer(ConvLayerBase):
    
    def conv2d(x, kSize, strides, padding, mu, sigma, name):
        (_, fh, fw, fd) = kSize
        (_, xh, xw, xc) = x.shape.as_list()
        F_W = tf.Variable(tf.truncated_normal(shape=(fh, fw, xc, fd), mean=mu, stddev=sigma, name=None), dtype=x.dtype)
        F_b = tf.Variable(np.zeros((fd, )), dtype=x.dtype)
        return tf.nn.conv2d(x, F_W, strides, padding) + F_b

    OPS = { '2d' : conv2d }
    
    def __init__(self, inLayer, convType='2d', kSize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name=None, mu=0.0, sigma=1.0, weights_initializer=None, bais_initializer=None):
        ConvLayerBase.__init__(self, 'conv_'+convType, inLayer, kSize, strides, padding, name=name)
        self.weights_initializer=weights_initializer            
        self.bais_initializer=bais_initializer
        self.mu = mu
        self.sigma = sigma
        self.op = ConvLayer.OPS[convType]
        self.createLayer()
    
    def createLayer(self): 
        self.setTensor(self.op(self.inTensor(), self.kSize(), self.strides(), self.padding(), self.mu, self.sigma, self.name()))
        return self.tensor()
    
    
class PoolLayer(ConvLayerBase):
    
    MAX     = tf.nn.max_pool
    AVG     = tf.nn.avg_pool
    MAX_3D  = tf.nn.max_pool3d
    AVG_3D  = tf.nn.avg_pool3d
    MAX_W_ARGMAX = tf.nn.max_pool_with_argmax

    OPS = {'max': MAX, 'avg': AVG, 'max3d': MAX_3D, 'avg3d': AVG_3D }
    
    def __init__(self, poolType, inLayer, kSize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name=None):
        ConvLayerBase.__init__(self, 'pool_'+poolType, inLayer, kSize, strides, padding, name=name)
        self.op = PoolLayer.OPS[poolType]
        self.createLayer()
        
    def createLayer(self): 
        self.setTensor(self.op(self.inTensor(), self.kSize(), self.strides(), self.padding(), name=self.name()))
        return self.tensor()


class ActivationLayer(Layer):
    OPS = { 'relu'      : tf.nn.relu,
            'relu6'     : tf.nn.relu6,
            'elu'       : tf.nn.elu,
            'softplus'  : tf.nn.softplus,
            'softsign'  : tf.nn.softsign,
            'sigmoid'   : tf.sigmoid,
            'tanh'      : tf.tanh }

    def __init__(self, actType, inLayer, name=None):
        Layer.__init__(self, 'act_'+actType, inLayer, name=name)
        self.op = ActivationLayer.OPS[actType]
        self.createLayer()
        
    def createLayer(self): 
        self.setTensor(self.op(self.inTensor(), name=self.name()))
        return self.tensor()


class FullyConnectedLayer(Layer):

    def fully_connected(x, num_outs, mu=0.0, sigma=1.0, name=None):
        num_ins = x.shape.as_list()[-1]        
        Ws = tf.Variable(tf.truncated_normal((num_ins, num_outs), mean=mu, stddev=sigma, name=name, dtype=x.dtype))
        Bs = tf.Variable(np.zeros((num_outs,)), dtype=x.dtype)
        return tf.matmul(x, Ws) + Bs

    def __init__(self, inLayer, numClasses, mu=0.0, sigma=1.0, name=None):
        Layer.__init__(self, 'fc', inLayer, name=name)
        self.mu=mu
        self.sigma=sigma
        self.numClasses=numClasses
        self.createLayer()
        
    def createLayer(self): 
        self.setTensor(FullyConnectedLayer.fully_connected(self.inTensor(), self.numClasses, mu=self.mu, sigma=self.sigma, name=self.name()))
        return self.tensor()
    

class InputLayer(Layer):    
    def __init__(self, inputTensor, name=None):
        Layer.__init__(self, 'input', None, name=name)            
        self.__tensor = inputTensor
    def tensor(self):      return self.__tensor
    def inTensor(self):    return self.tensor()
    def inSize(self):      return self.inTensor().get_shape().as_list()
    def outSize(self):     return self.tensor().get_shape().as_list()
    def createLayer(self): return self.tensor()
    def setTensor(self, tensor): 
        self.__tensor = tensor
        self.updateLayer()
        
        
class BasicLayer(Layer):
        
    def __init__(self, layerType, inLayer, func, func_args={}, name=None):
        Layer.__init__(self, layerType, inLayer, name=name)
        self.func = func
        self.func_args = func_args
        self.createLayer()
        
    def createLayer(self): 
        self.setTensor(self.func(self.inTensor(), **self.func_args))
        return self.tensor()

def FlattenLayer(inLayer, name=None, args={}):
    return BasicLayer('flatten', inLayer, flatten, name=name, func_args=args)
                

class GroupLayer(Layer):
    
    layerType = "GroupLayer"
    
    def __init__(self, layers, name=None): 
        if layers is None: raise Exception("None layers")
        if False == hasattr(layers, '__iter__'): layers = [layers] 
        self.__layers = layers
        self.__firstLayer = layers[0]
        self.__lastLayer  = layers[-1] 
        self.__groupName  = name if name is not None else GroupLayer.layerType
        for i in layers: i.setGroup(self)

    def __getitem__(self, key):  return self.__layers[key]
    def __len__(self):           return len(self.__layers)
    def __str__(self):           return "{}: {}".format(Layer.__str__(self),[i.name() for i in self.__layers])
    def length(self):            return len(self)
    
    
    def inLayer(self):      return self.__firstLayer.inLayer()
    def inTensor(self):     return self.inLayer().tensor()
    def inSize(self):       return self.inTensor().get_shape().as_list()
    def outSize(self):      return self.tensor().get_shape().as_list()
    def tensor(self):       return self.__lastLayer.tensor()
    def layerType(self):    return GroupLayer.layerType
    def name(self):         return self.__groupName
    def nextLayer(self):    return self.__lastLayer.nextLayer()
    def setNextLayer(self, layer): self.__lastLayer.setNextLayer(layer)
    def setTensor(self, tensor):   self.__lastLayer.setTensor(tensor)
    def updateLayer(self):         self.__firstLayer.updateLayer()
        
    def createLayer(self):          raise NotImplementedError('{} needs to implement method'.format(self.layerType()))
    def insertLayer(self, layer):   raise NotImplementedError('{} needs to implement method'.format(self.layerType()))
    def swapLayer(self, layer):     raise NotImplementedError('{} needs to implement method'.format(self.layerType()))       
    def addLayer(self, layer):      raise NotImplementedError('{} needs to implement method'.format(self.layerType())) 
    def removeLayer(self):          raise NotImplementedError('{} needs to implement method'.format(self.layerType()))

    def CreateGroup(layer, n=100, name=None):
        layers = []
        for i in range(n):
            if layer is not None:
                layers.insert(0, layer)
                layer = layer.inLayer()
            else: 
                break
        return GroupLayer(layers, name=name)
    
    def print(group):
        for i, l in enumerate(group):
            lStr = "{:20} in:{:20}, out:{:20} {}".format(l.name(), str(l.inSize()), str(l.outSize()), str(l))
            print("Layer {:2d} : {}".format(i, lStr))
            
class ConvParams:    
    def __init__(self, K=[1,1,1,1], S=[1,1,1,1], P='VALID'):
        self.K, self.S, self.P = K,S,P



def ConvReluPoolLayer(prev, convP=None, poolP=None, suffix='', mu=0.0, sigma=1.0):
    r = ()
    if convP is not None: 
        prev = ConvLayer(prev, name='conv_'+suffix, kSize=convP.K, strides=convP.S, padding=convP.P, mu=mu, sigma=sigma)
        r = r + (prev,)    
        prev = ActivationLayer('relu', prev, name='relu_'+suffix)
        r = r + (prev,)
    
    if poolP is not None:
        prev = PoolLayer('max', prev, name='maxpool_'+suffix, kSize=poolP.K, strides=poolP.S, padding=poolP.P)
        r = r + (prev,)
        
    return GroupLayer(r, "ConvReluPool"+suffix)


#%%


if __name__ == '__main__':
    print('ConvNet main loop')
    x  = tf.placeholder(tf.float32, (None, 32, 32, 3))
    num_classes = 43
    mu, sigma = 0.1, 0.1
    
    Group = lambda layer=None, n=100, name=None: GroupLayer.CreateGroup(layer,name=name, n=n)
    
    L = InputLayer(x)
    L = ConvReluPoolLayer(L, suffix='1_0', convP=ConvParams(K=[1,5,5,6],  S=[1,1,1,1], P='VALID'), poolP=ConvParams(K=[1,2,2,1], S=[1,2,2,1], P='VALID'), mu=mu, sigma=sigma)
    L = ConvReluPoolLayer(L, suffix='2_0', convP=ConvParams(K=[1,5,5,16], S=[1,1,1,1], P='VALID'), poolP=ConvParams(K=[1,2,2,1], S=[1,2,2,1], P='VALID'), mu=mu, sigma=sigma)
    L = FlattenLayer(L)
    L = Group(name='FC4', n=2, layer=ActivationLayer('relu', FullyConnectedLayer(L, 120, name='fc_4_0', mu=mu, sigma=sigma), name='fc_4_0_relu'))
    L = Group(name='FC5', n=2, layer=ActivationLayer('relu', FullyConnectedLayer(L,  84, name='fc_5_0', mu=mu, sigma=sigma), name='fc_5_0_relu'))
    L = FullyConnectedLayer(L,  num_classes, name='FC6', mu=mu, sigma=sigma)
    
    G = GroupLayer.CreateGroup(L, name='LeNet')
    GroupLayer.print(G)
    
