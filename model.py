import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
from prepare import *

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Dense
from keras.layers.pooling import AveragePooling2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.initializers import RandomNormal, glorot_uniform
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, LeakyReLU
from keras.applications.resnet50 import ResNet50
import scipy.misc as misc
import scipy.io
import os

def build_generator(input_x,input_mask,reuse=None,batch_normalization=False):
    with tf.variable_scope('gen',reuse=reuse):
        print("Building generator")
        kernels = [128, 128, 64, 64, 32, 32, 32]
        current_input = concatenate([input_x, input_mask], axis=-1)
        inputs = [current_input]

        #resize image to each layer
        for _ in range(len(kernels)-1):
            shape = int(inputs[-1].shape[1].value/2)
            img = tf.image.resize_nearest_neighbor(inputs[-1],(shape, shape))
            inputs.append(img)            
        
        #build model
        for i in range(len(kernels)):
            input = inputs.pop()
            kernel = kernels[i]       
            if i > 0:
                shape = int(net.shape[1].value*2)
                upsampled = tf.image.resize_nearest_neighbor(net, (shape, shape))
                input = concatenate([input, upsampled], axis=-1)

            shape = int(input.shape[1].value)            
            net = Conv2D(filters=kernel, kernel_size=(3,3),padding="same", activation='relu')(input)
            net = Conv2D(filters=kernel, kernel_size=(3,3),padding="same", activation='relu')(net)
            #test 
            if batch_normalization:
                print("batch normalization")
                net = BatchNormalization()(net)
            print(net.shape)
        output = Conv2D(filters=1, kernel_size=(1,1),activation='tanh',padding="same")(net)
        #activation tanh(?)
        print(net.shape, "g_lastconv_"+str(net.shape[1]))
        return output

def build_discriminator(input_x,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        weight_init = RandomNormal(mean=0., stddev=0.02)
        x = Input(tensor=input_x)
        print("Building discriminator")
        numKernels = 32
        # the "lead in" layer
        x = Conv2D(int(numKernels), (3, 3), activation=lrelu,padding='same', 
                            kernel_initializer=weight_init)(x)
        print(x.shape)
        for i in range(1, 5):
            x = Conv2D(int(numKernels*2**i), (3, 3), padding='same', 
                                kernel_initializer=weight_init)(x)
            print(x.shape)
            x = MaxPooling2D(pool_size=2)(x)
            x = LeakyReLU()(x)

        features = Flatten(name='d_flatten')(x)
        logits = Dense(1, activation='linear')(features)
        output = tf.nn.sigmoid(logits)
        return output,logits

#functions to VGG model
def build_vgg_net(ntype,nin,nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name)+nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers,i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, bias.size))
    return weights, bias


def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg", reuse=reuse) as scope:
        net = {}
        vgg_rawnet = scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
        vgg_layers = vgg_rawnet['layers'][0]
        #images are normalized, in tf we use -1 tot 1
        input /= 0.5
        input -= 1.
        
        net['input'] = K.repeat_elements(input,3,3)
        net['conv1_1'] = build_vgg_net('conv', net['input'], get_weight_bias(vgg_layers,0), name='vgg_conv1_1')
        net['conv1_2'] = build_vgg_net('conv', net['conv1_1'], get_weight_bias(vgg_layers,2), name='vgg_conv1_2')
        net['pool1'] = build_vgg_net('pool', net['conv1_2'])
        net['conv2_1'] = build_vgg_net('conv', net['pool1'], get_weight_bias(vgg_layers,5), name='vgg_conv2_1')
        net['conv2_2'] = build_vgg_net('conv', net['conv2_1'], get_weight_bias(vgg_layers,7), name='vgg_conv2_2')
        net['pool2'] = build_vgg_net('pool', net['conv2_2'])
        net['conv3_1'] = build_vgg_net('conv', net['pool2'], get_weight_bias(vgg_layers,10), name='vgg_conv3_1')
        net['conv3_2'] = build_vgg_net('conv', net['conv3_1'], get_weight_bias(vgg_layers,12), name='vgg_conv3_2')
        net['conv3_3'] = build_vgg_net('conv', net['conv3_2'], get_weight_bias(vgg_layers,14), name='vgg_conv3_3')
        net['conv3_4'] = build_vgg_net('conv', net['conv3_3'], get_weight_bias(vgg_layers,16), name='vgg_conv3_4')
        net['pool3'] = build_vgg_net('pool', net['conv3_4'])
        net['conv4_1'] = build_vgg_net('conv', net['pool3'], get_weight_bias(vgg_layers,19), name='vgg_conv4_1')
        net['conv4_2'] = build_vgg_net('conv', net['conv4_1'], get_weight_bias(vgg_layers,21), name='vgg_conv4_2')
        net['conv4_3'] = build_vgg_net('conv', net['conv4_2'], get_weight_bias(vgg_layers,23), name='vgg_conv4_3')
        net['conv4_4'] = build_vgg_net('conv', net['conv4_3'], get_weight_bias(vgg_layers,25), name='vgg_conv4_4')
        net['pool4'] = build_vgg_net('pool', net['conv4_4'])
        net['conv5_1'] = build_vgg_net('conv', net['pool4'], get_weight_bias(vgg_layers,28), name='vgg_conv5_1')
        net['conv5_2'] = build_vgg_net('conv', net['conv5_1'], get_weight_bias(vgg_layers,30), name='vgg_conv5_2')
        return net



def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

def dcgan_function(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))


def save(ckpt_name, i, saver, sess, ckpt=False):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    saver.save(sess, models_dir+ckpt_name)
    print("Saved ", ckpt_name)
    if ckpt:
        saver.save(sess, checkpoints_dir+ckpt_name+"_"+str(i))
        print("Saved ckpt ", ckpt_name+"_"+str(i))


def load(ckpt_name, saver, sess):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    saver.restore(sess, models_dir+ckpt_name)
    print("Loaded ", ckpt_name)
