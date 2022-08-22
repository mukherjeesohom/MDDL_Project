from termios import NL1
import tensorflow as tf
from tensorflow.keras import layers
import sys

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out