from termios import NL1
import tensorflow as tf
from tensorflow.keras import layers
import sys

from building_blocks import *
from global_layer import *

class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, global_ft=False):
        super(BuildResNet, self).__init__()
        n1, n2, n3, n4 = 16, 32, 64, 64
        self.in_channels = n1
        self.global_ft = global_ft
        
        self.conv1 = layers.Conv2D(self.in_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, n1, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, n2, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, n3, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, n4, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

        if self.global_ft:
            self.global1 = GlobalFeatureBlock_Diffusion(n1)   
            self.global2 = GlobalFeatureBlock_Diffusion(n2) 
            self.global3 = GlobalFeatureBlock_Diffusion(n3)

    
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        if self.global_ft:
            out = self.global1(out)
        
        out = self.layer2(out)
        if self.global_ft:
            out = self.global2(out)
        
        out = self.layer3(out)
        if self.global_ft:
            out = self.global3(out)
        
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def ResNet(model_type, num_classes):
    if model_type == 'resnet18':
        return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes, global_ft=False)
    elif model_type == 'resnet18_global':
        return BuildResNet(BasicBlock, [1, 1, 1, 1], num_classes, global_ft=True)
    elif model_type == 'resnet32':
        return BuildResNet(BasicBlock, [5, 5, 5, 5], num_classes, global_ft=False)
    elif model_type == 'resnet32_global':
        return BuildResNet(BasicBlock, [1, 1, 1, 1], num_classes, global_ft=True)

