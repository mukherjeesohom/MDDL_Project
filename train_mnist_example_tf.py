from __future__ import print_function

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers

class ConvBlock(tf.keras.Model):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, K=1, backbone='residual'):
        super(ConvBlock, self).__init__()

        self.f = layers.Conv2D(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.g = layers.Conv2D(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone = backbone
        self.K = K

        # self.act = layers.ReLU()

        self.bn_out = layers.BatchNormalization(c_out)
        self.bn_f1  = layers.BatchNormalization(c_in)
        self.bn_g   = layers.BatchNormalization(c_out)

        def call(self, x):
            f = self.f(tf.keras.activations.relu(self.bn_f1(x)))
            h = f

            K = self.K

            if self.backbone == "cnn":
                h = self.g(tf.keras.activations.relu(self.bn_g(h)))
            elif self.backbone == "residual":
                for k in range(self.K):
                    h = h + self.g(tf.keras.activations.relu(self.bn_g(h)))
                h = self.bn_out(h)
                h = tf.keras.activations.relu(h)
            else:
                h0 = h
        
                g  = self.g(tf.keras.activations.relu(self.bn_g(h)))
                g1 = g

                dt = 0.2
                dx = 1.
                dy = 1.
        
                Dx = 1.
                Dy = 1.
        
                ux = (1. / (2*dx)) * (tf.roll(g, 1, axis=2) - tf.roll(g, -1, axis=2))
                vy = (1. / (2*dy)) * (tf.roll(g1, 1, axis=3) - tf.roll(g1, -1, axis=3))
        
                Ax = g  * (dt / dx)
                Ay = g1 * (dt / dy)
                Bx = Dx * (dt / (dx*dx))
                By = Dy * (dt / (dy*dy))
                E  = (ux + vy) * dt
        
                D = (1. / (1 + 2*Bx + 2*By))
        
                for k in range(self.K):
                    prev_h = h
                        
                    h = D  *   (   (1 - 2*Bx - 2*By) * h0 - 2 * E * h 
                                + (-Ax  + 2*Bx) * tf.roll(h, 1, axis=2) 
                                + ( Ax  + 2*Bx) * tf.roll(h, -1, axis=2) 
                                + (-Ay  + 2*By) * tf.roll(h, 1, axis=3)  
                                + ( Ay  + 2*By) * tf.roll(h, -1, axis=3)  
                                + 2 * dt * f )
                    h0 = prev_h
                
                h = self.bn_out(h)
                h = tf.keras.activations.relu(h)

            return h

class NetworkMNIST(tf.keras.Model):
    def __init__(self, backbone='residual', K=5):
        super(NetworkMNIST, self).__init__()
        self.conv = ConvBlock(1, 1, 3, K=K, backbone=backbone)
        self.fc = layers.Dense(10, activation='softmax')
        self.flatten = layers.Flatten()
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)

    def forward(self, x): 
            
        x = self.conv(x)
        f1 = x

        x = tf.keras.activations.relu(x)
        x = self.avg_pool2d(x)
        f2 = x
            
        x = self.flatten(x)
        output = self.fc(x)

        return output, f1, f2

