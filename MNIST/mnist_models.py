from __future__ import print_function

import os
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class ConvBlock(tf.keras.Model):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, K=5, backbone='residual'):
        super(ConvBlock, self).__init__()

        self.f = layers.Conv2D(c_out, kernel_size=kernel_size, strides=stride, padding='same', use_bias=bias)
        self.g = layers.Conv2D(c_out, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.backbone = backbone
        self.K = K

        self.bn_out = layers.BatchNormalization()
        self.bn_f1  = layers.BatchNormalization()
        self.bn_g   = layers.BatchNormalization()

    def call(self, x):
        f = self.f(tf.keras.activations.relu(self.bn_f1(x)))
        # f = tf.keras.activations.relu(self.bn_f1(self.f(x)))
        h = f
        K = self.K

        if self.backbone == "cnn":
            # h = self.g(tf.keras.activations.relu(self.bn_g(h)))
            h = tf.keras.activations.relu(self.bn_g(self.g(h)))
            g = h

        elif self.backbone == "residual":

            h = h + self.g(tf.keras.activations.relu(self.bn_g(h)))
            # h = h + tf.keras.activations.relu(self.bn_g(self.g(h)))
            h = self.bn_out(h)
            h = tf.keras.activations.relu(h)
            g = h

        elif self.backbone == "pde":
            h0 = h
    
            g  = self.g(tf.keras.activations.relu(self.bn_g(h)))
            # g  = tf.keras.activations.relu(self.bn_g(self.g(h)))
            g1 = g

            dt = 0.2
            dx = 1.
            dy = 1.
    
            # # Const. Dxy default setup in paper Fig. 2
            # Dx = 1.
            # Dy = 1.

            # Dxy for nonlinear isotropic diffusion
            lambda_diffusion = 2.
            sobel = tf.image.sobel_edges(h)
            sobel = tf.math.reciprocal(tf.math.add(tf.math.divide(tf.math.square(sobel), (lambda_diffusion ** 2)), 1))
            Dx = sobel[:, :, :, :, 0]
            Dy = sobel[:, :, :, :, 1]
    
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

        return h, g

class BuildNetworkMNIST(tf.keras.Model):
    def __init__(self, backbone='residual', K=5):
        super(BuildNetworkMNIST, self).__init__()
        self.conv = ConvBlock(1, 8, 3, K=K, backbone=backbone)
        self.fc = layers.Dense(10, activation='softmax')
        self.flatten = layers.Flatten()
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)

    def call(self, x): 
        
        debug = False

        if debug: print("x = ", x.shape)
        
        x, g = self.conv(x)
        f1 = x
        if debug: print("conv1_x = ", x.shape)

        x = tf.keras.activations.relu(x)
        x = self.avg_pool2d(x)
        f2 = x
        if debug: print("avg_pool_x = ", x.shape)
            
        x = self.flatten(x)
        if debug: print("flatten_x = ", x.shape)

        output = self.fc(x)
        if debug: print("output_x = ", output.shape)

        return output, f1, f2, g

def NetworkMNIST(model_type):
    if model_type == "cnn":
        return BuildNetworkMNIST(backbone="cnn")
    elif model_type == "residual":
        return BuildNetworkMNIST(backbone="residual")
    elif model_type == "pde":
        return BuildNetworkMNIST(backbone="pde")

# @tf.function
# def train_step():
#     pass

# def main(model_name="residual"):
#     (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
#     train_images, test_images = train_images / 255.0, test_images / 255.0

#     if len(train_images.shape) == 3:
#         train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
#         test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

#     model = BuildNetworkMNIST(backbone=model_name)

#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

#     history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))

# if __name__ == "__main__":
#     main()