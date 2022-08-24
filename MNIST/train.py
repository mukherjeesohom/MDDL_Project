"""Train CIFAR-10 with TensorFlow2.0."""
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm

from mnist_models import *
from utils import *

parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
parser.add_argument('--model', default="pde", type=str, help='model type; choose from: "cnn", "residual", "pde"')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of training epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='specify which gpu to be used')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.model = args.model.lower()

class Model():
    def __init__(self, model_type, decay_steps, num_classes=10):

        if 'cnn' in model_type:
            self.model = NetworkMNIST(model_type)
        elif 'residual' in model_type:
            self.model = NetworkMNIST(model_type)
        elif 'pde' in model_type:
            self.model = NetworkMNIST(model_type)
        
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        learning_rate_fn = tf.keras.experimental.CosineDecay(args.lr, decay_steps=decay_steps)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
        self.weight_decay = 5e-4
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions, f1, f2, g = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.loss_object(labels, predictions)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = ce_loss + l2_loss*self.weight_decay
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions, f1, f2, g = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def val_step(self, images, labels):
        predictions, f1, f2, g = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        plt.figure()
        plt.imshow( images[1, :, :, 0].numpy() )
        plt.figure()
        plt.imshow(  f1[1, :, :, 0].numpy() )
        plt.figure()
        plt.imshow(  g[1, :, :, 0].numpy() )
        plt.show()
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.model.summary()
        
    def train(self, train_ds, test_ds, epoch):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)  # start from epoch 0 or last checkpoint epoch
        ckpt_path = './checkpoints/{:s}/'.format(args.model)
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc,
                                   optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'

            # Restore the weights
            ckpt.restore(manager.latest_checkpoint)
        
        for e in tqdm(range(int(curr_epoch), epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels)
                
            for images, labels in test_ds:
                self.test_step(images, labels)

            template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
            print (template.format(e+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))
            
            # Save checkpoint
            if self.test_accuracy.result() > best_acc:
                print('Saving...')
                if not os.path.isdir('./checkpoints/'):
                    os.mkdir('./checkpoints/')
                if not os.path.isdir(ckpt_path):
                    os.mkdir(ckpt_path)
                best_acc.assign(self.test_accuracy.result())
                curr_epoch.assign(e+1)
                manager.save()
    
    def predict(self, pred_ds, best):
        if best:
            ckpt_path = './checkpoints/{:s}/'.format(args.model)
            ckpt = tf.train.Checkpoint(model=self.model)
            manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
            
            # Load checkpoint
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
            ckpt.restore(manager.latest_checkpoint)
        
        self.test_accuracy.reset_states()
        for images, labels in pred_ds:
            # self.test_step(images, labels)
            self.val_step(images, labels)
        print ('Prediction Accuracy: {:.2f}%'.format(self.test_accuracy.result()*100))

def main():
    # Data
    print('==> Preparing data...')
    train_images, train_labels, test_images, test_labels = get_dataset()
    mean, std = get_mean_and_std(train_images)
    train_images = normalize(train_images, mean, std)
    test_images = normalize(test_images, mean, std)

    train_ds = dataset_generator(train_images, train_labels, args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
            batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    decay_steps = int(args.epoch*len(train_images)/args.batch_size)
    
    # Train
    print('==> Building model...')
    model = Model(args.model, decay_steps)
    model.train(train_ds, test_ds, args.epoch)
    
    # Evaluate
    model.predict(test_ds.take(5), best=True)
    
    
if __name__ == "__main__":
    main()
