'''
Created on 24.02.2018

@author: micha
'''


# import pickle
# import csv
# from sklearn.utils import shuffle 
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.contrib.layers import flatten
# import tensorflow as tf
# 
# import time
# import math
# import datetime
# 
# 
# import scipy.ndimage  
# import scipy.misc

import random
import sklearn
import numpy as np
import tensorflow as tf
import os
import scipy.ndimage  
import scipy.misc
import skimage
from skimage import color

from collections import Counter

# hello_constant = tf.constant("Hello World")
# 
# with tf.Session() as sess:
#     output = sess.run(hello_constant)
#     print (output)
 
def Lenet(features, keep_prob):
    mu = 0.
    sigma = 0.1
    
    
    conv1W       = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean = mu, stddev = sigma), name='conv1_W')
    conv1B       = tf.Variable(tf.zeros(32), name = 'conv1_B')
    conv1        = tf.nn.conv2d(features, conv1W, (1,1,1,1), padding='VALID') + conv1B
    conv1        = tf.nn.relu(conv1)
    conv1        = tf.nn.max_pool(conv1, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
    
    conv2W       = tf.Variable(tf.truncated_normal(shape=(5,5,32,43), mean = mu, stddev = sigma), name='conv2_W')
    conv2B       = tf.Variable(tf.zeros(43), name = 'conv2_B')
    conv2        = tf.nn.conv2d(conv1, conv2W, (1,1,1,1), padding='VALID') + conv2B
    conv2        = tf.nn.relu(conv2)
    conv2        = tf.nn.max_pool(conv2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

    fc0 = tf.contrib.layers.flatten(conv2)
    noOut = int(fc0.get_shape()[1])
    #84
    fc1Cnt = 84
    fc1W = tf.Variable(tf.truncated_normal(shape=(noOut, fc1Cnt), mean = mu, stddev = sigma), name='fc1_W')
    fc1B = tf.Variable(tf.zeros(fc1Cnt), name='fc1_b')
    
    fc1 = tf.matmul(fc0, fc1W,) + fc1B
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    #42
    fc2Cnt = 42
    fc2W = tf.Variable(tf.truncated_normal(shape=(fc1Cnt, fc2Cnt), mean = mu, stddev = sigma), name='fc2_W')
    fc2B = tf.Variable(tf.zeros(fc2Cnt), name='fc2_b')
    
    fc2    = tf.matmul(fc1, fc2W) + fc2B
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob)
    
    fc3Cnt = 21
    fc3W = tf.Variable(tf.truncated_normal(shape=(fc2Cnt, fc3Cnt), mean = mu, stddev = sigma), name='fc3_W')
    fc3B = tf.Variable(tf.zeros(fc3Cnt), name='fc3_b')

    fc3    = tf.matmul(fc2, fc3W) + fc3B
    fc3    = tf.nn.relu(fc3)
    fc3    = tf.nn.dropout(fc3, keep_prob)
    
    fc4Cnt = 3
    fc4W = tf.Variable(tf.truncated_normal(shape=(fc3Cnt, fc4Cnt), mean = mu, stddev = sigma), name='fc4_W')
    fc4B = tf.Variable(tf.zeros(fc4Cnt), name='fc4_b')
    
    logits = tf.matmul(fc3, fc4W) + fc4B
    return logits

def importCustomImages(filepath):
    resultingImages = [[], [], [] ]
    data = [ [], [], [] ]
    redCnt = 0
    yelCnt = 0
    greCnt = 0
    for subdir, dir, files in os.walk(filepath):
        for file in files :
            if file.endswith((".png", ".jpg", ".jpeg")):
                fqp = os.path.join(subdir, file)
                if -1 != fqp.find("red"):
                    data[0].append((fqp, 0))
                    redCnt += 1
                elif -1 != fqp.find("yellow"):
                    data[1].append( (fqp, 1) )
                    yelCnt += 1
                elif -1 != fqp.find("green"):
                    data[2].append( (fqp, 2) )
                    greCnt +=1
    
    for i in range(3):
        for file in data[i]:
            #resize to 32/32 px and remove alphachannel if availble as well
            resultingImages[i].append( (np.array(scipy.misc.imresize(scipy.misc.imread(file[0]), (32,32))[:,:,:3], dtype=np.float32)
                                        , file[1]) )
    return resultingImages

def normalizeZeroMeanData(dataarray):
    count = 0
    for data in dataarray:
#         if count == 0:
#             print("Lala" ,data[0])
#         print("Orig" ,data[0])
#         data[0] -= 128
#         print("Minus", data[0])
#         data[0] = data[0].astype(np.float32) / 128.
#         print("Div", data[0])
#         exit(0)
        data[0] -= 128.
#         data[0][0] = data[0][0].astype(np.float32) / 128.
        data[0] /= 128.#data[0][0].astype(np.float32) / 128.
#         if count == 0:
#             print("Lulu",data[0])
#             count += 1
    return dataarray

def dataNormalizeCnts(val):
    val.sort(key = lambda x : len(x), reverse=True)
    print(len(val[0]))
    print(len(val[1]))
    print(len(val[2]))
    fillUpTo = len(val[0])
    for i in range (1, len(val)):
        while len(val[i]) < fillUpTo:
            val[i].append(random.choice(val[i]))
    return val

def dataAugmentation(dataarray):
    resultingData = []
    #each image is now
    #turned for 15 degreed to the right and to the left
    #shifted by 2 px to the bottom right / top left
    #for each of this images, the x axis is turned
    for data in dataarray:
        img = data[0]
        label = data[1]
        resultingData.append(data)
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.rotate(img, 15., reshape=False, mode='nearest'), label))
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.rotate(img, 90., reshape=False, mode='nearest'), label))
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.shift(img,  (2., 2., 0.), mode='nearest'), label) )
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.rotate(img, -15., reshape=False, mode='nearest'), label))
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.rotate(img, -90., reshape=False, mode='nearest'), label))
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
        resultingData.append( (scipy.ndimage.interpolation.shift(img,  (-2., -2., 0.), mode='nearest'), label) )
        resultingData.append( (np.fliplr(resultingData[-1][0]), label) )
    return resultingData


def loadCNNAndVerify(filepath, val):
    batchsize = 128
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.constant(1.)
    logits = Lenet(x, keep_prob)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y, 3), 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    total_accuracy = 0.
    
    X = np.array(val)[...,0].tolist()
    Y = np.array(val)[...,1].tolist()
    
    num_examples = len(X)
    
    
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, filepath)
        
        for offset in range(0, num_examples, batchsize):
            batch_x, batch_y = X[offset:offset+batchsize], Y[offset:offset+batchsize]
            accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y})
            total_accuracy += (accuracy * len(batch_x))
        
    print("Total accuracy of {0} samples is {1:.3f}".format(num_examples, total_accuracy / num_examples))
        
        

def trainCNN(data):
    rate = 0.0005
    epochs = 30
    batchsize = 128
    keep_probability = 0.6
    
    input_layer  = tf.placeholder(tf.float32, (None, 32, 32, 3))
    inputY = tf.placeholder(tf.int32, (None))
    labels = tf.one_hot(inputY, 3)
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Lenet(input_layer, keep_prob)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf. global_variables_initializer()
    
    
    in_samples, in_labels = sklearn.utils.shuffle(np.array(data)[...,0],np.array(data)[...,1])
    
    noTraining = int(len(in_samples) * 0.52)
    noValidation = int(len(in_samples) * 0.12)
    noTest = int(len(in_samples) * 0.36)
    
    
    
    training_x = in_samples[:noTraining].tolist()
    
    
    training_y = in_labels[:noTraining].tolist()
    red= 0
    yel= 0
    gre= 0
    for ele in training_y:
      if ele == 0:
        red += 1
      elif ele == 1:
        yel += 1
      else:
        gre +=1  
    
    print("Distribution {0} {1} {2}".format(red, yel, gre))
    validation_x = in_samples[noTraining:noTraining+noValidation].tolist()
    validation_y = in_labels[noTraining:noTraining+noValidation].tolist()
    test_x = in_samples[noTraining+noValidation:].tolist()
    test_y = in_labels[noTraining+noValidation:].tolist()
    
    print("Training {0} Validation {1} Test {2} - {3}".format(noTraining, noValidation, noTest, len(data)), str(type(training_x)))
    print(training_x[0].shape)
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
 
        for epoch in range(epochs):
            stopLoop = False
            batchIdx = 0
            training_x, training_y = sklearn.utils.shuffle(training_x, training_y)
            while(not stopLoop):
                start = batchIdx * batchsize
                end = min((batchIdx+1) * batchsize, len(training_x))
                batch_x = training_x[start: end]
                batch_y = training_y[start:end]
                stopLoop = end==len(training_x)
                sess.run(optimizer, feed_dict={
                    input_layer: batch_x,
                    inputY: batch_y,
                    keep_prob: keep_probability})
     
                # Calculate batch loss and accuracy
#                 loss = sess.run(cost, feed_dict={
#                     input_layer: batch_x,
#                     labels: batch_y,
#                     keep_prob: 1.})
#                 valid_acc = sess.run(accuracy, feed_dict={
#                     input_layer: validation_x[:],
#                     labels: validation_y[:],
#                     keep_prob : 1.})
#      
#                 print('Epoch {:>2}, Batch {:>3} -'
#                       'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
#                     epoch + 1,
#                     batchIdx + 1,
#                     loss,
#                     valid_acc))
                batchIdx += 1
            valid_acc = sess.run(accuracy, feed_dict={
                input_layer: validation_x[:],
                inputY: validation_y[:],
                keep_prob : 1.})
            print('Epoch {:>2}, Validation Accuracy: {:.6f}'.format(epoch+1, valid_acc))
     
        # Calculate Test Accuracy
        test_acc = sess.run(accuracy, feed_dict={
            input_layer: test_x[:],
            inputY: test_y[:],
            keep_prob: 1.})
        print('Testing Accuracy: {}'.format(test_acc))
        tf.train.Saver().save(sess, 'E:/tensorflow/tensor{:.3f}'.format(test_acc))
    
    
class TLClassifier(object):
    def __init__(self, filepath):
#         self.session = tf.InteractiveSession()
        self.path = filepath
        
        self.session = None 
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.rate = tf.constant(1.)
        self.classifier = tf.argmax(Lenet(self.x, self.rate), 1)
        
        
#         tf.train.Saver().restore(self.session, filepath)
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.session = tf.Session(graph = self.graph)
#             with self.session as sess:
#                 tf.train.Saver().restore(sess, filepath)
# #             with tf.Session() as sess:
# #                 tf.train.Saver().restore(sess, filepath)
#                 self.defaultSession = sess
#         self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
#         self.classifier = tf.argmax(Lenet(self.x, tf.constant(1.)))
                
    def classifyImageFromPath(self, path):
        return self.classifyImage(scipy.misc.imread(path))
    
    def classifyImage(self, img):
        ##resize
        image = np.array(scipy.misc.imresize(img, (32,32))[:,:,:3], dtype=np.float32)
        ##normalize
#         image -= 128.
#         image = image.astype(np.float32) / 128.
        image = [image]
        
#         x = tf.placeholder(tf.float32, (None, 32, 32, 3))
#         rate = tf.constant(1.)
#         classifier = tf.argmax(Lenet(x, rate), 1)


#Interactive
        if self.session is None:
            self.session = tf.InteractiveSession()
            tf.train.Saver().restore(self.session, self.path)
        return self.session.run(self.classifier, feed_dict={self.x:image})[0]
     
    
    
#         with tf.Session() as sess:
#             tf.train.Saver().restore(sess, self.path)
#             result = sess.run(classifier, feed_dict={x:image})
#             return result[0]
            
    
if __name__ == '__main__':
#     tf.app.run()
    train = False
    verify = True
    if train:
        val = importCustomImages("C:/Users/micha/Desktop/Udacity/Last/training")
#     scipy.misc.imsave('E:/rot{0}.png'.format(val[0][0][1]), val[0][0][0])
#     scipy.misc.imsave('E:/gelb{0}.png'.format(val[1][0][1]), val[1][0][0] )
#     scipy.misc.imsave('E:/gruen{0}.png'.format(val[2][0][1]), val[2][0][0])
#     print("Image count red {0}, yellow {1} and green {2}".format(len(val[0]), len(val[1]), len(val[2])))
        val = dataNormalizeCnts(val)
#     print("Image count red {0}, yellow {1} and green {2}".format(len(val[0]), len(val[1]), len(val[2])))
#         print("lala", val[0][0][0])
        val = np.concatenate(val)
#         print("lulu", val[0][0])
#         val = normalizeZeroMeanData(val)
        val = dataAugmentation(val)
#         print("lulu", val[0][0])
#         exit(0)
        trainCNN(val)
        exit(0)
# # 
#     exit(0)
    filepath='E:/tensorflow/tensor0.999'
    if verify:
        val = importCustomImages("C:/Users/micha/Desktop/Udacity/Last/training/sim_basti2")
        val = np.concatenate(val)
        loadCNNAndVerify(filepath, val)
        exit(0)
#     scipy.misc.imsave('R:/test.png', val[0][0][0])
#     scipy.misc.imsave('R:/test2.png', scipy.ndimage.interpolation.shift(val[0][0][0],  (2., 2., 0.), mode='nearest'))
#     scipy.misc.imsave('R:/test3.png', np.fliplr(val[0][0][0]))
# #     scipy.misc.imsave('R:/test4.png', color.rgb2hsv(val[0][0][0]))
#     scipy.misc.imsave('R:/test5.png', val[0][0][0])
    test = TLClassifier(filepath)
    color = [ 'RED', 'YELLOW', 'GREEN']
#     c = test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/img_20180207-0401380.png')
#     print( color[c])
    

# C:/Users/micha/Desktop/Udacity/Last/training/sim_samples/green
# C:/Users/micha/Desktop/Udacity/Last/training/sim_samples/red
# C:/Users/micha/Desktop/Udacity/Last/training/sim_samples/yellow
# C:/Users/micha/Desktop/Udacity/Last/training/green
# C:/Users/micha/Desktop/Udacity/Last/training/red
# C:/Users/micha/Desktop/Udacity/Last/training/yellow
    cnt = 0    
    for subdir, dir, files in os.walk('C:/Users/micha/Desktop/Udacity/Last/training/yellow'):
        for file in files :
            if file.endswith((".png", ".jpg", ".jpeg")):
                print(color[test.classifyImageFromPath(os.path.join(subdir, file))])
                cnt += 1
    print( "Total "+str(cnt))
#     print(test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/ecd9d40f-23fb-49db-acd8-ec50384fad59.jpg'))
#     print(test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/c214279e-7a8c-462b-a0a9-2f6c14eeb1bd.jpg'))
#     print(test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/f32f4884-7024-44c9-92d5-538e27406ac9.jpg'))
#     print(color[test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/ecd9d40f-23fb-49db-acd8-ec50384fad59.jpg')])
#     print(color[test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/c214279e-7a8c-462b-a0a9-2f6c14eeb1bd.jpg')])
#     print(color[test.classifyImageFromPath('C:/Users/micha/Desktop/Udacity/Last/training/samples/training/yellow/f32f4884-7024-44c9-92d5-538e27406ac9.jpg')])
    
