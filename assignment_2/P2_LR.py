# had to add this - RL
import sys
import os
import input_data
import tensorflow as tf
import shutil
import matplotlib.pyplot as pyplt
from random import randint
import numpy as np
from multilayer_perceptron import inference, loss
from scipy.ndimage import interpolation
import pandas as pd
sys.path.append('../../')
sys.path.append('../')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 100
display_step = 1

randomScale = True


def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 10], initializer=init)
    b = tf.get_variable("b", [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    # w_hist = tf.summary.histogram("weights", W)
    # b_hist = tf.summary.histogram("biases", b)
    # y_hist = tf.summary.histogram("output", output)

    return output


def displayImage(imageSet):
    if imageSet is None:
        print "err"
        return
    for i in range(0, 10):
        nextImage = False
        while nextImage is False:
            labelIndex = randint(0, len(imageSet) - 1)
            if mnist.train.labels[labelIndex][i] == 1:
                image = imageSet[labelIndex]
                image = np.array(image, dtype='float')
                data = image.reshape((28, 28))
                pyplt.figure()
                pyplt.imshow(data, cmap='gnuplot')
                nextImage = True
    pyplt.show()

    # Now Graph the weights
    # for i in range (0, 10):
    #
    #             image = y.shape[i][i]
    #             image = np.array(image, dtype = 'float32')
    #             data = image.reshape((28,28))
    #             pyplt.figure()
    #             pyplt.imshow(data, cmap = 'gnuplot')
    #             nextImage = True
    # pyplt.show()


def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row
    # into a single value. In general, reduction along axis i
    # collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, axis=1)

    loss = tf.reduce_mean(xentropy)

    return loss


def training(cost, global_step):

    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


# Adapted from:
# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = interpolation.zoom(img, zoom_tuple, **kwargs)

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

if __name__ == '__main__':
    if os.path.exists("logistic_logs/"):
        shutil.rmtree("logistic_logs/")

    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28=784
        y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

        output = inference(x)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = training(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("logistic_logs/", graph_def=sess.graph_def)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

                if randomScale:
                    # print 'scaling'
                    ref = np.zeros_like(minibatch_x)
                    # minibatch_x = interpolation.zoom(minibatch_x, 1)
                    minibatch_x = clipped_zoom(minibatch_x, 0.5)
                    # image = np.array(minibatch_x[0], dtype='float')
                    # data = image.reshape(28,28)
                    # pyplt.figure()
                    # pyplt.imshow(data, cmap='gnuplot')
                    # nextImage = True
                    # pyplt.show()

                    # imresize(minibatch_x, 2)
                    # print 'done scaling'
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

                print("Validation Error:", (1 - accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)

        print("Optimization Finished!")

        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print("Test Accuracy:", accuracy)

        print "Confusion Matrix:"
        with sess.as_default():
            res = tf.stack([tf.argmax(y, 1), tf.argmax(y, 1)])
            ans = res.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
            confusion = np.zeros([10, 10], int)

            for p in ans.T:
                confusion[p[0], p[1]] += 1
            print(pd.DataFrame(confusion))

    displayImage(minibatch_x)
    print ("done")
