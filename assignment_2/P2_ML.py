# had to add this - RL
import input_data
import tensorflow as tf
import shutil
import os
from P2_LR import displayImage
from scipy.ndimage import rotate
import numpy as np
from random import randint
import pandas as pd

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 10000
display_step = 1

doRotation = True


def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [784, n_hidden_1], [n_hidden_1])

    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])

    with tf.variable_scope("output"):
        output = layer(hidden_2, [n_hidden_2, 10], [10])

    return output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
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
    tf.summary.scalar("validation", accuracy)
    return accuracy


if __name__ == '__main__':

    if os.path.exists("mlp_logs/"):
        shutil.rmtree("mlp_logs/")

    with tf.Graph().as_default():

        with tf.variable_scope("mlp_model"):

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

            summary_writer = tf.summary.FileWriter("mlp_logs/", graph_def=sess.graph_def)

            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            # saver.restore(sess, "mlp_logs/model-checkpoint-66000")

            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                    if doRotation:
                        print 'rotating'
                        minibatch_x = rotate(minibatch_x, randint(0, 360), reshape=False)
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

                    saver.save(sess, "mlp_logs/model-checkpoint", global_step=global_step)

            print("Optimization Finished!")
            accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})

            print "Confusion Matrix:"
            with sess.as_default():
                res = tf.stack([tf.argmax(y, 1), tf.argmax(y, 1)])
                ans = res.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
                confusion = np.zeros([10, 10], int)

                for p in ans.T:
                    confusion[p[0], p[1]] += 1
                print(pd.DataFrame(confusion))

            print("Test Accuracy:", accuracy)
            displayImage(minibatch_x)
