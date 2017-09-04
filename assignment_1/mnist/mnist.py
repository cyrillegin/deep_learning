import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

startTime = time.time()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Although the tutorial says to use this version of cross_entropy, it provides a worse result by about 2%
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 0.5 is the learning rate - Doesn't reeally effect time but can really effect accuracy 
train_step = tf.train.GradientDescentOptimizer(1.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# run training step 1000 times
for _ in range(1000):  # increases time and accuracy
    batch_xs, batch_ys = mnist.train.next_batch(100)  # increases time and accuracy
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Results:"
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print "Took {} seconds".format(time.time() - startTime)
