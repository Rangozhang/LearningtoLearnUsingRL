import numpy as np
import tensorflow as tf

x = tf.Variable(0.5)
y = x*x
opt = tf.train.AdagradOptimizer(1)
grads = opt.compute_gradients(y, [x])
grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
apply_placeholder_op = opt.apply_gradients(grad_placeholder)
transform_grads = [(grad[0], grad[1]) for grad in grads]
apply_transform_op = opt.apply_gradients(transform_grads)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

grad_vals = sess.run([grad[0] for grad in grads])
feed_dict = {}
for i in xrange(len(grad_placeholder)):
  feed_dict[grad_placeholder[i][0]] = grad_vals[i]
sess.run(apply_placeholder_op, feed_dict=feed_dict)
print grad_vals
print x.eval(session=sess)

sess.run(apply_transform_op)
print sess.run([grad[0] for grad in grads])
print x.eval(session=sess)

print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print [x]
