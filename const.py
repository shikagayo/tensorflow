import tensorflow as tf
tf.compat.v1.disable_eager_execution()

first_graph = tf.compat.v1.get_default_graph()
with first_graph.as_default():
    c1 = tf.constant(1.0)
    c2 = tf.constant(101.0)
    c3 = tf.constant(12.0)

second_graph = tf.Graph()
with first_graph.as_default():
    c4 = tf.constant(11.0)
    c5 = tf.constant(10.0)
    c6 = tf.constant(16.0)

third_graph = tf.Graph()
with first_graph.as_default():
    c7 = tf.constant(21.0)
    c8 = tf.constant(15.0)
    c9 = tf.constant(31.0)

with tf.compat.v1.Session() as sess:
    print(c1.eval(session = sess))

sess = tf.compat.v1.Session()
print(c1.eval(session = sess))
sess.close

sess = tf.compat.v1.Session()
print(c2.eval(session = sess))
sess.close

sess = tf.compat.v1.Session()
print(c3.eval(session = sess))
sess.close