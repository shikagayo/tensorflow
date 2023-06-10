import tensorflow as tf

my_graph = tf.Graph()

with tf.compat.v1.Session(graph = my_graph) as sess:
    x = tf.constant([2, 20, 15, 41, 6])
    y = tf.constant([4, 1, 19, 21, 11])

    op = tf.add(x,y)

    x1 = tf.constant([16, 19, 34, 20, 7])
    y1 = tf.constant([41, 25, 17, 9, 40])

    op1 = tf.maximum(x1,y1)

    x2 = tf.constant([71, 20, 44, 9, 18])
    y2 = tf.constant([14, 27, 88, 7, 90])

    op2 = tf.minimum(x2,y2)

    result = sess.run(fetches = op)
    result1 = sess.run(fetches = op1)
    result2 = sess.run(fetches = op2)
    print(result)
    print(result1)
    print(result2)



    