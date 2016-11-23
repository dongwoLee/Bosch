import os
import csv, sys, getopt
import numpy as np
import tensorflow as tf

def file_list(dirname):
    filenames = os.listdir(dirname)
    test_list = []
    cnt = 0
    for f in filenames:
        if f[0:4] == 'test':
            test_list.append(f[28:len(f)])
    return test_list

def read_input(path,is_train):
    with open(path, "r") as infile:
        reader = csv.reader(infile)
        list_x = []
        list_y = []
        for row in reader:
            if is_train:
                xls = row[1:len(row) - 1]
            else:
                xls = row[1:len(row)]
            for i in range(0, len(xls)):
                if (len(xls[i]) == 0):
                    xls[i] = '0.0'
            list_x.append(np.float32(xls))
            if is_train:
		y_vec = [1, 0] if (row[(len(row) - 1)] == '0') else [0, 1]
                list_y.append(y_vec)
            else:
                list_y.append(row[0])
    return np.array(list_x), np.array(list_y)

input_path = "missing_value_cluster/"
mini_batch_size = 100
num_hidden_node = 500
num_iteration = 100
learning_rate = 0.02
myopts, args = getopt.getopt(sys.argv[1:],"i:m:h:r:l:")
for o, a in myopts:
  if o == '-i':
    input_path = a
  elif o == '-m':
    mini_batch_size = np.int(a)
  elif o == '-h':
    num_hidden_node = np.int(a)
  elif o == '-r':
    num_iteration = np.int(a)
  elif o == '-l':
    learning_rate = np.float32(a)

input_train = "changed_train/train_numeric_changed_cluster"
input_test = "changed_test/test_numeric_changed_cluster"
test_list = file_list("changed_test/")
with open("result.csv", "w") as outfile:
    wr = csv.writer(outfile)
    wr.writerow(["id", "Response"])
for i in test_list:
    test_ids = []
    test_y = []
    if not os.path.exists(input_train + i):
        test_path = input_test + i
        test_x, test_ids = read_input(test_path,False)
        for j in test_ids:
            test_y.append(0)
    else:
        train_path = input_train + i
        test_path = input_test + i
        train_x, train_y = read_input(train_path, True)
        test_x, test_ids = read_input(test_path, False)
        num_features = train_x.shape[1]
        num_entries = train_y.shape[0]
        print(train_x.shape, test_x.shape)
	print i
	if train_x.shape[1] == 0:
	    for j in test_ids:
                test_y.append(0)
	else:
          with tf.Graph().as_default():
            X = tf.placeholder("float", [None, num_features], name='X')
            Y = tf.placeholder("float", [None, 2], name='Y')
            W1_init = tf.random_uniform(shape=[num_features, num_hidden_node], minval=-2.0, maxval=2.0)
            W1 = tf.Variable(W1_init, name='W1')
            b1=tf.Variable(tf.zeros([num_hidden_node]), name='b1')
            #H1=tf.nn.sigmoid(tf.matmul(X, W))
            H1=tf.nn.relu(tf.add(tf.matmul(X, W1) , b1))

            W2_init = tf.random_uniform(shape=[num_hidden_node, num_hidden_node], minval=-2.0, maxval=2.0)
            W2 = tf.Variable(W2_init, name='W2')
            b2=tf.Variable(tf.zeros([num_hidden_node]), name='b2')
            H2=tf.nn.sigmoid(tf.add(tf.matmul(H1, W2), b2))

            W3_init = tf.random_uniform(shape=[num_hidden_node, 2], minval=-2.0, maxval=2.0)
            W3 = tf.Variable(W3_init, name='W3')
            b3 = tf.Variable(tf.zeros([2]), name = 'b3')

            #H3 = tf.add(tf.matmul(H2, W3), b3)
            H3 = tf.add(tf.matmul(H1, W3), b3)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H3, Y))
            #cost = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H3, Y)), tf.mul(0.0001, tf.reduce_sum(tf.square(W1))))
            #cost = tf.reduce_mean(tf.square(H3 - Y))
            #cost = tf.add(tf.add(tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H3, Y)), tf.reduce_sum(tf.abs(W1))), tf.reduce_sum(tf.abs(W2))), tf.reduce_sum(tf.abs(W3)))
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                for i in range(num_iteration) :
                    train_error = 0.0
                    for j in range(0, num_entries, mini_batch_size) :
                        X_ = train_x[j:j+mini_batch_size]
                        Y_ = train_y[j:j+mini_batch_size]
                        sess.run(train_op, feed_dict={X: X_, Y: Y_})
                        train_error = train_error + sess.run(cost, feed_dict={X: X_, Y: Y_})  
                    if i % 10 == 0: 
		        print("index: %d train_error : %f" % (i, train_error))
                X_ = test_x
                Y_expected = sess.run(H3, feed_dict={X:X_})
                test_y = np.argmax(Y_expected, 1)
          tf.reset_default_graph()
    with open("result.csv", "a+") as outfile:
        for i in range(0,len(test_ids)):
            wr = csv.writer(outfile)
            wr.writerow([test_ids[i],test_y[i]])
