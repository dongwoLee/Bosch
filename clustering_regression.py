import os
import csv, sys, getopt
import numpy as np
import tensorflow as tf


def file_list(dirname):
    filenames = os.listdir(dirname)
    test_list = []
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
		if (row[(len(row) - 1)] == '0'):
                    y_vec = [1, 0]
                else:
                    y_vec = [0, 1]
		list_y.append(y_vec)
            else:
                xls = row[1:len(row)]
		list_y.append(row[0])
            for i in range(0, len(xls)):
                if (len(xls[i]) == 0):
                    xls[i] = '0.0'
            list_x.append(np.float32(xls))
    return np.array(list_x), np.array(list_y)

input_path = "missing_value_cluster/"
mini_batch_size = 100
num_iteration = 100
learning_rate = 0.02
myopts, args = getopt.getopt(sys.argv[1:], "i:m:r:l:")
for o, a in myopts:
    if o == '-i':
        input_path = a
    elif o == '-m':
        mini_batch_size = np.int(a)
    elif o == '-r':
        num_iteration = np.int(a)
    elif o == '-l':
        learning_rate = np.float32(a)

input_test = "changed_test/test_numeric_changed_cluster"
input_train = "changed_train/train_numeric_changed_cluster"
test_list = file_list("changed_test/")
with open("result.csv", "w") as outfile:
    wr = csv.writer(outfile)
    wr.writerow(["Id", "Response"])
for i in test_list:
    test_ids = []
    test_y = []
    if not os.path.exists(input_train + i):
        test_path = input_test + i
        test_x, test_ids = read_input(test_path,False)
        for j in test_ids:
            test_y.append(0)
    elif i == "1.csv":
        test_path = input_test + i
        test_x, test_ids = read_input(test_path,False)
        for j in test_ids:
            test_y.append(0)
    else:
        train_path = input_train + i
        test_path = input_test + i
        print train_path
	train_x, train_y = read_input(train_path, True)
        test_x, test_ids = read_input(test_path, False)
        num_features = train_x.shape[1]
        num_entries = train_y.shape[0]
        if num_entries < 3:
            for j in test_ids:
                test_y.append(0)
        else:
          with tf.Graph().as_default():
            W = tf.Variable(tf.random_normal([num_features, 2], stddev=0.01))
            X = tf.placeholder("float", [None, num_features], name='X')
            Y = tf.placeholder("float", [None, 2], name='Y')
            b = tf.Variable(tf.zeros([2]))

            Ye = tf.add(tf.matmul(X, W),b)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ye, Y))
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                for i in range(num_iteration):
                    train_error = 0.0
                    for j in range(0, num_entries, mini_batch_size):
                        X_ = train_x[j:j + mini_batch_size]
                        Y_ = train_y[j:j + mini_batch_size]
                        sess.run(train_op, feed_dict={X: X_, Y: Y_})
                        train_error = train_error + sess.run(cost, feed_dict={X: X_, Y: Y_})
                    if i % 100 == 0:
                        print("index: %d train_error : %f" % (i, train_error))
                X_ = test_x
                Y_expected = sess.run(Ye, feed_dict={X: X_})
                test_y = np.argmax(Y_expected, 1)
          tf.reset_default_graph()      
    with open("result.csv", "a+") as outfile:
        for i in range(0,len(test_ids)):
            wr = csv.writer(outfile)
            wr.writerow([test_ids[i],test_y[i]])
