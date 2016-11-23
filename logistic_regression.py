#!/usr/bin/python

import csv, sys, getopt
import numpy as np
import tensorflow as tf

def read_input(path, balance_input, train_portion):
  with open(path, "r") as infile:
    reader = csv.reader(infile)
    next(reader, None) # skip headers
    train_y = []
    train_x = []
    test_x = []
    test_y = []
    current_response = '0'
    for row in reader:
      xls = row[1:len(row)-1]
      yls = row[len(row)-1]
      if balance_output and yls == current_response:
        continue
      current_response = yls
      for i in range(0,len(xls)) :
        if(len(xls[i]) == 0) :
          xls[i] = '0.0'
      y_vec = [1, 0] if (row[(len(row)-1)] == '0') else [0, 1]
      if (np.random.random_sample() < train_portion):
        train_x.append(np.float32(xls))
        train_y.append(y_vec)
      else:
        test_x.append(np.float32(xls))
        test_y.append(y_vec)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def calculate_mcc(expected, real):
  tn = 0
  tp = 0
  fn = 0
  fp = 0
  for i in range(len(expected)):
    if(expected[i] == 0 and real[i] == 0):
      tn = tn + 1
    elif(expected[i] == 1 and real[i] == 1):
      tp = tp + 1
    elif(expected[i] ==1 and real[i] == 0):
      fp = fp + 1
    elif(expected[i] == 0 and real[i] == 1):
      fn = fn + 1
  return(((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))

input_path = "dataset/train_numeric.sample"
mini_batch_size = 100
train_portion = 1.0
balance_output = False
num_iteration = 100
learning_rate = 0.02
myopts, args = getopt.getopt(sys.argv[1:],"i:m:b:t:r:l:")
for o, a in myopts:
  if o == '-i':
    input_path = a
  elif o == '-m':
    mini_batch_size = np.int(a)
  elif o == '-b':
    balance_output = True
  elif o == '-t':
    train_portion = np.float32(a)
  elif o == '-r':
    num_iteration = np.int(a)
  elif o == '-l':
    learning_rate = np.float32(a)

train_x, train_y, test_x, test_y = read_input(input_path, balance_output, train_portion)
num_features = train_x.shape[1]
num_entries = train_y.shape[0]

W = tf.Variable(tf.random_normal([num_features, 2], stddev=0.01))
X = tf.placeholder("float", [None, num_features], name='X')
Y = tf.placeholder("float", [None, 2], name='Y')
b = tf.Variable(tf.zeros([2]))

Ye = tf.matmul(X, W)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ye, Y))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(Ye,1), tf.argmax(Y,1))
all_correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  tf.initialize_all_variables().run()
  for i in range(num_iteration):
    train_error = 0.0
    expected = []
    real = []
    for j in range(0, num_entries, mini_batch_size) :
      X_ = train_x[j:j+mini_batch_size]
      Y_ = train_y[j:j+mini_batch_size]
      sess.run(train_op, feed_dict={X: X_, Y: Y_})
      train_error = train_error + sess.run(cost, feed_dict={X: X_, Y: Y_})    
    if i%100 == 0:
      test_entries_x = train_x
      test_entries_y = train_y
      num_test_entries = train_y.shape[0]
      acc = 0.0
      expected_ones = 0
      correct_predicts = 0
      test_ones = 0
      for a in range(num_test_entries):
        if np.equal(test_entries_y[a], [0,1]).all() == True:
          test_ones = test_ones + 1
      for j in range(0, test_entries_x.shape[0], mini_batch_size) :
        X_ = test_entries_x[j:j+mini_batch_size]
        Y_ = test_entries_y[j:j+mini_batch_size]
        correct_predicts = correct_predicts + np.sum(sess.run(all_correct_sum, feed_dict={X: X_, Y: Y_}))
        Y_expected = sess.run(Ye, feed_dict={X:X_})
        expected.extend(np.argmax(Y_expected, 1))
        real.extend(np.argmax(Y_, 1))
        expected_ones = expected_ones + np.sum(np.argmax(Y_expected, 1))
      mcc = calculate_mcc(expected, real)
      print("index: %d train_error : %f mcc: %f accuracy: %f expected ones:%d real ones: %d test cases: %d"%(i, train_error, mcc, (correct_predicts/num_test_entries), expected_ones, test_ones, num_test_entries))
