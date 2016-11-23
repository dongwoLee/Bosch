#!/usr/bin/python

import csv
import getopt, sys
import operator

def get_key(row):
  key=""
  for e in row:
    if e == "":
      key = key + "0"
    else:
      key = key + "1"
  return key

def read_input(path,is_train=True):
  with open(path, "r") as infile:
    reader = csv.reader(infile)
    next(reader, None) # skip headers
    key_list = {}
    for row in reader:
      if is_train :
        key = get_key(row[1:len(row)-1])
      else :
        key = get_key(row[1:len(row)])
      ids = key_list.get(key, [])
      ids.append(row[0])
      key_list[key] = ids
  return key_list


input_train = "dataset/train_numeric.csv"
input_test = "dataset/test_numeric.csv"

myopts, args = getopt.getopt(sys.argv[1:],"tr:te:")
for o, a in myopts:
  if o == '-tr':
    input_train = a
  elif o == '-te':
    input_test = a

kl_train = read_input(input_train,True)
kl_test = read_input(input_test,False)

id_index = 0
train_file_index = {}
test_file_index = {}
sort_dic={}

for k, v in kl_train.iteritems():
  for i in v:
    train_file_index[str(i)] = "train_id-"+str(id_index)
  sort_dic["id-" + str(id_index)] = [len(kl_train[str(k)])]
  if k in kl_test :
    for i in kl_test.get(k) :
      test_file_index[str(i)] = "test_id-" + str(id_index)
    sort_dic["id-"+str(id_index)].append(len(kl_test[str(k)]))
  else:
    sort_dic["id-" + str(id_index)].append(0)
  id_index = id_index + 1

for k, v in kl_test.iteritems():
   if k not in kl_train :
     for i in v :
       test_file_index[str(i)] = "test_id-" + str(id_index)
     sort_dic["id-" + str(id_index)] = [0,len(kl_test[str(k)])]
     id_index = id_index + 1

sort_dic = sorted(sort_dic.items(),key=operator.itemgetter(1))

with open('dict.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for i in sort_dic:
       writer.writerow(i)

with open(input_train, "r") as infile:
  reader = csv.reader(infile)
  next(reader, None) # skip headers
  for row in reader:
    file_name = train_file_index[row[0]]
    with open("missing_value_cluster/"+file_name+".csv", "a+") as outfile:
      wr = csv.writer(outfile)
      wr.writerow(filter(None, row))

with open(input_test, "r") as infile:
  reader = csv.reader(infile)
  next(reader, None) # skip headers
  for row in reader:
    file_name = test_file_index[row[0]]
    with open("missing_value_cluster/"+file_name+".csv", "a+") as outfile:
      wr = csv.writer(outfile)
      wr.writerow(filter(None, row))
