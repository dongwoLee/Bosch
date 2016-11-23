#!/usr/bin/python

import csv
import getopt, sys


def get_key(row):
  key=""
  for e in row:
    if e == "":
      key = key + "0"
    else:
      key = key + "1"
  return key


def read_input(path):
  with open(path, "r") as infile:
    reader = csv.reader(infile)
    next(reader, None) # skip headers
    key_list = {}
    for row in reader:
      key = get_key(row[1:len(row)-1])
      ids = key_list.get(key, [])
      ids.append(row[0])
      key_list[key] = ids
  return key_list

input_path = "dataset/train_numeric.sample"
#input_path = "train_numeric.csv"

myopts, args = getopt.getopt(sys.argv[1:],"i:")
for o, a in myopts:
  if o == '-i':
    input_path = a

kl = read_input(input_path)

id_index = 0
id_to_file_index = {}
for k, v in kl.iteritems():
  for i in v:
    id_to_file_index[str(i)] = "id-"+str(id_index)
  id_index = id_index + 1

with open(input_path, "r") as infile:
  reader = csv.reader(infile)
  next(reader, None)  # skip headers
  for row in reader:
    file_name = id_to_file_index[row[0]]
    with open("missing_value_cluster/"+file_name+".csv", "a+") as outfile:
      wr = csv.writer(outfile)
      wr.writerow(filter(None,row))

