# coding=utf-8
import collections
import csv
from collections import defaultdict
from math import *

import numpy as np

reader = csv.reader(open('train.csv', 'r'))
dict_list = {}
count=0
for row in reader:
    k, v = row
    dict_list[k] = v  # Making Dictionary
d=defaultdict(list)
for k, v in dict_list.iteritems():
    original_seperate_pattern = k.split()
    number_of_id = original_seperate_pattern.count('1')
    d[k].append(v)  # value 를 넣고
    d[k].append(number_of_id)   # 거기에다가 id의 개수를 집어 넣었다
dict_list=dict(d)



def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    if denominator == 0:
        denominator = 0.0001
    return round(numerator / float(denominator), 3)


def f(a, N):
    return np.argsort(a)[::-1][:N]


ordered_dict_list = collections.OrderedDict(sorted(dict_list.items()))  # Sorting Dictionary
keys = ordered_dict_list.keys()  # Key Assigning
n = len(keys)  # Assigning Range

element_list = []  # List for skipping already clustered pattern

cluster_number = 1   # index for cluster_number
common_1_dict=collections.OrderedDict()

for i in range(n):
    cosine_list = []  # list for saving  cosine result by each pattern

    new_dict_for_clustering = collections.OrderedDict()

    cosine_exception = -3

    if i in element_list:
        continue

    thisKey = keys[i]

    original_seperate_pattern = thisKey.split()

    original_seperate_pattern_int = list(map(int, original_seperate_pattern))

    number_of_id = 0

    for j in range(n):

        nextKey = keys[j]

        if thisKey == nextKey:
            cosine_list.append(1)
            continue

        if j in element_list:
            cosine_list.append(cosine_exception)
            continue

        compare_seperate_pattern = nextKey.split()

        compare_seperate_pattern_int = list(map(int, compare_seperate_pattern))

        compute_cosine_similarity = cosine_similarity(original_seperate_pattern_int, compare_seperate_pattern_int)

        cosine_list.append(compute_cosine_similarity)

    b = f(cosine_list, n)     # sorting descending order



    for k in b:
        if cosine_list[k]== -3: continue
        new_dict_for_clustering[keys[k]] = ordered_dict_list[keys[k]]
        element_list.append(k)
        number_of_id = 0
        for item in new_dict_for_clustering.values():
            number_of_id += int(item[0])
        if number_of_id >= 10000: break

    keyList = new_dict_for_clustering.keys()

    test = keyList[0].split()

    for i, v in enumerate(keyList):

        if i == len(new_dict_for_clustering) - 1: break;

        test_next = keyList[i + 1].split()

        for m in range(len(test)):

            if (test_next[m]=='1' and test[m] == '1'):

                test[m] = '1'

            else:

                test[m] = '0'

    number_of_1 = test.count('1')   # 1의 개수 세주기

    common_1_dict[cluster_number]=test

    with open("train_cluster/" + 'cluster-' + str(cluster_number) + '-id-' + str(
            number_of_id) + '-number_of_value-' + str(number_of_1) + '.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in new_dict_for_clustering.items():
            writer.writerow([key, value])
    cluster_number += 1
with open('common_pattern_train.csv', 'wb') as csv_file1:
    writer = csv.writer(csv_file1)
    for key, value in common_1_dict.items():
        writer.writerow([key, value])
