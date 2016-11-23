import collections
import csv
import glob

def get_key(row):
    key=""
    for e in row:
        if e == "":
            key = key + "0 "
        else:
            key = key + "1 "
    return key


path = "train_cluster/*"
dict_for_cid_patternlists = collections.OrderedDict()
for fname in glob.glob(path):
    reader = csv.reader(open(fname, 'r'))
    cluster_name=fname.split("-")
    for row1 in reader:
        k, v = row1
        patterns = dict_for_cid_patternlists.get(cluster_name[1], [])
        patterns.append(k)
        dict_for_cid_patternlists[cluster_name[1]] = patterns
dict_for_cid_patternlists["99"]="X"

train_dict_for_pid_cid=collections.OrderedDict()
test_dict_for_pid_cid=collections.OrderedDict()

with open("train_numeric.csv", "rb") as infile:
    reader1 = csv.reader(infile)
    next(reader1, None) # skip headers
    for row in reader1:
        key = get_key(row[1:len(row) - 1])
        for cid, patternlists in dict_for_cid_patternlists.iteritems():
            if key in patternlists:
                cids = train_dict_for_pid_cid.get(int(row[0]), [])
                cids.append(cid)
                train_dict_for_pid_cid[int(row[0])] = cids
infile.close()

with open("test_numeric.csv", "rb") as infile2:
    reader2 = csv.reader(infile2)
    next(reader2, None)  # skip headers
    for row in reader2:
        key = get_key(row[1:len(row)])
        for cid, patternlists in dict_for_cid_patternlists.iteritems():
            if key in patternlists:
                cids = test_dict_for_pid_cid.get(int(row[0]), [])
                cids.append(cid)
                test_dict_for_pid_cid[int(row[0])] = cids
        if not test_dict_for_pid_cid.get(int(row[0])):
            test_dict_for_pid_cid[int(row[0])]="99"
infile2.close()

with open("common_pattern_train.csv", "r") as infile3:
    reader3 = csv.reader(infile3)
    dict_for_cid_pattern = {}
    for row2 in reader3:
        k, v = row2
        dict_for_cid_pattern[k] = v
infile3.close()

with open("train_numeric.csv", "rb") as infile4:
    reader4 = csv.reader(infile4)
    next(reader4, None)  # skip headers

    for row in reader4:
        list=train_dict_for_pid_cid[int(row[0])]
        for value in list:
            temp_row=row
            key_to_change= dict_for_cid_pattern[value]
            index_for_key_to_change=key_to_change.split()
            for index in range(len(index_for_key_to_change)):
                if index_for_key_to_change[index] == "0"  :
                    temp_row[index+1]=""
            with open("changed_train/train_numeric_changed_cluster"+value+".csv", "a+") as outfile:
                wr = csv.writer(outfile)
                wr.writerow(filter(None, temp_row))
infile4.close()

with open("test_numeric.csv", "rb") as infile5:
    reader5 = csv.reader(infile5)
    next(reader5, None)  # skip headers
    for row in reader5:

        list = test_dict_for_pid_cid[int(row[0])]
        if "99" in list:
            with open("changed_test/test_numeric_changed_cluster99.csv", "a+") as outfile:
                wr = csv.writer(outfile)
                wr.writerow([row[0],0])
        for value in list:
            temp_row=row
            key_to_change= dict_for_cid_pattern[value]
            index_for_key_to_change=key_to_change.split()
            for index in range(len(index_for_key_to_change)):
                if index_for_key_to_change[index] == "0" :
                    temp_row[index + 1] = ""
            with open("changed_test/test_numeric_changed_cluster"+value+".csv", "a+") as outfile:
                wr = csv.writer(outfile)
                wr.writerow(filter(None, temp_row))
infile5.close()


