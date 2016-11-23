import csv
import numpy as np
import xgboost

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
          xls[i] = np.nan
      y_vec = [0] if (row[(len(row)-1)] == '0') else [1]
      if (np.random.random_sample() < train_portion):
        train_x.append(np.float32(xls))
        train_y.append(y_vec)
      else:
        test_x.append(np.float32(xls))
        test_y.append(y_vec)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def read_submit_input(path,is_train):
    with open(path, "r") as infile:
        reader = csv.reader(infile)
        list_x = []
        list_y = []
        next(reader, None)
        for row in reader:
            if is_train:
                xls = row[1:len(row) - 1]
            else:
                xls = row[1:len(row)]
            for i in range(0, len(xls)):
                if (len(xls[i]) == 0):
                    xls[i] = np.nan
            list_x.append(np.float32(xls))
            if is_train:
        y_vec = 0 if (row[(len(row) - 1)] == '0') else 1
                list_y.append(y_vec)
            else:
                list_y.append(row[0])
    return np.array(list_x), np.array(list_y)

train_sample_x,train_sample_y,test_sample_x,test_sample_y = read_input("/home/ubuntu/train_numeric_100000.csv",False,0.8)
model = xgboost.XGBClassifier()
model.fit(train_sample_x, train_sample_y)
y_sample_pred = model.predict(test_sample_x)
calculate_mcc(test_sample_y,y_sample_pred)

train_x,train_y=read_submit_input("train_numeric.csv", True)
test_x,test_y=read_submit_input("test_numeric.csv", False)

with open("result.csv", "a+") as outfile:
  for i in range(0,len(test_y)):
    wr = csv.writer(outfile)
    wr.writerow([test_y[i],y_pred[i]])

