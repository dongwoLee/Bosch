import csv

input_path = "test_numeric.csv"

maxList=[]
with open(input_path, "r") as infile:
  reader = csv.reader(infile)
  next(reader, None) # skip headers
  count =0
  for row in reader:
      count+=1
      if(count==1):
          maxList = (row)
      else:
          for i in range(1,len(row)-1):
            if(row[i]==""):
                continue
            elif(maxList[i]==""):
                maxList[i] = float(row[i])
            elif(float(row[i]) >= float(maxList[i])):
                maxList[i] = float(row[i])


with open(input_path,"r") as infile:
    reader = csv.reader(infile)
    next(reader,None)
    for row in reader:
        for i in range(1,len(row)-1):
            if((row[i])==""):
                row[i]=(maxList[i])
        with open("test_numeric_max.csv","a+")  as outfile:
            wr = csv.writer(outfile)
            wr.writerow(row)



















