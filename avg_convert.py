import csv

input_path = "dataset/train_numeric.csv"

maxList=[]
with open(input_path, "r") as infile:
  reader = csv.reader(infile)
  next(reader, None) # skip headers
  count =0
  for row in reader:
      count+=1
      if(count==1):
	  maxList=row
          for i in range(1,len(row)-1):
	    if(row[i]==""):
 		maxList[i]=0
	    else:
		maxList[i]=float(row[i])
      else:
          for i in range(1,len(row)-1):
            if(row[i]==""):
                continue
            else:
                maxList[i] += float(row[i])
  for i in range(1,len(maxList)-1):
    maxList[i] = round((maxList[i]/1183748),3) 

with open(input_path,"r") as infile:
    reader = csv.reader(infile)
    next(reader,None)
    for row in reader:
        for i in range(1,len(row)-1):
            if((row[i])==""):
                row[i]=(maxList[i])
        with open("train_numeric_max.csv","a+")  as outfile:
            wr = csv.writer(outfile)
            wr.writerow(row)
