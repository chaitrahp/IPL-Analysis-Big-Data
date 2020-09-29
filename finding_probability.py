from __future__ import print_function
import sys
import numpy as np
import csv, io
from pyspark import SparkContext
from pyspark.sql import SQLContext
from operator import add
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

def toCSVLine1(data):
  return ','.join(str(d) for d in data)   

def batbowlcluster(x):
    s = x[1][1][0]+"#"+x[0]
    return s,(x[1][1][1],x[1][0])    #returns batsman#bowler,(batcluster,bowlcluster)

def batbowlfun(x):
    return x[1][1][0],(x[0],x[1][0]) 

def computeclu(x):
    inp = x.split(',')
    return(inp[0],inp[1])

def computeruns(x):
    inp = x[0].split("#")
    yield (inp[0]+"#"+inp[1],(int(inp[2]),x[1]))

def list_to_csv_str(x):
    """Given a list of strings, returns a properly-csv-formatted string."""
    output = io.StringIO("")
    csv.writer(output).writerow(x)
    return output.getvalue().strip() # remove extra newline
   
def computeprob(bat,run):
    l = [r[1] for r in run]
    total = sum(l)
    if total > 10:
        for i in run:
            yield(bat,(i[0],i[1]/total))
def toCSVLine(x):
  name= x[0].split("#")
  l = [0 for i in range(7)]
  d = {0:0,1:1,2:2,3:3,4:4,6:5,7:6}
  for i in x[1]:
     l[d[i[0]]] = i[1]  # i = (run,probability)
  final = name + l
  
  t = tuple(i for i in final)
  t1 = t[1:]
  yield (t[0],t1)

def toCSVcluster(x):
  name= [x[0]]
  l = [0 for i in range(7)]
  d = {0:0,1:1,2:2,3:3,4:4,6:5,7:6}
  for i in x[1]:
     l[d[i[0]]] = i[1]  # i = (run,probability)
  final = name + l
  
  t = tuple(i for i in final)
  t1 = t[1:]
  yield (t[0],t1)

def toline(x):
    l = []
    l.append(x[1][0])
    l.append(x[0])
    for i in x[1][1]:
       l.append(i)
    listToStr = ' '.join(map(str, l))
    return listToStr

def finalcluster(x):
    clu = [x[1][0][0],x[1][0][1]]
    batbowl = x[0].split('#')
    run = list()
    for i in x[1][1]:
       run.append(i)
    return tuple( i for i in clu+batbowl+run)

def wicket(x):
    if len(x[9]) >3 and x[9] not in ('retired hurt','run out'):
      	res = (x[4]+"#"+x[6]+"#"+'7',1)
    else:
        res = (x[4]+"#"+x[6]+"#"+x[7],1)
    return res

if __name__ == "__main__":

    
    sc = SparkContext(appName="Probability")
    lines = sc.textFile(sys.argv[1])
    cen = lines.map(lambda line:line.split(","))
    play = cen.filter(lambda x: x[0] == "ball")
    play = play.map(lambda x: wicket(x))
    bat = play.groupByKey().mapValues(len)
    
    
    
    bat1 = bat.flatMap(lambda x: computeruns(x))
    bat1 = bat1.filter(lambda x: x[1][0] not in (5,) )
    
    bat1 = bat1.groupByKey()
    run = bat1.flatMap(lambda x: computeprob(x[0],x[1])).groupByKey()
    #print(run.collect()[:10])
    lines = run.flatMap(lambda x :toCSVLine(x))
    linesclu = run.flatMap(lambda x :toCSVcluster(x))
    #print(linesclu.collect()[:10])
    #linesclu = (batsman#bowler,(probabilities of 0,1,2,3,4,6))
    bat = sc.textFile("/home/chaitra/Desktop/project/Step1/batcluster").map(lambda x: computeclu(x))
    #ssprint(bat.collect())
    
    bowl = sc.textFile("/home/chaitra/Desktop/project/Step1/bowlcluster").map(lambda x: computeclu(x))
    
    print(bowl.collect()[:10]) 
    final = bat.join(lines)
    #lines = ('GJ Bailey', ('3', ('AB Dinda', 0.4666666666666667, 0.26666666666666666, 0.06666666666666667, 0, 0.2, 0)))
    #print(final.collect()[:10])
    batbowl = final.map(lambda x : batbowlfun(x))
    
    batbowl = bowl.join(batbowl)
    # batbowl = (bowler,(bowlercluster,(batsman,batsmancluster))
    #print(batbowl.collect()[:10])
    batcluster = batbowl.map(lambda x:batbowlcluster(x))
    
    batcluster1 = batcluster.join(linesclu)
    cluster = batcluster1.map(lambda x: finalcluster(x))
    #print(cluster.collect()[:10])
    lines1 = cluster.map(toCSVLine1)
    #print(lines1.collect()[:10])
    #rdd = final.map(lambda x: toline(x))
    #print(rdd.collect()[:10])
    lines1.saveAsTextFile("finalcluster")
    
    

    sc.stop()

