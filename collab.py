from __future__ import print_function

import numpy as np
import csv, io
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from operator import add
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField,StringType
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def compute(x):
    key = [x[0],x[1]]
    run = list(float(i) for i in x[2:])
    return tuple(i for i in key+run)



def computerandrun(prob,randnum):
   if(randnum<prob[0]):
        return 0
   elif randnum >= prob[0] and randnum<prob[1]:
        return 1
   elif randnum >= prob[1] and randnum<prob[2]:
        return 2
   elif randnum >= prob[2] and randnum<prob[3]:
        return 3
   elif randnum >= prob[3] and randnum<prob[4]:
        return 4
   else:
        return 6
def findcluster(x):
   return x[0],int(x[1][1])

def findprob(batbowl,player,models,playerprob):
    
    prob = player.filter(lambda x : x[0] == batbowl[0] and x[1] == batbowl[1]).collect()
    
    if(len(prob)>0):
      prob = prob[0][2:]
    else :
      prob = []
      batsman_index = playerprob.filter(lambda x : x[0] == batbowl[0]).collect()[0][9]
      bowler_index =playerprob.filter(lambda x : x[1] == batbowl[1]).collect()[0][10]
      l = [[batsman_index,bowler_index]]
      testdata = sc.parallelize(tuple(i for i in l))
      testdata = testdata.map(lambda p: (p[0], p[1]))
      
      prob.append(models[0].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[1].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[2].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[3].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[4].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[5].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      prob.append(models[6].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()[0][1])
      
    prob1 = prob[:len(prob)-1]
    total = sum(prob1)
    if(total==0):
       prob = [1 for i in range(len(prob))]
       total=6
    prob1 = [i/total for i in prob1]
    for i in range(1,len(prob1)):
      prob1[i] = prob1[i-1]+prob1[i]
    prob1.append(prob[-1])

    return prob1


    
 
   

def out_probability(rdd):
    prob = {}
    l = rdd
    for i in l:
       prob[i[0]] = 1
    return prob

def swap(batsk,batnsk):
    return batnsk,batsk
def computerandrun(prob,randnum):
   if(randnum<prob[0]):
        return 0
   elif randnum >= prob[0] and randnum<prob[1]:
        return 1
   elif randnum >= prob[1] and randnum<prob[2]:
        return 2
   elif randnum >= prob[2] and randnum<prob[3]:
        return 3
   elif randnum >= prob[3] and randnum<prob[4]:
        return 4
   else:
        return 6

if __name__ == "__main__":

   
    sc = SparkContext(appName="collaborative_filtering")
    sqlContext = SQLContext(sc)
    playerprob = sc.textFile("finalcluster2")
    playerprob=playerprob.map(lambda x:x.split(','))
    playerprobfinal=playerprob.map(lambda x:compute(x))
   
    player_schema = StructType() \
          .add("Bat", "string") .add("Bowl","string").add("0","float").add("1","float").add("2","float").add("3","float") .add("4","float").add("5","float").add("w","float")
    dataframe = sqlContext.createDataFrame(playerprobfinal, player_schema)
    rddtodf = [StringIndexer(inputCol=j, outputCol=j+"_i") for j in list(set(dataframe.columns)-set(['0'])-set(['1'])-set(['2'])-set(['3'])-set(['4'])-set(['6'])-set(['w']))]
    dataframe1 = Pipeline(stages=rddtodf)
    indexed = dataframe1.fit(dataframe).transform(dataframe)
    playerprob = indexed.rdd.map(tuple)


    rank = 10
    numIterations = 10
    ratingszero = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[2])))
    ratingsone = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[3])))
    ratingstwo = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[4])))
    ratingsthree = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[5])))
    ratingsfour = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[6])))
    ratingssix = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[7])))
    ratingswickets = playerprob.map(lambda x: Rating(int(x[9]), int(x[10]), float(x[8])))

    modelzero = ALS.train(ratingszero, rank, numIterations)
    modelone = ALS.train(ratingsone, rank, numIterations)
    modeltwo = ALS.train(ratingstwo, rank, numIterations)
    modelthree = ALS.train(ratingsthree, rank, numIterations)
    modelfour = ALS.train(ratingsfour, rank, numIterations)
    modelsix = ALS.train(ratingssix, rank, numIterations)
    modelwickets = ALS.train(ratingswickets, rank, numIterations)
    models = [modelzero,modelone,modeltwo,modelthree,modelfour,modelsix,modelwickets]





    batting_order1 = sc.textFile("batting_order1.txt")
    bat_ord1 = batting_order1.collect()
    batting_order1 = batting_order1.map(lambda x: (x,-1))
    batting_order2 = sc.textFile("batting_order2.txt")
    bat_ord2 = batting_order2.collect()
    batting_order2 = batting_order2.map(lambda x: (x,-1))
    bowling_order1 = sc.textFile("bowling_order1.txt")
    bowl_ord1 = bowling_order1.collect()
    bowling_order1 = bowling_order1.map(lambda x: (x,-1))
    bowling_order2 = sc.textFile("bowling_order2.txt")
    bowl_ord2 = bowling_order2.collect()
    bowling_order2 = bowling_order2.map(lambda x: (x,-1))
    
   
    #innings 1
    print("***********INNINGS 1 ***********")
    total = 0
    bat1 = bat_ord1[0]
    bat2 = bat_ord1[1]
    bowl = bowl_ord1[0]
    batnext = 2
    bowlnext = 1
    wickets = 0
    ball = 0
    over = 0
    prob1 = findprob((bat1,bowl),playerprobfinal,models,playerprob)
    prob2 = findprob((bat2,bowl),playerprobfinal,models,playerprob)
    final_bat_ord=bat_ord1+bat_ord2
    
   
 
    outprobability = {}
    for i in bat_ord1:
      outprobability[i] = 1
    for i in bat_ord2:
      outprobability[i] = 1
    
    probdict = {bat1:prob1,bat2:prob2}

    batsk = bat1
    batnsk = bat2
    while(over<20 and wickets<10):
       randrun = computerandrun(probdict[batsk],random.random())
       total+=randrun
       ball+=1
       
       outprobability[batsk]= outprobability[batsk]-probdict[batsk][6]
       if(outprobability[batsk]<0.5):
             
             wickets+=1
             print("Wicket number : ",wickets,"\t Batsman : ",batsk,"\tBowler : ",bowl)
             if(wickets != 10):
           
                batsk = bat_ord1[batnext]
                
                batnext+=1
             total-=randrun
             
             probdict[batsk] = findprob((batsk,bowl),playerprobfinal,models,playerprob)
       elif(randrun%2==1):
            batsk,batnsk = swap(batsk,batnsk)
       if(ball%6==1):
            over+=1
            print("runs scored in ",int(ball/6)+1,"over : ",total,"/",wickets)
            if(bowlnext!=20):
               bowl = bowl_ord1[bowlnext]
            bowlnext+=1
            batsk,batnsk = swap(batsk,batnsk)
            probdict[batsk] = findprob((batsk,bowl),playerprobfinal,models,playerprob)
            probdict[batnsk] = findprob((batnsk,bowl),playerprobfinal,models,playerprob)
    total1 = total
    print("Team 1 Total Score  :", total1,"/",wickets,"\n\n")
    print("***********INNINGS 2 ***********")
    
    total = 0
    bat1 = bat_ord2[0]
    bat2 = bat_ord2[1]
    bowl = bowl_ord2[0]
    batnext = 2
    bowlnext = 1
    wickets = 0
    ball = 0
    over = 0
    prob1 = findprob((bat1,bowl),playerprobfinal,models,playerprob)
    prob2 = findprob((bat2,bowl),playerprobfinal,models,playerprob)
    probdict = {bat1:prob1,bat2:prob2}
    
    batsk = bat1
    batnsk = bat2
    while(over<20 and wickets<10):
       randrun = computerandrun(probdict[batsk],random.random())
       total+=randrun
       ball+=1
       
       outprobability[batsk]= outprobability[batsk]-probdict[batsk][6]
       if(outprobability[batsk]<0.5):
             wickets+=1
             print("Wicket number : ",wickets,"\t Batsman : ",batsk,"\tBowler : ",bowl)
             if(wickets != 10):
                 
              batsk = bat_ord2[batnext]
              
              batnext+=1
             total-=randrun
             
             probdict[batsk] = findprob((batsk,bowl),playerprobfinal,models,playerprob)
       elif(randrun%2==1):
            batsk,batnsk = swap(batsk,batnsk)
       if(ball%6==1):
            over+=1
            print("Runs scored in ",int(ball/6)+1,"over : ",total,"/",wickets)
            if(bowlnext!=20):
               bowl = bowl_ord2[bowlnext]
            bowlnext+=1
            batsk,batnsk = swap(batsk,batnsk)
            probdict[batsk]=findprob((batsk,bowl),playerprobfinal,models,playerprob)
            probdict[batnsk] = findprob((batnsk,bowl),playerprobfinal,models,playerprob)
    total2 = total
    print("Team 2 Total Score  :", total2,"/",wickets,"\n\n")
    print("**************************************************************************")
    if(total1 == total2):
       print("\t\t\tDraw\t\t\t")
    elif(total1> total2):
       print("\t\t\tWinner is Team1\t\t\t")
    else:
       print("\t\t\tWinner is Team2\t\t\t")
       
 
    sc.stop()







