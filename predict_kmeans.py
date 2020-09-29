from __future__ import print_function

import numpy as np
import csv, io
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from operator import add
import random

def computeprob(x):
    key = (x[2],x[3])
    run = list(float(i) for i in x[4:])
    tot = sum(run)
    run = [runs/tot for runs in run]
    res = [0 for i in range(6)]
    cum = 0
    for i in range(6):
        cum += run[i]
        res[i] = cum

    res.append(float(x[10]))
    return (key,res)

def computeprob0(x):
    key = (int(x[0]),int(x[1]))
    run = list(float(i) for i in x[2:])
    tot = sum(run)
    run = [runs/tot for runs in run]
    res = [0 for i in range(6)]
    cum = 0
    for i in range(6):
        cum += run[i]
        res[i] = cum
    res.append(float(x[8]))    
    return (key,res) 

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

def findprob(batbowl,player,cluster,batclust,bowlclust):
    
    prob = player.filter(lambda x : x == batbowl).collect()
    
    if(len(prob)>0):
      prob = prob[0][1]
    else :
      
      batcn = batclust.filter(lambda x: x[0] == batbowl[0]).collect()[0][1]
      bowlcn = bowlclust.filter(lambda x: x[0] == batbowl[1]).collect()[0][1]
      
      clust = (batcn,bowlcn)
      prob = cluster.filter(lambda x: x[0] == clust).collect()
      prob = prob[0][1]
    return prob

def out_probability(rdd):
    prob = {}
    l = rdd.collect()
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

    
    sc = SparkContext(appName="Probability")
    clusterprob = sc.textFile("clusterprobability")
    clusterprob = clusterprob.map(lambda line:line.split(","))
    clusterprob = clusterprob.map(lambda x: computeprob0(x))
    playerprob = sc.textFile("finalcluster")
    playerprob = playerprob.map(lambda line:line.split(","))
    playerprob = playerprob.map(lambda x: computeprob(x))
    batcluster = sc.textFile("batcluster")
    batcluster = batcluster.map(lambda x : tuple(i for i in x.split(",")))
    bowlcluster = sc.textFile("bowlcluster")
    bowlcluster = bowlcluster.map(lambda x : tuple(i for i in x.split(",")))
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
    battingordclustnum = batting_order1.union(batting_order2).distinct().join(batcluster)
    bowlingordclustnum = bowling_order1.union(bowling_order2).distinct().join(bowlcluster)
    battingordclustnum = battingordclustnum.map(lambda x: findcluster(x))
    bowlingordclustnum = bowlingordclustnum.map(lambda x: findcluster(x))
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
    prob1 = findprob((bat1,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    prob2 = findprob((bat2,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    outprobability = out_probability(battingordclustnum)
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
             probdict[batsk] = findprob((batsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
       elif(randrun%2==1):
            batsk,batnsk = swap(batsk,batnsk)
       if(ball%6==1):
            over+=1
            print("Runs scored in ",int(ball/6)+1,"over : ",total,"/",wickets)
            if(bowlnext!=20):
               bowl = bowl_ord1[bowlnext]
            bowlnext+=1
            batsk,batnsk = swap(batsk,batnsk)
            probdict[batsk] = findprob((batsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
            probdict[batnsk] = findprob((batnsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    total1 = total
    print("Team 1 Total Score  :", total1,"/",wickets,"\n\n")
    #innings2
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
    prob1 = findprob((bat1,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    prob2 = findprob((bat2,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    outprobability = out_probability(battingordclustnum)
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
             
             probdict[batsk] = findprob((batsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
       elif(randrun%2==1):
            batsk,batnsk = swap(batsk,batnsk)
       if(ball%6==1):
            over+=1
            print("Runs scored in ",int(ball/6)+1,"over : ",total,"/",wickets)
            if(bowlnext!=20):
               bowl = bowl_ord2[bowlnext]
            bowlnext+=1
            batsk,batnsk = swap(batsk,batnsk)
            probdict[batsk]=findprob((batsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
            probdict[batnsk] = findprob((batnsk,bowl),playerprob,clusterprob,battingordclustnum,bowlingordclustnum)
    total2 = total
    print("Team 2 Total Score  :", total2,"/",wickets,"\n\n")           
    if(total1 == total2):
       print("Draw")
    elif(total1> total2):
       print("Winner is Team1")
    else:
       print("Winner is Team2")
       
  
    sc.stop()
    
