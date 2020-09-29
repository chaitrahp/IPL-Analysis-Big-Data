from __future__ import print_function
import sys
import csv,io
import numpy as np
from pyspark import SparkContext
from operator import add
from math import sqrt


def closest_Point(pt1, cts1):
    bstidx = 0
    clst = float("+inf")
    for i in range(len(cts1)):
             tmpdst = np.sum((pt1- cts1[i]) ** 2)
             if tmpdst < clst:
                 clst = tmpdst
                 bstidx = i
    return bstidx
def tocsv(x):
    
    output = io.StringIO("")
    csv.writer(output).writerow(x)
    return output.getvalue().strip() 
def closestCluster(pt1, cts1):
    bstidx = len(cts1)+1
    clst = float("+inf")
    for i in range(0,len(cts1)):
             tmpdst = np.sum((np.array(p[1]) - cts1[i]) ** 2)
             if(tmpdst < clst):
                 clst = tmpdst
                 bstidx = i
    
    return pt1[0],bstidx
def convert(yy):
        
	return (yy[6],np.array([float(yy[1]),float(yy[2]),float(yy[3])/float(yy[0]),float(yy[4])/float(yy[0]),float(yy[5])/float(yy[0])]))
def conver(yy):
	return np.array([float(yy[1]),float(yy[2]),float(yy[3])/float(yy[0]),float(yy[4])/float(yy[0]),float(yy[5])/float(yy[0])])

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: K-means <file> <k> <convdst>", file=sys.stderr)
        exit(-1)

    sprkctx = SparkContext(appName="PythonKMeans")
    lin = sprkctx.textFile(sys.argv[1])
    indc=[2,7,9,11,13,14,0]
    hd = lin.first() 
    lin = lin.filter(lambda row:row != hd)
    centr1 = lin.map(lambda line:line.split(","))
    centr1= centr1.map(lambda yy:[yy[idx] for idx in indc])
    centr1 = centr1.map(lambda yy:conver(yy)).cache()
    dt = lin.map(lambda line:line.split(","))
    dt = dt.map(lambda yy:[yy[idx] for idx in indc])		
    forct= dt.map(lambda yy:conver(yy))
    btmn=dt.map(lambda yy:convert(yy))
    k = int(sys.argv[2])
    convdst = float(sys.argv[3])

    kpts = centr1.takeSample(False, k, 1)
    tmpdst = 100.0
    
    while tmpdst > convdst:
        clst = centr1.map(lambda p:( closest_Point(p, kpts), (p, 1)))
        
        ptst = clst.reduceByKey(add)
        
        npts = ptst.map(lambda s_t: (s_t[0], s_t[1][0] / s_t[1][1])).collect()
        
        tmpdst = sum(np.sum((kpts[ik] - p) ** 2) for (ik, p) in npts)

        for (ik, p) in npts:
            kpts[ik] = p
    print("Final centroids: " + str(kpts))
    btclst = btmn.map(lambda yy : closestCluster(yy,kpts)).map(lambda yy :tocsv(yy)).saveAsTextFile("batcluster")
    

    sprkctx.stop()

