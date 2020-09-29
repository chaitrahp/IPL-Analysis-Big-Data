from __future__ import print_function
import sys
import csv,io
import numpy as np
from pyspark import SparkContext
from operator import add

from math import sqrt
def closest_Point(pt1, cts1):
    bstidx = 0
    clst1 = float("+inf")
    for i in range(len(cts1)):
             tmpdst = np.sum((pt1 - cts1[i]) ** 2)
             if tmpdst < clst1:
                 clst1 = tmpdst
                 bstidx = i
    
         
    return bstidx
def tocsv(y):
    
    output = io.StringIO("")
    csv.writer(output).writerow(y)
    return output.getvalue().strip() 
def closest_Cluster(pt1, cts1):
    bstidx = len(cts1)+1
    clst1 = float("+inf")
    for i in range(0,len(cts1)):
             tmpdst = np.sum((np.array(pt1[1]) - cts1[i]) ** 2)
             if tmpdst < clst1:
                 clst1 = tmpdst
                 bstidx = i
    
    return pt1[0],bstidx
def computet(yy):
	return (yy[6],np.array([float(yy[1])/float(yy[0]),float(yy[2])/float(yy[0]),float(yy[3]),float(yy[4]),float(yy[5])]))
def compute(yy):
	return np.array([float(yy[1])/float(yy[0]),float(yy[2])/float(yy[0]),float(yy[3]),float(yy[4]),float(yy[5])])

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: K-means <file> <k> <convdist>", file=sys.stderr)
        exit(-1)

    spkctxt = SparkContext(appName="PythonKMeans")
    lin = spkctxt.textFile(sys.argv[1])
    indi=[2,5,7,9,10,11,0]
    hdr = lin.first() 
    lin = lin.filter(lambda row:row != hdr)
    cen1 = lin.map(lambda line:line.split(","))
    cen1 = cen1.map(lambda yy:[yy[indx] for indx in indi])
    cen1 = cen1.map(lambda yy:compute(yy)).cache()
    dt = lin.map(lambda line:line.split(","))
    dt = dt.map(lambda yy:[yy[indx] for indx in indi])		
    forct= dt.map(lambda yy:compute(yy))
    btsmn=dt.map(lambda yy:computet(yy))
    k = int(sys.argv[2])
    convdist = float(sys.argv[3])

    kpts = cen1.takeSample(False, k, 1)
    print(kpts)
    tmpdst = 100.0
    
    while tmpdst > convdist:
        clst1 = cen1.map(lambda p:( closest_Point(p, kpts), (p, 1)))
        
        ptst = clst1.reduceByKey(add)
        
        nwpts = ptst.map(lambda s_t: (s_t[0], s_t[1][0] / s_t[1][1])).collect()
        
        tmpdst = sum(np.sum((kpts[ik] - p) ** 2) for (ik, p) in nwpts)

        for (ik, p) in nwpts:
            kpts[ik] = p
    print("Final centeroids: " + str(kpts))
    btclst = btsmn.map(lambda xx : closest_Cluster(xx,kpts))
    btclst = btclst.map(lambda xx :tocsv(xx))
    btclst.saveAsTextFile("bowlcluster")
    

    spkctxt.stop()

