
import numpy as np
import pandas as pd

def proplummer(re,rp):
    return (1 + re**2/rp**2)**(-2)

def proking(re,rc,rt):
    return (1/np.sqrt(1+(re/rc)**2) - 1/np.sqrt(1+(rt/rc)**2))**2

def proexpo(re,rex):
    return np.exp(-re/rex)

def clip(t,x, y, xMin, xMax, Nbin, verbose=1): 
    # first generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    xBin = np.linspace(0, 1, Nbin)
    nPts = 0*np.linspace(0, 1, Nbin)
    medianBin = 0*np.linspace(0, 1, Nbin)
    sigGbin = [-1+0*np.linspace(0, 1, Nbin),-1+0*np.linspace(0, 1, Nbin)] #lower and upper
    tt=[]
    for i in range(0, Nbin): 
        xBin[i] = 0.5*(xEdge[i]+xEdge[i+1]) 
        t1 = t[(x>xEdge[i])&(x<=xEdge[i+1])]
        yAux1 = y[(x>xEdge[i])&(x<=xEdge[i+1])]
        if (yAux1.size > 0):
            yAux = yAux1#[(yAux1<np.percentile(yAux1,99.85))&(yAux1>np.percentile(yAux1,0.15))]#3sigma clipping
            nPts[i] = yAux.size
            medianBin[i] = np.median(yAux) #mean
            # robust estimate of standard deviation: np.sqrt(np.pi/2)*0.741*(q75-q50),np.sqrt(np.pi/2)*0.741*(q50-q25)
            sigmaG1 = np.sqrt(np.pi/2)*0.741*(np.percentile(yAux,75)-np.percentile(yAux,50))*2
            sigmaG2 = np.sqrt(np.pi/2)*0.741*(np.percentile(yAux,50)-np.percentile(yAux,25))*2
            sigGbin[0][i] = sigmaG1
            sigGbin[1][i] = sigmaG2
            t1=t1[(yAux<np.median(yAux)+3*sigmaG1)&(yAux>np.median(yAux)-3*sigmaG2)]
            tt.append(t1)
    tt=pd.concat(tt)
    return tt, medianBin , sigGbin