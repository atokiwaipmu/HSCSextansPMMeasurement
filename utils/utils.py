import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
import emcee

def expo(x,c0,c1):
    return np.exp(c0+c1*x)

def flux_to_mag(x):
    if np.isnan(x):
        return np.nan
    elif x > 0:
        return 27- 2.5*np.log10(x)
    else:
        return np.nan
    
def rad_to_deg(x):
    if np.isnan(x):
        return np.nan
    else:
        return x*180/np.pi
    
def pix_to_deg(x):
    if np.isnan(x):
        return np.nan
    else:
        return x*0.168/3600
    
def fluxerr_to_magerr(x, y):
    if np.isnan(x):
        return np.nan
    elif x > 0:
        return - 2.5*x/(y*np.log(10))
    else:
        return np.nan

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

def calR(ra, dec, center):
    R = np.sqrt((ra - center[0])**2 + (dec - center[1])**2)
    return R

def calRe(ra, dec, centers, theta, eps):
    Re = np.sqrt((((ra - centers[0]) * np.cos(theta) - (dec - centers[1]) * np.sin(theta))/ (1 - eps))**2 +
                 ((ra - centers[0]) * np.sin(theta) + (dec - centers[1]) * np.cos(theta))**2)
    return Re

def bts(s,nsamp): #t: sample, nsamp: number of sub-sample
    tmp = np.array(s)
    n=len(tmp)#total number
    ran=np.random.randint(0,n, (nsamp,n))
    abts = tmp[ran]
    return abts

def per1_sigma(x):
    sigmaG1 = np.sqrt(np.pi/2)*1.428*(np.percentile(x,75)-np.percentile(x,50))
    sigmaG2 = np.sqrt(np.pi/2)*1.428*(np.percentile(x,50)-np.percentile(x,25))
    return (sigmaG1+sigmaG2)/2

def err_all(x,r0,c0,r1):
    return np.sqrt(intri_fix(x,r1)**2 + errmag(x,r0,c0)**2)

def errmag(x,r0,c0):
    return 10**(0.4*(x-r0)) + c0

def intri_fix(x,r1):
    return 10**(-0.2*(x-r1))

#clip
def calc_cov(mc):
    if mc.shape[0]==2 and mc.shape[1]!=2:
        mc=mc.T
    cov=np.cov(mc.T)
    return cov

def clip_grid(t):
    l=[]
    mx2=np.array([t.pmra,t.pmdec])
    mx2=mx2.T
    for k,y in enumerate(mx2):
        cov=calc_cov(mx2)
        if (y-mx2.mean(0)).dot(np.linalg.inv(cov)).dot(y-mx2.mean(0))<6.17:
            l.append(k)
    return t.iloc[l]

def grid(j,mg,x,y, Rflag=False, Reflag=False, args=None):
    Z3=[]
    #print(j)
    for i in range(len(x)-1): #columns, ra
        #print(j,i)
        t=mg[(mg.i_sdsscentroid_ra<x[i+1])&(mg.i_sdsscentroid_ra>x[i])&(mg.i_sdsscentroid_dec>y[j])&(mg.i_sdsscentroid_dec<y[j+1])]
        if (len(t)<10):
            tem=t.copy()
        else:
            #tem=clip(t)#clip 3 sigma for each grid
            tem = clip_grid(t).copy()
        if Rflag & (args is not None):
            tem['R'] = calR(tem.i_sdsscentroid_ra, tem.i_sdsscentroid_dec, args[0])
        if Reflag & (args is not None): #calculate Re
            tem['Re'] = calRe(tem.i_sdsscentroid_ra, tem.i_sdsscentroid_dec, args[0], args[1], args[2])
        Z3.append(tem)
    #print(j,'-')
    return Z3

def Zs(data, pmracol, pmdeccol):
    Znum, Zra, Zdec, Zrasem, Zdecsem = [], [], [], [], []

    for sublist in data:
        Znum_sub, Zra_sub, Zdec_sub, Zrasem_sub, Zdecsem_sub = [], [], [], [], []
        for tmp in sublist:
            Znum_sub.append(len(tmp))
            if len(tmp)==0:
                Zra_sub.append(np.nan)
                Zdec_sub.append(np.nan)
                Zrasem_sub.append(np.nan)
                Zdecsem_sub.append(np.nan)
            else:
                pmra_mean = np.mean(tmp[pmracol])
                pmdec_mean = np.mean(tmp[pmdeccol])
                Zra_sub.append(pmra_mean)
                Zdec_sub.append(pmdec_mean)
                if len(tmp)>1:
                    Zrasem_sub.append(stats.sem(tmp[pmracol]))
                    Zdecsem_sub.append(stats.sem(tmp[pmdeccol]))
                else:
                    Zrasem_sub.append(np.nan)
                    Zdecsem_sub.append(np.nan)

        Znum.append(Znum_sub)
        Zra.append(Zra_sub)
        Zdec.append(Zdec_sub)
        Zrasem.append(Zrasem_sub)
        Zdecsem.append(Zdecsem_sub)

    return [pd.DataFrame(x) for x in [Znum, Zra, Zdec, Zrasem, Zdecsem]]

def cali_star_galerr(i, ms, x, y, Zra, Zdec, Zraerr, Zdecerr):
    t = [ms[(ms.i_sdsscentroid_ra<x[j+1]) & (ms.i_sdsscentroid_ra>x[j]) & (ms.i_sdsscentroid_dec>y[i]) & (ms.i_sdsscentroid_dec<y[i+1])].assign(
            pmra_cl=lambda df: df.pmra - Zra.loc[i, j],
            pmdec_cl=lambda df: df.pmdec - Zdec.loc[i, j],
            pmra_galerr=Zraerr.loc[i, j],
            pmdec_galerr=Zdecerr.loc[i, j]
        ) for j in range(len(x) - 1)]
    
    ttt = pd.concat(t)
    return ttt

def medians(t, x, y, xMin, xMax, Nbin, verbose=1): 
    # Generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    medianBin = np.zeros(Nbin)
    sigGbin = np.zeros(Nbin)

    for i in range(Nbin): 
        mask = (x > xEdge[i]) & (x <= xEdge[i+1])
        yAux = y[mask]

        if yAux.size > 0:
            medianBin[i] = np.median(yAux)
            percentile_diff = np.percentile(yAux, 75) - np.percentile(yAux, 25)

            if verbose == 1:
                sigGbin[i] = np.sqrt(np.pi / 2) * 0.714 * percentile_diff / np.sqrt(yAux.size)
            else:
                sigGbin[i] = np.sqrt(np.pi / 2) * 0.714 * percentile_diff

    return medianBin, sigGbin

def wmeans(t, x, y, yerr, xMin, xMax, Nbin, verbose=1): 
    # Generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    medianBin = np.zeros(Nbin)
    sigGbin = np.zeros(Nbin)

    for i in range(Nbin): 
        mask = (x > xEdge[i]) & (x <= xEdge[i+1])
        yAux = y[mask]
        yerrAux = yerr[mask]

        if yAux.size > 0:
            medianBin[i] = wmean(yAux, yerrAux)
            sigGbin[i] = wmeanerr(yerrAux)

    return medianBin, sigGbin

def apply_async_pool(pool_size, func, yedges, *args):
    with mp.Pool(pool_size) as pool:
        results = [pool.apply_async(func, (i, *args)) for i in range(len(yedges) - 1)]
        return [result.get() for result in results]
    
def wmean(x,y):
    w=1/y**2
    return np.sum(x*w)/np.sum(w)

def wmeanerr(x):
    return np.sqrt(1/np.sum(1/x**2))

def gaussian(x, mean, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/(2*sigma**2))

def logprior(x, priormin, priormax, verbose=False):
    if np.any(np.array(x) < np.array(priormin)) or np.any(np.array(x) > np.array(priormax)):
        if verbose:
            print(x, priormin, priormax, "prior out of range")
        return -np.inf
    else:
        return 0
    
def calarea(gal_path, xedges, yedges, args, Rflag=False, Reflag=False, ):
    mg=pd.read_csv(gal_path)
    data_2d = apply_async_pool(8, grid, yedges, mg, xedges, yedges, Rflag, Reflag, args)
    if Rflag:
        ZdistR = np.array([[np.nanmean(df['R']) if not df.empty else -1 for df in row] for row in data_2d])
        return ZdistR
    elif Reflag:
        ZdistRe = np.array([[np.nanmean(df['Re']) if not df.empty else -1 for df in row] for row in data_2d])
        return ZdistRe
    else:
        print("Rflag or Reflag must be True")

def mcmc(p0, ndim, nwalkers, loglikelihood, args, max_n=10000):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglikelihood, args=args)
                                    
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    return sampler