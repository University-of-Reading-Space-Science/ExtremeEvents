# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:05:47 2020

@author: mathewjowens

A script to reproduce the anaylsis of Owens et al., Solar Physics 2021,
"Extreme space-weather events and the solar cycle"

"""
import numpy as np
import pandas as pd
from datetime import datetime
import os as os
from scipy.stats import pearsonr
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import helio_time as time


plt.rcParams.update({
    "text.usetex": False,
    'font.size': 12,
    "font.sans-serif": ["Helvetica"]})

# <codecell> data file paths
ssnfilepath = '..\\data\\SN_m_tot_V2.0.txt'
aaHfilepath = '..\\data\\aaH3Hr_SWSC.txt'
solarminfilepath = '..\\data\\SolarMinTimes.txt'

# <codecell> User-defined variables
#aa_processing = 'mean' #how to process daily aaH, either mean or max
nbins=30 #Number of phase bins for SPE [30]
nthreshbins = 20 #number of storm threshold bins [20]
plot_percentiles = [90,99,99.9,99.99]
Nmc = 5000 #number of Monte Carlo iterations [5000]
active_amp = 9 #probability during active phases relative to break
cycle_amp = 1.5 
d_earlylate = 0.6 #change in probability in early/late active phase
demoSCs=[12,19,23] #example solar cycles for projecting SC25

# <codecell> custom functions to be used later

#bin up the data
def binxdata(xdata,ydata,bins):
    
    #check whether the number of bins or the bin edges have been specified
    if isinstance(bins,np.ndarray):
        xbinedges=bins
    else:
        xbinedges = np.arange(xdata.min(),xdata.max()+0.01,(xdata.max()-xdata.min())/(bins+1))  
    numbins = len(xbinedges) - 1
        
    xbindata=np.zeros((numbins,3))*np.nan
    for n in range(0,numbins):
        #time at bin centre
        xbindata[n,0]=(xbinedges[n]+xbinedges[n+1])/2
        
        #find the data of interest
        mask =  (xdata >= xbinedges[n]) & (xdata < xbinedges[n+1])
        if np.nansum(mask) > 0:
            xbindata[n,1]=np.nanmean(ydata[mask])
            xbindata[n,2]=np.nanstd(ydata[mask])
            
    return xbindata

#remove nans
def nonans(array):
    return array[~np.isnan(array)]

#perform a super-posed eopch analysis with two trigger times
def SPE_dualepoch(times, data, epochs_start, epochs_stop, bins):
    
    assert(isinstance(times,np.ndarray))
    assert(isinstance(data,np.ndarray))
    assert(isinstance(epochs_start,np.ndarray))
    assert(isinstance(epochs_stop,np.ndarray))
    
    #check whether the number of bins or the bin edges have been specified
    if isinstance(bins,np.ndarray):
        xbinedges=bins
    else:
        xbinedges = np.arange(data.min(),data.max(),(data.max()-data.min())/(bins+1))  
    numbins = len(xbinedges) - 1
    
    #set up the variables
    binned_mean = np.empty((len(epochs_start), numbins))
    binned_std = np.empty((len(epochs_start), numbins))

    #bin up the data for each epoch
    nodata=True
    for i in range(0,len(epochs_start)):
        
        triggerstart=epochs_start[i]
        triggerstop=epochs_stop[i]
        epochlength=triggerstop-triggerstart
        
        if (triggerstart >= times[0]) | (triggerstop < times[-1]):
                    
            #composite on data 
            mask = (times >= triggerstart) & (times < triggerstop)
            xdata = (times[mask] - triggerstart)/epochlength
            ydata = data[mask]
            temp = binxdata(xdata,ydata,bins)
            binned_mean[i,:]=temp[:,1]
            binned_std[i,:]=temp[:,2]
            nodata=False
                      
        else:
            binned_mean[i,:]=np.nan
            binned_std[i,:]=np.nan

          
    #put together the super-posed epoch analysis
    if nodata == False:
        spe = np.empty((nbins, 4))
        spe[:,0]=temp[:,0] 
        for i in range(0,numbins):
            spe[i,1] = np.nanmean(binned_mean[:,i])
            spe[i,2] = np.nanstd(binned_mean[:,i])
            #standard error on the mean
            Nreal = len(nonans(binned_mean[:,1]))
            spe[i,3] = spe[i,2] / np.sqrt(Nreal)
    else:
        spe = np.empty((nbins, 4))
        
    return spe

def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.txt'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = time.datetime2mjd(dfdt)
    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)
    ssn_df.drop(2,axis=1,inplace=True)
    ssn_df.drop(3,axis=1,inplace=True)
    ssn_df.drop(4,axis=1,inplace=True)
    ssn_df.drop(5,axis=1,inplace=True)
    
    #add the 13-month running smooth
    window = 13*30
    temp = ssn_df.rolling(str(window)+'D', on='datetime').mean()
    ssn_df['smooth'] = np.interp(ssn_df['mjd'],temp['mjd'],temp['ssn'],
                                              left =np.nan, right =np.nan)
    
    return ssn_df

def LoadaaH(filepath = 'null'):
    #(download from https://www.swsc-journal.org/articles/swsc/olm/2018/01/swsc180022/swsc180022-2-olm.txt)
    if filepath == 'null':
        filepath = os.environ['DBOX'] + 'Data\\aaH3Hr_SWSC.txt'

    aaH_df = pd.read_csv(filepath,
                     skiprows = 35, delim_whitespace=True,
                     names=['year','month','day','hr',
                            'fracyr','aaH','aaHN',
                            'aaHS','aaS','aaSN','aaSS'])


    #convert the date/time info into a datetime object
    aaH_df['datetime'] = pd.to_datetime({'year' : aaH_df['year'],
                                  'month' : aaH_df['month'],
                                  'day' : aaH_df['day'],
                                  'hour' : aaH_df['hr']})


    #add a MJD column
    aaH_df['mjd'] = time.date2mjd(aaH_df['year'].to_numpy(),
                              aaH_df['month'].to_numpy(),
                              aaH_df['day'].to_numpy(),
                              aaH_df['hr'].to_numpy())


    #set the dataframe index to be datetime and delete the unwanted columns
    #aaH_df = aaH_df.set_index(aaH_df['datetime'])
    aaH_df.drop('year',axis=1,inplace=True)
    aaH_df.drop('month',axis=1,inplace=True)
    aaH_df.drop('day',axis=1,inplace=True)
    aaH_df.drop('hr',axis=1,inplace=True)
    
    return aaH_df

def LoadSolarMinTimes(filepath = 'null'):
    if filepath == 'null':
        filepath = os.environ['DBOX'] + 'Data\\SolarMinTimes.txt'
    solarmintimes_df = pd.read_csv(filepath,
                         delim_whitespace=True,
                         names=['fracyear','cyclenum'])
    doy = (solarmintimes_df['fracyear'] - np.floor(solarmintimes_df['fracyear']))*364 + 1
    doy = doy.to_numpy()
    yr = np.floor(solarmintimes_df['fracyear']).to_numpy()
    yr=yr.astype(int)
    solarmintimes_df['mjd'] = time.doyyr2mjd(doy,yr)
    
    solarmintimes_df['datetime'] = pd.to_datetime(solarmintimes_df['fracyear'], format='%Y', errors='coerce')

    return solarmintimes_df

def load_gle_list(filepath = 'null'):
    """
        Function to load in a list of GLEs provided by the NGDC at
        ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/COSMIC_RAYS/ground-level-enhancements/ground-level-enhancements.txt
        :return: DF - DataFrame containing a list of GLE events, including fields:
           time     : Datetime index of GLE onset
           jd       : Julian date of GLE onset
           sc_num   : Solar cycle number of GLE onset
           sc_phase : Solar cycle phase of GLE onset
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    if filepath == 'null':
        filepath = os.environ['DBOX'] + 'Data\\gle_list.txt'
    
    
    
    colnam = ['id', 'date', 'datenum', 'baseline']
    dtype = {'id': 'int', 'date': 'str', 'datenum': 'int', 'baseline': 'str'}
    df = pd.read_csv(filepath, sep=',', names=colnam, header=None, dtype=dtype)
    # Convert time and loose useless columns. Calculate julian date of each observation.
    df['time'] = pd.to_datetime(df['date'])
    df.drop(['date', 'datenum', 'baseline'], axis=1, inplace=True)
    df['jd'] = pd.DatetimeIndex(df['time']).to_julian_date()
    # # Add in solar cycle number, phase and rise or fall state
    # number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    # df['sc_num'] = number
    # df['sc_phase'] = phase
    # df['sc_state'] = state
    # df['hale_num'] = hale_num
    # df['hale_phase'] = hale_phase
    return df

#draw events from a probability time series
@jit(nopython=True)
def generate_events(rel_prob, Nstorms):
    """
    Function to generate a quasi-random series of events using the prescribed time-varying probability

    Parameters
    ----------
    rel_prob : time series of the relative probability of an event. Will be normalised to 1.
    Nstorms : The number of events to generate

    Returns
    -------
    storm_list : time series of events, on same time step as rel_prob


    """
    storm_list = np.zeros(len(rel_prob))
    
    #put together the CDF from the relative probability series
    prob_cdf = np.zeros(len(rel_prob))
    prob_cdf[0] = rel_prob[0]
    for i in range(1,len(rel_prob)):
        prob_cdf[i] = prob_cdf[i-1] + rel_prob[i]
    #normalise the CDF
    prob_cdf = prob_cdf /  prob_cdf[-1]
    
    #now place the events
    for n in range(0,Nstorms):
        placedstorm = False
        while placedstorm == False:
            x = np.random.rand()
            #find closest point in CDF
            delta = np.abs(prob_cdf - x)
            imin = delta.argmin() 
            
            #check there's not a storm here already
            if storm_list[imin] < 1:
                storm_list[imin] = 1
                placedstorm = True
    return storm_list

def PlotAlternateCycles(solarmintimes_df):
    for i in range(6, len(solarmintimes_df['mjd'])-1):
        yy=plt.gca().get_ylim()
        if (solarmintimes_df['cyclenum'][i] % 2) == 0:
            rect = patches.Rectangle((solarmintimes_df['datetime'][i],yy[0]),
                                     solarmintimes_df['datetime'][i+1]-solarmintimes_df['datetime'][i],
                                     yy[1]-yy[0],edgecolor='none',facecolor='lightgrey',zorder=0)
            ax.add_patch(rect)
# <codecell> read in the data
#=============================================================================
#read in the aaH data file 
aaH_df = LoadaaH(aaHfilepath)

#produce 24-hour rolling mean - data is on standard 3-hour time step
#aaH_rolling = aaH_df.rolling(int(24/3), min_periods=1).mean()
#aaH_rolling['datetime'] = time.mjd2datetime(aaH_rolling['mjd'].to_numpy())

#find the max value in daily windows
#aaH_1d = aaH_rolling.resample('1D', on='datetime').max() 

aaH_1d = aaH_df.resample('1D', on='datetime').mean() 

aaH_1d['datetime'] = aaH_1d.index
aaH_1d.reset_index(drop=True, inplace=True)

aaH_27d = aaH_df.resample('27D', on='datetime').mean() 
aaH_27d['datetime'] = time.mjd2datetime(aaH_27d['mjd'].to_numpy())
aaH_27d.reset_index(drop=True, inplace=True)

aaH_1y = aaH_df.resample('1Y', on='datetime').mean() 
aaH_1y['datetime'] = time.mjd2datetime(aaH_1y['mjd'].to_numpy())
aaH_1y.reset_index(drop=True, inplace=True)

#=============================================================================
#read in the solar minimum times
solarmintimes_df = LoadSolarMinTimes(solarminfilepath)

#=============================================================================
#Read in the sunspot data 
ssn_df = LoadSSN(ssnfilepath)

#interpolate the SSN to daily values
aaH_1d['ssn'] = np.interp(aaH_1d['mjd'],ssn_df['mjd'],ssn_df['ssn'])
del ssn_df

#plt.figure()
#plt.plot(ssn_df['datetime'],ssn_df['ssn'])
#plt.plot(ssn_df['datetime'],ssn_df['smooth'])
         
#gles = load_gle_list()

# <codecell> time series processing and plots





#compute the solar cycle phase at the daily level
aaH_1d['phase'] = np.nan
for i in range(0, len(solarmintimes_df['mjd'])-1):
    smjd = solarmintimes_df['mjd'][i]
    fmjd = solarmintimes_df['mjd'][i+1]
    
    mask = (aaH_1d['mjd'] >= smjd) & (aaH_1d['mjd'] < fmjd)
    
    thiscyclemjd = aaH_1d['mjd'].loc[mask].to_numpy()
    cyclelength = fmjd - smjd
    
    aaH_1d.loc[mask,'phase'] = (thiscyclemjd - smjd)/cyclelength
              

everyyear = mdates.YearLocator(1)   # every year
every5thyear = mdates.YearLocator(5)   # every 5th year

plt.figure()
ax=plt.subplot(311)        
plt.plot(aaH_1d['datetime'],aaH_1d['ssn'],'k')
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])
plt.ylabel('Sunspot number')
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
plt.show()
#plot alternate cycles
PlotAlternateCycles(solarmintimes_df)

ax=plt.subplot(312)
plt.plot(aaH_1d['datetime'],aaH_1d['aaH'],'r',label='1 day')
plt.plot(aaH_27d['datetime'],aaH_27d['aaH'],'k',label='27 day')
plt.plot(aaH_1y['datetime'],aaH_1y['aaH'],'w',label='1 year')
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])
plt.legend(framealpha=0.5,facecolor=[0.7, 0.7, 0.7],loc='upper right')
plt.ylabel('$aa_H$ [nT]')
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
PlotAlternateCycles(solarmintimes_df)

ax=plt.subplot(313)
#compute storm occurrence at different thresholds
for plot_percentile in plot_percentiles:
    thresh = np.percentile(aaH_1d['aaH'],plot_percentile)
    print(thresh)
    
    aaH_1d['storm'] = 0
    mask = aaH_1d['aaH'] > thresh
    aaH_1d.loc[mask, 'storm'] = 1
    
    #take the annual mean
    aaH_1y = aaH_1d.resample('1Y', on='datetime').mean() 
    aaH_1y['datetime'] = aaH_1y.index
    aaH_1y.reset_index(drop=True, inplace=True)
    
    #plot it
    #plt.plot(aaH_1y['datetime'],aaH_1y['storm'],'o-',label= '$aa_H$ > {:0.0f} nT'.format(thresh) )
    plt.plot(aaH_1y['datetime'],aaH_1y['storm'],'o-',
             label= '{:0.02f}-th centile'.format(plot_percentile) )
    
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])  
plt.legend(framealpha=0.7,loc='upper left',ncol=2)
plt.ylabel('Occurrence probability [day$^{-1}$]')
plt.yscale('log')
plt.gca().set_ylim((0.002,1))
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
plt.xlabel('Year')
PlotAlternateCycles(solarmintimes_df)

plt.show
# <codecell> Super-posed epoch plots


bins=np.arange(0,1.00001,1/(nbins)) 
dbin = bins[1]-bins[0]
bin_centres = np.arange(dbin/2, 1-dbin/2 +0.01 ,dbin)

epochs_start = solarmintimes_df['mjd'][:-1].to_numpy()       
epochs_stop = solarmintimes_df['mjd'][1:].to_numpy()        
            
times = aaH_1d['mjd'].to_numpy()  
data = aaH_1d['ssn'].to_numpy()  
spe_ssn = SPE_dualepoch(times, data, epochs_start, epochs_stop, bins)         
      
times = aaH_1d['mjd'].to_numpy()  
data = aaH_1d['aaH'].to_numpy()  
spe_aaH = SPE_dualepoch(times, data, epochs_start, epochs_stop, bins)  
      


#plt1=plt.subplot(212)
#plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3])

#find the cut off solar cycle phase. Use FWHM
# ssn_thresh = spe_ssn[:,1].max()/2 
# mask_smin = spe_ssn[:,1] < ssn_thresh
# mask_smax = spe_ssn[:,1] >= ssn_thresh

#record teh phase boundaries for solar min/max
# phase_lower = bin_centres[mask_smax].min()
# phase_upper = bin_centres[mask_smax].max()
phase_lower = 0.18
phase_upper = 0.79

phase_mid = phase_lower + (phase_upper-phase_lower)/2

amplitude = np.empty(len(solarmintimes_df['mjd']))
aaH_1d['parity'] = 0
#compute the mean SSN for each cycle
for i in range(0, len(solarmintimes_df['mjd'])-1):
    mask = (aaH_1d['mjd'] >= solarmintimes_df['mjd'][i]) & (aaH_1d['mjd'] < solarmintimes_df['mjd'][i+1])
    amplitude[i] =  aaH_1d['ssn'].loc[mask].mean()
    
    #also add a flag for odd/even cycles
    mask = (aaH_1d['mjd'] >= solarmintimes_df['mjd'][i]) & (aaH_1d['mjd'] < solarmintimes_df['mjd'][i+1])
    
    if solarmintimes_df['cyclenum'][i] % 2:
         aaH_1d.loc[mask,'parity'] = -1
    else:
         aaH_1d.loc[mask,'parity'] = 1
         

        
amplitude[i+1]=np.nan    
solarmintimes_df['amplitude'] = amplitude
    

#SPE of SSN for odd/even
#==============================================================================
ssndata = aaH_1d[aaH_1d['parity'] == 1]['ssn'].to_numpy()
times = aaH_1d[aaH_1d['parity'] == 1]['mjd'].to_numpy()
spe_ssneven = SPE_dualepoch(times, ssndata, epochs_start, epochs_stop, bins)    
ssndata = aaH_1d[aaH_1d['parity'] == -1]['ssn'].to_numpy()
times = aaH_1d[aaH_1d['parity'] == -1]['mjd'].to_numpy()
spe_ssnodd = SPE_dualepoch(times, ssndata, epochs_start, epochs_stop, bins)  

plt.figure()


plt1=plt.subplot(221)
plt.errorbar(spe_ssn[:,0],spe_ssn[:,1], yerr = spe_ssn[:,3],fmt='k',label='All data')
plt.errorbar(spe_ssneven[:,0],spe_ssneven[:,1], yerr = spe_ssneven[:,3],fmt='r',label='Even cycles')
plt.errorbar(spe_ssnodd[:,0],spe_ssnodd[:,1], yerr = spe_ssnodd[:,3],fmt='b',label='Odd cycles')
plt.ylabel('SSN')
plt.xlabel('Solar cycle phase')
#plt.plot(spe_ssn[:,0],ssn_thresh + 0*spe_ssn[:,0],'k')
yy=plt.gca().get_ylim()
plt.gca().set_ylim(yy)
#plt.plot([phase_lower,phase_lower],yy,'k--')
#plt.plot([phase_upper,phase_upper],yy,'k--')
plt.title('(a) Sunspot number')
plt.legend()
yy=plt.gca().get_ylim()
rect = patches.Rectangle((phase_lower, yy[0]), phase_upper-phase_lower, yy[1]-yy[0],
                         edgecolor='none',facecolor='lightgrey')
plt1.add_patch(rect)
rect = patches.Rectangle((phase_mid, yy[0]), phase_upper-phase_mid, yy[1]-yy[0],
                             edgecolor='none',facecolor='darkgrey')
plt1.add_patch(rect)    


#super-posed epoch plots of storm occurrence
#==============================================================================
minval = np.percentile(aaH_1d['aaH'],90) #aaH_1d_mean['aaH'].median()
maxval = np.percentile(aaH_1d['aaH'],99.99)
storm_thresh = np.arange(minval,maxval+1,(maxval-minval)/nthreshbins)


for nsubplot in range(0,3):
    if nsubplot == 0:
        plt1=plt.subplot(222)
        aaHdata = aaH_1d.copy()
        title = '(b) Storms: All data'
        tcol = 'k'
        ax=plt1
    elif nsubplot == 1:
        plt2=plt.subplot(223)
        aaHdata = aaH_1d[aaH_1d['parity'] == 1].copy()
        title = '(c) Storms: Even cycles'
        tcol = 'r'
        ax=plt2
    elif nsubplot == 2:
        plt3=plt.subplot(224)
        ax=plt3
        aaHdata = aaH_1d[aaH_1d['parity'] == -1].copy()
        title = '(d) Storms: Odd cycles'
        tcol = 'b'
        
        
        
    storm_occur = np.empty((nbins,len(plot_percentiles)))
    storm_occur_norm = np.empty((nbins,len(plot_percentiles)))
    storm_prob = np.empty((len(plot_percentiles),3))
    for i in range(0,len(plot_percentiles)):
        thresh = np.percentile(aaHdata['aaH'],plot_percentiles[i])
    
        aaHdata.loc[:,'storm'] = 0
        mask = aaHdata['aaH'] >= thresh
        aaHdata.loc[mask,'storm'] = 1
        
        times = aaHdata['mjd'].to_numpy()
        data = aaHdata['storm'].to_numpy()
        print(title, str(np.sum(data==1)))
        
        spe_aaH = SPE_dualepoch(times, data, epochs_start, epochs_stop, bins) 
        storm_occur[:,i] = spe_aaH[:,1]
        storm_occur_norm[:,i] = (spe_aaH[:,1] ) / (spe_aaH[:,1].max() )
        
        # storm_prob[i,0] = thresh
        # storm_prob[i,1] = spe_aaH[mask_smin,1].mean()
        # storm_prob[i,2] = spe_aaH[mask_smax,1].mean()
    
        #plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
        #              label= '$aa_H$ > {:0.0f} nT'.format(thresh))
        plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
                      label= '{:0.02f}-th centile'.format(plot_percentiles[i]))
    plt.yscale('log')
    
    plt.gca().set_ylim((0.0005,1))
    yy=plt.gca().get_ylim()
    #plt.plot([phase_lower,phase_lower],yy,'k--')
    #plt.plot([phase_upper,phase_upper],yy,'k--')
    plt.ylabel('Probability [day$^{-1}$]')
    plt.title(title, color=tcol)
    
    plt.xlabel('Solar cycle phase')
    
    yy=plt.gca().get_ylim()
    rect = patches.Rectangle((phase_lower, yy[0]), phase_upper-phase_lower, yy[1]-yy[0],
                             edgecolor='none',facecolor='lightgrey')
    ax.add_patch(rect)
    rect = patches.Rectangle((phase_mid, yy[0]), phase_upper-phase_mid, yy[1]-yy[0],
                             edgecolor='none',facecolor='darkgrey')
    ax.add_patch(rect)
    
plt1.legend(framealpha=1,loc='upper left',ncol=2)
plt1.yaxis.set_ticks_position("right")
plt3.yaxis.set_ticks_position("right")
plt1.yaxis.set_label_position("right")
plt3.yaxis.set_label_position("right")


# plt.figure()
# plt1=plt.subplot(221)
# plt.pcolor(storm_thresh,spe_aaH[:,0],storm_occur)
# plt.ylabel('Solar cycle phase')
# plt.xlabel('Storm threshold')
# plt.title('Daily max aaH')

# plt1=plt.subplot(222)
# plt.pcolor(storm_thresh,spe_aaH[:,0],storm_occur_norm)
# plt.ylabel('Solar cycle phase')
# plt.xlabel('Storm threshold')


# plt1=plt.subplot(223)
# plt.plot(storm_prob[:,0],np.log(storm_prob[:,1]),label='Solar min')
# plt.plot(storm_prob[:,0],np.log(storm_prob[:,2]),label='Solar max')
# plt.xlabel('Storm threshold')
# plt.ylabel('log(Occurrence probability)')
# plt.legend()

# plt1=plt.subplot(224)
# plt.plot(storm_prob[:,0],np.log(storm_prob[:,1]),label='Solar min')
# plt.plot(storm_prob[:,0],np.log(storm_prob[:,2]),label='Solar max')
# plt.xlabel('Storm threshold')
# plt.ylabel('log(Occurrence probability)')
# plt.legend()

# # <codecell> Super-posed epoch plots of annual variations


# bins=np.arange(0,1.00001,1/(nbins)) 
# dbin = bins[1]-bins[0]
# bin_centres = np.arange(dbin/2, 1-dbin/2 +0.01 ,dbin)


# firstyear =  aaH_1d['datetime'].iloc[0].year + 1
# lastyear =  aaH_1d['datetime'].iloc[-1].year -1
# years_start = time.date2mjd(np.arange(firstyear, lastyear-1,1),1,1)      
# years_stop = time.date2mjd(np.arange(firstyear+1, lastyear,1),1,1)
            
# times = aaH_1d['mjd'].to_numpy()  
# data = aaH_1d['ssn'].to_numpy()  
# spe_ssn = SPE_dualepoch(times, data, years_start, years_stop, bins)         
      
# times = aaH_1d['mjd'].to_numpy()  
# data = aaH_1d['aaH'].to_numpy()  
# spe_aaH = SPE_dualepoch(times, data, years_start, years_stop, bins)  
      


    

# #SPE for odd/even
 
# ssndata = aaH_1d[aaH_1d['parity'] == 1]['ssn'].to_numpy()
# times = aaH_1d[aaH_1d['parity'] == 1]['mjd'].to_numpy()
# spe_ssneven = SPE_dualepoch(times, ssndata, years_start, years_stop, bins)    
# ssndata = aaH_1d[aaH_1d['parity'] == -1]['ssn'].to_numpy()
# times = aaH_1d[aaH_1d['parity'] == -1]['mjd'].to_numpy()
# spe_ssnodd = SPE_dualepoch(times, ssndata, years_start, years_stop, bins)  

# #produce the super-posed epoch plots
# #==============================================================================


# plt.figure()


# plt1=plt.subplot(221)
# plt.errorbar(spe_ssn[:,0],spe_ssn[:,1], yerr = spe_ssn[:,3],fmt='k',label='All data')
# plt.errorbar(spe_ssneven[:,0],spe_ssneven[:,1], yerr = spe_ssneven[:,3],fmt='r',label='Even cycles')
# plt.errorbar(spe_ssnodd[:,0],spe_ssnodd[:,1], yerr = spe_ssnodd[:,3],fmt='b',label='Odd cycles')
# plt.ylabel('SSN')
# plt.title('(a) Sunspot number')
# plt.legend()



# for nsubplot in range(0,3):
#     if nsubplot == 0:
#         plt1=plt.subplot(222)
#         aaHdata = aaH_1d.copy()
#         title = '(b) Storms: All data'
#         tcol = 'k'
#         ax=plt1
#     elif nsubplot == 1:
#         plt2=plt.subplot(223)
#         aaHdata = aaH_1d[aaH_1d['parity'] == 1].copy()
#         title = '(c) Storms: Even cycles'
#         tcol = 'r'
#         ax=plt2
#     elif nsubplot == 2:
#         plt3=plt.subplot(224)
#         ax=plt3
#         aaHdata = aaH_1d[aaH_1d['parity'] == -1].copy()
#         title = '(d) Storms: Odd cycles'
#         tcol = 'b'
       
#     for i in range(0,len(plot_percentiles)):
#         thresh = np.percentile(aaHdata['aaH'],plot_percentiles[i])
    
#         aaHdata.loc[:,'storm'] = 0
#         mask = aaHdata['aaH'] > thresh
#         aaHdata.loc[mask,'storm'] = 1
    
#         times = aaHdata['mjd'].to_numpy()
#         data = aaHdata['storm'].to_numpy()
#         spe_aaH = SPE_dualepoch(times, data, years_start, years_stop, bins) 

    
#         #plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
#         #              label= '$aa_H$ > {:0.0f} nT'.format(thresh))
#         plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
#                       label= '{:0.02f}-th centile'.format(plot_percentiles[i]))
        
#     plt.yscale('log')
#     plt.ylabel('Probability [day$^{-1}$]')
#     plt.xlabel('Fraction of year')
#     plt.title(title, color=tcol)
#     plt.gca().set_ylim((0.0005,1))
    
    
# plt1.legend(framealpha=1,loc='upper left',ncol=2)
# plt1.yaxis.set_ticks_position("right")
# plt3.yaxis.set_ticks_position("right")
# plt1.yaxis.set_label_position("right")
# plt3.yaxis.set_label_position("right")

# <codecell> plot of solar min/max storm freq
min_mask = ( (aaH_1d['phase'] < phase_lower) | (aaH_1d['phase'] > phase_upper))
max_mask = ( (aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] <= phase_upper))

early_mask = ( (aaH_1d['phase'] < phase_lower) | (aaH_1d['phase'] > phase_mid))
late_mask = ( (aaH_1d['phase'] >= phase_mid) & (aaH_1d['phase'] <= phase_upper))

storm_prob_minmax = np.empty((len(storm_thresh),4))
for i in range(0,len(storm_thresh)):

    aaH_1d['storm'] = 0
    mask = aaH_1d['aaH'] > storm_thresh[i]
    aaH_1d.loc[mask,'storm'] = 1

    storm_prob_minmax[i,0] = storm_thresh[i]
    storm_prob_minmax[i,1] = aaH_1d.loc[min_mask,'storm'].mean()
    storm_prob_minmax[i,2] = aaH_1d.loc[max_mask,'storm'].mean()
    storm_prob_minmax[i,3] = storm_prob_minmax[i,2] - storm_prob_minmax[i,1]
    
    Nstorms =  sum(aaH_1d['aaH'] >= storm_thresh[i]) 
    
    print(storm_thresh[i], Nstorms, storm_prob_minmax[i,2]/storm_prob_minmax[i,1])
    
plt.figure()
plt1=plt.subplot(131)
plt.plot(storm_prob_minmax[:,0],(storm_prob_minmax[:,1]),label='Break')
plt.plot(storm_prob_minmax[:,0],(storm_prob_minmax[:,2]),label='Active')
plt.xlabel('Storm threshold')
plt.ylabel('log(probability) [day$^{-1}$]')
plt.legend() 
plt.yscale('log')
plt.xlim(20,300)
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:  
    Nstorms =  sum(aaH_1d['aaH'] >= np.percentile(aaH_1d['aaH'],plot_percentile))  
    plt.plot(np.percentile(aaH_1d['aaH'],plot_percentile)*np.ones(2),[0,yval],'k--')
    t = plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_1d['aaH']-dx,plot_percentile), yval),
                 ha='center')   
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='none'))
    
    
plt1=plt.subplot(132)
plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,3],'k')
plt.xlabel('Storm threshold')
plt.ylabel('Active-break log(probability) [day$^{-1}$]')
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:  
    Nstorms =  sum(aaH_1d['aaH'] >= np.percentile(aaH_1d['aaH'],plot_percentile))  
    plt.plot(np.percentile(aaH_1d['aaH'],plot_percentile)*np.ones(2),[0,yval],'k--')
    t = plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_1d['aaH']-dx,plot_percentile), yval),
                 ha='center')   
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='none'))
plt.yscale('log')
plt.xlim(20,300)


plt1=plt.subplot(133)
plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,2]/storm_prob_minmax[:,1],'k')
plt.xlabel('Storm threshold')
plt.ylabel('Active/break probability')
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:  
    Nstorms =  sum(aaH_1d['aaH'] >= np.percentile(aaH_1d['aaH'],plot_percentile))  
    plt.plot(np.percentile(aaH_1d['aaH'],plot_percentile)*np.ones(2),[0,yval],'k--')
    t = plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_1d['aaH']-dx,plot_percentile), yval),
                 ha='center')   
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='none'))

plt.xlim(20,300)


print(np.nanmean(storm_prob_minmax[:,2]/storm_prob_minmax[:,1]))

# <codecell> effect of solar cycle amplitude
meanssn=np.nanmean(aaH_1d['ssn'])

cycle_total = np.empty((len(solarmintimes_df['mjd'])-1, len(storm_thresh)+2))
correl = np.empty((len(storm_thresh),2))
for i in range(0, len(storm_thresh)):
    
    #define storms for this threshold
    aaH_1d['storm'] = 0
    mask = aaH_1d['aaH'] > storm_thresh[i]
    aaH_1d.loc[mask, 'storm'] = 1
    
    #loop over each cycle and find the average storm occurrence rate
    for n in range(0,len(solarmintimes_df['mjd'])-1 ):
        cycle_total[n,0] = solarmintimes_df['cyclenum'][n]
        cycle_total[n,1] = solarmintimes_df['amplitude'][n] /meanssn
        
        mask = (aaH_1d['mjd'] >= solarmintimes_df['mjd'][n]) & \
              (aaH_1d['mjd'] < solarmintimes_df['mjd'][n+1])
        cycle_total[n,i+2] = aaH_1d['storm'].loc[mask].mean()
    
    #compute the covariance
    mask = ~np.isnan(cycle_total[:,i+2])
    r_i = pearsonr(cycle_total[mask,1],cycle_total[mask,i+2])
    correl[i,:] = r_i

      
# plt.figure()
# plt1=plt.subplot(121)
# for i in range(0, len(storm_thresh)):
#     plt.plot(cycle_total[:,1],np.log(cycle_total[:,i+2]),
#              'o',label='aa_H >' +str(storm_thresh[i]))
# #plt.legend()
# plt.xlabel('Solar cycle amplitude')
# plt.ylabel('log(Occurrence prob)')
    
    
   
# plt1=plt.subplot(122)
# plt.plot(storm_thresh,correl[:,0], label='Correlation (cycle amplitude - occurrence)')
# plt.plot(storm_thresh,correl[:,1], label='p value')
# plt.xlabel('aa_H threshold')
# plt.legend()

#scatter plots for the plot percentiles
#==============================================================================
plt.figure()
cycle_total = np.empty((len(solarmintimes_df['mjd'])-1, len(plot_percentiles)+2))
for i in range(0,len(plot_percentiles)):
    thresh = np.percentile(aaH_1d['aaH'],plot_percentiles[i])
    Nstorms =  sum(aaH_1d['aaH'] >= np.percentile(aaH_1d['aaH'],plot_percentiles[i]))  
    #define storms for this threshold
    aaH_1d['storm'] = 0
    mask = aaH_1d['aaH'] > thresh
    aaH_1d.loc[mask, 'storm'] = 1
    
    #loop over each cycle and find the average storm occurrence rate
    for n in range(0,len(solarmintimes_df['mjd'])-1 ):
        cycle_total[n,0] = solarmintimes_df['cyclenum'][n]
        cycle_total[n,1] = solarmintimes_df['amplitude'][n] / meanssn
        
        mask = (aaH_1d['mjd'] >= solarmintimes_df['mjd'][n]) & \
             (aaH_1d['mjd'] < solarmintimes_df['mjd'][n+1])
        cycle_total[n,i+2] = aaH_1d['storm'].loc[mask].mean() 
    
    #compute the covariance
    mask = ~np.isnan(cycle_total[:,i+2])
    r_i = pearsonr(cycle_total[mask,1],cycle_total[mask,i+2])
    #correl[i,:] = r_i
    
    ax = plt.subplot(2,2,i+1)
    plt.plot(cycle_total[:,1],cycle_total[:,i+2],'ko')
    plt.ylabel('Cycle-average probability [day$^{-1}$]')
   
    ymax = np.nanmax(cycle_total[:,i+2])
    yspan = ymax - np.nanmin(cycle_total[:,i+2])
    plt.annotate('{:0.02f}-th centile'.format(plot_percentiles[i]) + ' \nN=' 
                 + str(Nstorms) +'\n' + 'r = {:0.02f}'.format(r_i[0])  
                 + ' (p = {:0.04f}'.format(r_i[1]) + ')',
                 xy=(52, ymax-yspan/8),   ha='left')   
   
    if i % 2:
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
    if i>1:
        plt.xlabel('Cycle-average sunspot number')
    # if i<2:
    #     ax.get_xaxis().set_ticklabels([])



# <codecell> Models of storm occurrence


#assume extreme events perfectly follow the solar cycle


aaH_model = aaH_1d.copy()
storm_thresh_model = storm_thresh

#generate the relative probability time series

#function of phase only
aaH_model['rel_prob_phase'] = 1
aaH_model.loc[max_mask,'rel_prob_phase'] = active_amp

#random
aaH_model['rel_prob_rand'] = np.nanmean( aaH_model['rel_prob_phase'] )


#phase and amplitude
aaH_model['rel_prob_phaseamp'] = 1 


for n in range(0,len(solarmintimes_df['mjd'])-1 ):
    amp = solarmintimes_df['amplitude'][n]
        
    mask = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
             (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
              ((aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] <= phase_upper))
    #aaH_model.loc[mask,'rel_prob_phaseamp'] =  amp * active_amp/meanssn
    aaH_model.loc[mask,'rel_prob_phaseamp'] =  active_amp * (1 - cycle_amp + cycle_amp * amp/meanssn)
    

#odd/even difference
aaH_model['rel_prob_oddeven'] = 1

#add the solar cycle probabilities
for n in range(0,len(solarmintimes_df['mjd'])-1 ):
    amp = solarmintimes_df['amplitude'][n]
        
    mask = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
             (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
              ((aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] <= phase_upper))
    #aaH_model.loc[mask,'rel_prob_oddeven'] = amp * active_amp/meanssn
    aaH_model.loc[mask,'rel_prob_oddeven'] = active_amp * (1 - cycle_amp + cycle_amp * amp/meanssn)

#Then adjust the late/early portion depending on the cycle number
for n in range(0,len(solarmintimes_df['mjd'])-1 ):
    amp = solarmintimes_df['amplitude'][n]
        
    mask_early = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
             (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
              ((aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] < phase_mid))
    mask_late = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
             (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
              ((aaH_1d['phase'] >= phase_mid) & (aaH_1d['phase'] <= phase_upper))
              
    if solarmintimes_df['cyclenum'][n] % 2:
        aaH_model.loc[mask_early,'rel_prob_oddeven'] = aaH_model.loc[mask_early,'rel_prob_oddeven'] *(1-d_earlylate)
        aaH_model.loc[mask_late,'rel_prob_oddeven'] = aaH_model.loc[mask_late,'rel_prob_oddeven'] * (1+d_earlylate)
    else:
        aaH_model.loc[mask_early,'rel_prob_oddeven'] = aaH_model.loc[mask_early,'rel_prob_oddeven'] * (1+d_earlylate)
        aaH_model.loc[mask_late,'rel_prob_oddeven'] = aaH_model.loc[mask_late,'rel_prob_oddeven'] * (1-d_earlylate)
        
        

model_name_list = ['rand','phase','phaseamp','oddeven']

#CDF for plotting only
for model in model_name_list:
    aaH_model['rel_prob_cdf_'+model] = 0
    for i in range(1,len(aaH_model)):
        aaH_model['rel_prob_cdf_' +model].values[i] = \
            aaH_model['rel_prob_cdf_'+model][i-1] + aaH_model['rel_prob_'+model][i]
    #normalise the rel_prob values
    aaH_model['rel_prob_'+model] = aaH_model['rel_prob_'+model] /aaH_model['rel_prob_cdf_'+model].iloc[-1]    
    #normalise the cdf
    aaH_model['rel_prob_cdf_'+model] =(aaH_model['rel_prob_cdf_'+model] /  
                                 aaH_model['rel_prob_cdf_'+model].iloc[-1] )
    

    

plt.figure()
ax=plt.subplot(211)
plt.fill_between(aaH_model['datetime'],aaH_model['rel_prob_phase']*0,aaH_model['rel_prob_phase'],
                 color='red',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_rand'],'b',label='Random model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_phaseamp'],'k',label='Phase+Amp model')
#plt.plot(aaH_model['datetime'],aaH_model['rel_prob_phase'],'r--',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_oddeven'],'k--',label='EarlyLate model')
plt.ylim(0,)
plt.ylabel('Relative probability')
plt.legend()
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='b')
ax.tick_params(which='major', length=10, color='k')
PlotAlternateCycles(solarmintimes_df)
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])

ax=plt.subplot(212)
plt.fill_between(aaH_model['datetime'],aaH_model['rel_prob_cdf_phase']*0,aaH_model['rel_prob_cdf_phase'],
                 color='red',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_cdf_rand'],'b',label='Random model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_cdf_phaseamp'],'k',label='Phase+Amp model')
#plt.plot(aaH_model['datetime'],aaH_model['rel_prob_cdf_phase'],'r--',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_cdf_oddeven'],'k--',label='EarlyLate model')
plt.ylim(0,1)
plt.ylabel('Cumulative probability')
plt.xlabel('Year')
plt.legend()
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
PlotAlternateCycles(solarmintimes_df)
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])

# <codecell> Project models forward




forecast=pd.DataFrame()
startmjd=time.date2mjd(2021,1,1)[0]
forecast['mjd'] = np.arange(startmjd, startmjd+11*365,1)
forecast['datetime'] = time.mjd2datetime(forecast['mjd'].to_numpy())


#produce the probabilities for each demo solar cycle
for i in range(0,len(demoSCs)):
    #find the right SC
    n = solarmintimes_df[solarmintimes_df['cyclenum']==demoSCs[i]].index[0]
    sclength= solarmintimes_df['mjd'][n+1] - solarmintimes_df['mjd'][n]
    
    stopmjd = startmjd + sclength
    #create the times and phases using the solar cycle length
    forecast['phase'] = 0
    mask = ((forecast['mjd']>=startmjd) & (forecast['mjd'] <=stopmjd))
    forecast.loc[mask,'phase'] = (forecast.loc[mask,'mjd'] - startmjd)/sclength
    
    #get the probabilities
    mask = ((aaH_model['mjd']>=solarmintimes_df['mjd'][n]) & 
            (aaH_model['mjd'] <=solarmintimes_df['mjd'][n+1]))
    pquiet = np.min(aaH_model.loc[mask,'rel_prob_phaseamp'])
    pactive = np.max(aaH_model.loc[mask,'rel_prob_phaseamp'])
    #for cycle 25, the early phase will have enhanced probability
    plate = np.max(aaH_model.loc[mask,'rel_prob_oddeven'])
    pearly = pactive - (plate - pactive)
    
    #generate the forecast probability - quiet/active difference
    forecast['rel_prob'] = pquiet
    mask = ((forecast['phase']>=phase_lower) & 
            (forecast['phase']<=phase_upper))
    forecast.loc[mask,'rel_prob'] = pactive
    forecast['rel_prob'+str(i)] = forecast['rel_prob']
    
    #add the early/late difference for most extreme events
    forecast['rel_prob_extreme'] = forecast['rel_prob']
    mask = ((forecast['phase']>=phase_lower) & 
            (forecast['phase']<phase_mid))
    forecast.loc[mask,'rel_prob_extreme'] = pearly
    mask = ((forecast['phase']>=phase_mid) & 
            (forecast['phase']<=phase_upper))
    forecast.loc[mask,'rel_prob_extreme'] = plate
    
    forecast['rel_prob_extreme'+str(i)] = forecast['rel_prob_extreme']

# fig = plt.figure
# for i in range(0,len(demoSCs)):
#     plt.plot(forecast['datetime'],forecast['rel_prob'+str(i)],
#              label='SC'+str(demoSCs[i])+' amplitude and length')
# plt.legend()
# plt.ylim(0,)
# plt.ylabel('Relative probability')


#the rel_prob is normalised to give a probablity of 1 over the whole aa interval
#So to produce the prob per day for a threshold that has N events in that interval, 
#mulitple rel_prob by N
fig = plt.figure(figsize=(12, 5))

#first plot the sunspot number
linestyles = ['r','b','k--']
linewidths=[3,1,1]
ax = plt.subplot(1,3,1)
for i in range(0,len(demoSCs)):
    n = solarmintimes_df[solarmintimes_df['cyclenum']==demoSCs[i]].index[0]
    mask = ((aaH_model['mjd']>=solarmintimes_df['mjd'][n]) & 
            (aaH_model['mjd'] <=solarmintimes_df['mjd'][n+1]))
    
    smjd = solarmintimes_df['mjd'][n] 
    times = aaH_model.loc[mask,'mjd'] - smjd + startmjd
    
    plt.plot(time.mjd2datetime(times.to_numpy()),
             aaH_model.loc[mask,'ssn'],linestyles[i],
             linewidth=linewidths[i],
                 label='SC'+str(demoSCs[i]))
plt.legend()  
plt.ylabel('Sunspot number', fontsize=16)
plt.xlabel('Year',fontsize=16)
plt.ylim(0,)  
plt.xlim(datetime(2021,1,1),datetime(2032,1,1)) 


#Now plot the 99th percentil storms
ax = plt.subplot(1,3,2)
thresh = np.percentile(aaH_1d['aaH'],plot_percentiles[1])
Nstorms =  sum(aaH_1d['aaH'] >= thresh)      
plt.title(str(plot_percentiles[1]) 
              + 'th percentile\n $aa_H$ > {:0.0f} nT'.format(thresh) )    
for j in range(0,len(demoSCs)):
    plt.plot(forecast['datetime'],forecast['rel_prob'+str(j)] * Nstorms,
             linestyles[j],
             linewidth=linewidths[j],
             label='SC'+str(demoSCs[j])+' amplitude and length')
plt.ylabel('Probability [day$^{-1}$]',fontsize=16)
plt.xlabel('Year',fontsize=16)
plt.ylim(0,)
plt.xlim(datetime(2021,1,1),datetime(2032,1,1)) 


#Now plot the 99.99th percentil storms
ax = plt.subplot(1,3,3)
thresh = np.percentile(aaH_1d['aaH'],plot_percentiles[-1])
Nstorms =  sum(aaH_1d['aaH'] >= thresh)      
plt.title(str(plot_percentiles[-1]) 
              + 'th percentile\n $aa_H$ > {:0.0f} nT'.format(thresh) )    
for j in range(0,len(demoSCs)):
    plt.plot(forecast['datetime'],forecast['rel_prob_extreme'+str(j)] * Nstorms,
             linestyles[j],
             linewidth=linewidths[j],
             label='SC'+str(demoSCs[j])+' amplitude and length')
plt.ylabel('Probability [day$^{-1}$]',fontsize=16)
plt.xlabel('Year',fontsize=16)
plt.ylim(0,)
plt.xlim(datetime(2021,1,1),datetime(2032,1,1)) 

plt.tight_layout()


# <codecell> Monte Carlo models
early_mask_odd = ( (aaH_model['phase'] > phase_lower) & (aaH_model['phase'] < phase_mid)) \
                & (aaH_model['parity'] == -1)
late_mask_odd = ( (aaH_model['phase'] >= phase_mid) & (aaH_model['phase'] <= phase_upper)) \
                & (aaH_model['parity'] == -1)
early_mask_even = ( (aaH_model['phase'] > phase_lower) & (aaH_model['phase'] < phase_mid)) \
                & (aaH_model['parity'] == 1)
late_mask_even = ( (aaH_model['phase'] >= phase_mid) & (aaH_model['phase'] <= phase_upper)) \
                & (aaH_model['parity'] == 1)

for model in model_name_list:


    #model where storm occurrence is function of phase only
    model_r = np.empty((Nmc,len(storm_thresh_model)))
    model_min = np.empty((Nmc,len(storm_thresh_model)))
    model_max = np.empty((Nmc,len(storm_thresh_model)))
    model_diff = np.empty((Nmc,len(storm_thresh_model)))
    model_ratio = np.empty((Nmc,len(storm_thresh_model)))
    
    model_early_odd = np.empty((Nmc,len(storm_thresh_model)))
    model_late_odd = np.empty((Nmc,len(storm_thresh_model)))
    model_diffparity_odd = np.empty((Nmc,len(storm_thresh_model)))
    
    model_early_even = np.empty((Nmc,len(storm_thresh_model)))
    model_late_even = np.empty((Nmc,len(storm_thresh_model)))
    model_diffparity_even = np.empty((Nmc,len(storm_thresh_model)))
    
    for nthresh in range(0,len(storm_thresh_model)):
        print('Model:' + model + '; MC simulation of threshold ' +str(nthresh+1) 
              + ' of ' +str(len(storm_thresh_model)) )
        Nstorms =  sum(aaH_model['aaH'] >= storm_thresh_model[nthresh]) 
        
        for Ncount in range(0,Nmc):
            aaH_model['storm'] = generate_events(aaH_model['rel_prob_'+model].to_numpy(), Nstorms)
                        
            cycle_total_model = np.empty((len(solarmintimes_df['mjd'])-1, 3))
            #loop over each cycle and find the average storm occurrence rate
            for n in range(0,len(solarmintimes_df['mjd'])-1 ):
                cycle_total_model[n,0] = solarmintimes_df['cyclenum'][n]
                cycle_total_model[n,1] = solarmintimes_df['amplitude'][n]
                
                mask = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
                     (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1])
                cycle_total_model[n,2] = aaH_model['storm'].loc[mask].mean()
            
            #compute the correlation of cycle amplitude and storm occurrence
            mask = ~np.isnan(cycle_total_model[:,2])
            model_r[Ncount,nthresh], _  = pearsonr(cycle_total_model[mask,1],cycle_total_model[mask,2])
            
            #compute the mean values for solar min/max
            model_min[Ncount,nthresh] = aaH_model.loc[min_mask,'storm'].mean()
            model_max[Ncount,nthresh] = aaH_model.loc[max_mask,'storm'].mean()
            model_diff[Ncount,nthresh] = model_max[Ncount,nthresh] - model_min[Ncount,nthresh]
            
            #compute the early/late probs for even and odd
            
            model_early_odd[Ncount,nthresh] = aaH_model.loc[early_mask_odd,'storm'].mean()
            model_late_odd[Ncount,nthresh] = aaH_model.loc[late_mask_odd,'storm'].mean() 
            model_diffparity_odd[Ncount,nthresh] = model_early_odd[Ncount,nthresh] - model_late_odd[Ncount,nthresh]
            
            
            model_early_even[Ncount,nthresh] = aaH_model.loc[early_mask_even,'storm'].mean()
            model_late_even[Ncount,nthresh] = aaH_model.loc[late_mask_even,'storm'].mean() 
            model_diffparity_even[Ncount,nthresh] = model_early_even[Ncount,nthresh] - model_late_even[Ncount,nthresh]
           
            #if model_phase_min[Ncount,nthresh] >0 :

            #else:
            #   model_phase_ratio[Ncount,nthresh] = model_phase_max[Ncount,nthresh] / 0.000001
    
        
    #find the median and 1- and 2- sigma percentiles of the Monte Carlo runs
    model_mc = np.empty((len(storm_thresh_model),5))
    model_min_mc = np.empty((len(storm_thresh_model),5))
    model_max_mc = np.empty((len(storm_thresh_model),5))
    model_diff_mc = np.empty((len(storm_thresh_model),5))
    model_early_odd_mc = np.empty((len(storm_thresh_model),5))
    model_late_odd_mc = np.empty((len(storm_thresh_model),5))
    model_diffparity_odd_mc = np.empty((len(storm_thresh_model),5))
    model_early_even_mc = np.empty((len(storm_thresh_model),5))
    model_late_even_mc = np.empty((len(storm_thresh_model),5))
    model_diffparity_even_mc = np.empty((len(storm_thresh_model),5))
    
    for nthresh in range(0,len(storm_thresh_model)):
        model_mc[nthresh,0] = np.percentile(model_r[:,nthresh],50)
        model_mc[nthresh,1] = np.percentile(model_r[:,nthresh],32)
        model_mc[nthresh,2] = np.percentile(model_r[:,nthresh],69)
        model_mc[nthresh,3] = np.percentile(model_r[:,nthresh],5)
        model_mc[nthresh,4] = np.percentile(model_r[:,nthresh],95)
        
        model_min_mc[nthresh,0] = np.percentile(model_min[:,nthresh],50)
        model_min_mc[nthresh,1] = np.percentile(model_min[:,nthresh],32)
        model_min_mc[nthresh,2] = np.percentile(model_min[:,nthresh],68)
        model_min_mc[nthresh,3] = np.percentile(model_min[:,nthresh],5)
        model_min_mc[nthresh,4] = np.percentile(model_min[:,nthresh],95)
        
        model_max_mc[nthresh,0] = np.percentile(model_max[:,nthresh],50)
        model_max_mc[nthresh,1] = np.percentile(model_max[:,nthresh],32)
        model_max_mc[nthresh,2] = np.percentile(model_max[:,nthresh],68)
        model_max_mc[nthresh,3] = np.percentile(model_max[:,nthresh],5)
        model_max_mc[nthresh,4] = np.percentile(model_max[:,nthresh],95)
        
        model_diff_mc[nthresh,0] = np.percentile(model_diff[:,nthresh],50)
        model_diff_mc[nthresh,1] = np.percentile(model_diff[:,nthresh],32)
        model_diff_mc[nthresh,2] = np.percentile(model_diff[:,nthresh],68)
        model_diff_mc[nthresh,3] = np.percentile(model_diff[:,nthresh],5)
        model_diff_mc[nthresh,4] = np.percentile(model_diff[:,nthresh],95)
        
        model_early_odd_mc[nthresh,0] = np.percentile(model_early_odd[:,nthresh],50)
        model_early_odd_mc[nthresh,1] = np.percentile(model_early_odd[:,nthresh],32)
        model_early_odd_mc[nthresh,2] = np.percentile(model_early_odd[:,nthresh],68)
        model_early_odd_mc[nthresh,3] = np.percentile(model_early_odd[:,nthresh],5)
        model_early_odd_mc[nthresh,4] = np.percentile(model_early_odd[:,nthresh],95)
        
        model_late_odd_mc[nthresh,0] = np.percentile(model_late_odd[:,nthresh],50)
        model_late_odd_mc[nthresh,1] = np.percentile(model_late_odd[:,nthresh],32)
        model_late_odd_mc[nthresh,2] = np.percentile(model_late_odd[:,nthresh],68)
        model_late_odd_mc[nthresh,3] = np.percentile(model_late_odd[:,nthresh],5)
        model_late_odd_mc[nthresh,4] = np.percentile(model_late_odd[:,nthresh],95)
        
        model_diffparity_odd_mc[nthresh,0] = np.percentile(model_diffparity_odd[:,nthresh],50)
        model_diffparity_odd_mc[nthresh,1] = np.percentile(model_diffparity_odd[:,nthresh],32)
        model_diffparity_odd_mc[nthresh,2] = np.percentile(model_diffparity_odd[:,nthresh],68)
        model_diffparity_odd_mc[nthresh,3] = np.percentile(model_diffparity_odd[:,nthresh],5)
        model_diffparity_odd_mc[nthresh,4] = np.percentile(model_diffparity_odd[:,nthresh],95)
        
        model_early_even_mc[nthresh,0] = np.percentile(model_early_even[:,nthresh],50)
        model_early_even_mc[nthresh,1] = np.percentile(model_early_even[:,nthresh],32)
        model_early_even_mc[nthresh,2] = np.percentile(model_early_even[:,nthresh],68)
        model_early_even_mc[nthresh,3] = np.percentile(model_early_even[:,nthresh],5)
        model_early_even_mc[nthresh,4] = np.percentile(model_early_even[:,nthresh],95)
        
        model_late_even_mc[nthresh,0] = np.percentile(model_late_even[:,nthresh],50)
        model_late_even_mc[nthresh,1] = np.percentile(model_late_even[:,nthresh],32)
        model_late_even_mc[nthresh,2] = np.percentile(model_late_even[:,nthresh],68)
        model_late_even_mc[nthresh,3] = np.percentile(model_late_even[:,nthresh],5)
        model_late_even_mc[nthresh,4] = np.percentile(model_late_even[:,nthresh],95)
        
        model_diffparity_even_mc[nthresh,0] = np.percentile(model_diffparity_even[:,nthresh],50)
        model_diffparity_even_mc[nthresh,1] = np.percentile(model_diffparity_even[:,nthresh],32)
        model_diffparity_even_mc[nthresh,2] = np.percentile(model_diffparity_even[:,nthresh],68)
        model_diffparity_even_mc[nthresh,3] = np.percentile(model_diffparity_even[:,nthresh],5)
        model_diffparity_even_mc[nthresh,4] = np.percentile(model_diffparity_even[:,nthresh],95)
    
    #attribute this output to the specific model
    exec('model_mc_' + model +' = model_mc')
    exec('model_min_mc_' + model +'= model_min_mc')
    exec('model_max_mc_' + model +'= model_max_mc')
    exec('model_diff_mc_' + model +'= model_diff_mc')
    exec('model_early_odd_mc_' + model +'= model_early_odd_mc')
    exec('model_late_odd_mc_' + model +'= model_late_odd_mc')
    exec('model_diffparity_odd_mc_' + model +'= model_diffparity_odd_mc')
    exec('model_early_even_mc_' + model +'= model_early_even_mc')
    exec('model_late_even_mc_' + model +'= model_late_even_mc')
    exec('model_diffparity_even_mc_' + model +'= model_diffparity_even_mc')    
 


# <codecell> plot solar active/quietx storm occurrence

plt.figure()
plt1=plt.subplot(311)

plt.fill_between(storm_thresh_model,model_min_mc_phase[:,3],model_min_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_min_mc_phase[:,1],model_min_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_min_mc_rand[:,3],model_min_mc_rand[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_min_mc_rand[:,1],model_min_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
    plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_model['aaH']-dx,plot_percentile), yval),
                  ha='center')   
    
plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,1],'ko',label='Quiet phase')
plt.plot(storm_thresh_model,model_min_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_min_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_min_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
xx=plt.gca().get_xlim()
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   


    
    

plt1=plt.subplot(312)

plt.fill_between(storm_thresh_model,model_max_mc_phase[:,3],model_max_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_max_mc_phase[:,1],model_max_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_max_mc_rand[:,3],model_max_mc_rand[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_max_mc_rand[:,1],model_max_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,2],'ko',label='Active phase')
plt.plot(storm_thresh_model,model_max_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_max_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_max_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  




plt1=plt.subplot(313)
plt.fill_between(storm_thresh_model,model_diff_mc_phase[:,3],model_diff_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_diff_mc_phase[:,1],model_diff_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_diff_mc_rand[:,3],model_diff_mc_rand[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_thresh_model,model_diff_mc_rand[:,1],model_diff_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')

plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,3],'ko',label='Active - quiet phase')
plt.plot(storm_thresh_model,model_diff_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_diff_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_diff_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
plt.xlabel('$aa_H$ threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  


 
# plt1=plt.subplot(224)
# plt.fill_between(storm_thresh_model,model_phase_ratio_mc[:,3],model_phase_ratio_mc[:,4],
#                  color='pink', alpha=0.5)
# plt.fill_between(storm_thresh_model,model_phase_ratio_mc[:,1],model_phase_ratio_mc[:,2],
#                  color='red', alpha=0.5)
# plt.fill_between(storm_thresh_model,model_rand_ratio_mc[:,3], model_rand_ratio_mc[:,4],
#                  color='blue', alpha=0.3)
# plt.fill_between(storm_thresh_model,model_rand_ratio_mc[:,1],model_rand_ratio_mc[:,2],
#                  color='blue', alpha=0.3)

# plt.plot(storm_prob_minmax[:,0],storm_prob_minmax[:,2]/storm_prob_minmax[:,1],'ko',label='OBservations')
# plt.plot(storm_thresh_model,model_phase_ratio_mc[:,0], 'r', label='Prob: f(snn)')
# plt.plot(storm_thresh_model,model_rand_ratio_mc[:,0], 'b', label='Prob: rand')
# plt.xlabel('$aa_H$ threshold [nT]')
# plt.ylabel('p(Smax)/p(Smin)')
# #plt.yscale('log')
# plt.ylim(-0.5,5.5)
# plt.xlim(20,300)
# plt.legend(frameon=True,framealpha=1,loc='upper right')  

# #plot the percentiles
# yval=5.5
# for plot_percentile in plot_percentiles:
#     Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
#     plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'k--')   
    
    
# <codecell> plot correlation of  cycle amplitude with storm occurrence   
plt.figure()

plt.fill_between(storm_thresh_model,model_mc_phaseamp[:,3],model_mc_phaseamp[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_mc_phaseamp[:,1],model_mc_phaseamp[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_mc_rand[:,3],model_mc_rand[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_mc_rand[:,1],model_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
yval = 1.05
dx=0
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[-.6,yval],'grey')
    plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_model['aaH']-dx,plot_percentile), yval), ha='center') 
    
xx=plt.gca().get_xlim()
yy=plt.gca().get_ylim()
#plt.plot([xx[0]-10, xx[1]+10],[0,0],'k--')
plt.gca().set_xlim([xx[0]-10, xx[1]+10])

plt.plot(storm_thresh_model,correl[:,0], 'ok', label='Observed')
plt.plot(storm_thresh_model,model_mc_phaseamp[:,0], 'r', label='Phase+Amp model')
plt.plot(storm_thresh_model,model_mc_rand[:,0], 'b', label='Random model')
#plt.plot(storm_thresh_model,model_mc_phase[:,0], 'r--', label='Phase model')

plt.xlabel('$aa_H$ threshold [nT]')
plt.ylabel('$r$, correlation of cycle\n amplitude with storm occurrence')
plt.legend(frameon=True,framealpha=1,loc='upper right')
plt.ylim(-0.6,1.05)
plt.xlim(20,300)


# <codecell> plot  early/late odd/even storm occurrence

#observations
storm_prob_earlylate_odd = np.empty((len(storm_thresh),4))
storm_prob_earlylate_even = np.empty((len(storm_thresh),4))
for i in range(0,len(storm_thresh)):

    aaH_1d['storm'] = 0
    mask = aaH_1d['aaH'] > storm_thresh[i]
    aaH_1d.loc[mask,'storm'] = 1

    storm_prob_earlylate_odd[i,0] = storm_thresh[i]
    storm_prob_earlylate_odd[i,1] = aaH_1d.loc[early_mask_odd,'storm'].mean()
    storm_prob_earlylate_odd[i,2] = aaH_1d.loc[late_mask_odd,'storm'].mean()
    storm_prob_earlylate_odd[i,3] = storm_prob_earlylate_odd[i,1] - storm_prob_earlylate_odd[i,2]
    
    storm_prob_earlylate_even[i,0] = storm_thresh[i]
    storm_prob_earlylate_even[i,1] = aaH_1d.loc[early_mask_even,'storm'].mean()
    storm_prob_earlylate_even[i,2] = aaH_1d.loc[late_mask_even,'storm'].mean()
    storm_prob_earlylate_even[i,3] = storm_prob_earlylate_even[i,1] - storm_prob_earlylate_even[i,2]




plt.figure()

#early phase, even cycles
plt1=plt.subplot(321)

plt.fill_between(storm_thresh_model, model_early_even_mc_phaseamp[:,3],model_early_even_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_early_even_mc_phaseamp[:,1],model_early_even_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_early_even_mc_oddeven[:,3],model_early_even_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_early_even_mc_oddeven[:,1],model_early_even_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
    plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_model['aaH']-dx,plot_percentile), yval),
                  ha='center')   
    
plt.plot(storm_prob_earlylate_even[:,0],storm_prob_earlylate_even[:,1],'ko',label='Early Active phase (even cycles)')
#plt.plot(storm_thresh_model,model_early_even_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_early_even_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,model_early_even_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
xx=plt.gca().get_xlim()
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   


    
    
#late phase, even cycles
plt1=plt.subplot(323)

plt.fill_between(storm_thresh_model,model_late_even_mc_phaseamp[:,3],model_late_even_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_late_even_mc_phaseamp[:,1],model_late_even_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_late_even_mc_oddeven[:,3],model_late_even_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_late_even_mc_oddeven[:,1],model_late_even_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
plt.plot(storm_prob_earlylate_even[:,0],storm_prob_earlylate_even[:,2],'ko',label='Late Active phase (even cycles)')
#plt.plot(storm_thresh_model,model_late_even_mc_rand[:,0], 'b', label='Random')
plt.plot(storm_thresh_model,model_late_even_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,model_late_even_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  



#diff, even cycles 
plt1=plt.subplot(325)
plt.fill_between(storm_thresh_model,model_diffparity_even_mc_phaseamp[:,3],model_diffparity_even_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_diffparity_even_mc_phaseamp[:,1],model_diffparity_even_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_diffparity_even_mc_oddeven[:,3],model_diffparity_even_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_diffparity_even_mc_oddeven[:,1],model_diffparity_even_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')

plt.plot(storm_prob_earlylate_even[:,0],storm_prob_earlylate_even[:,3],'ko',label='Early - Late (even cycles)')
#plt.plot(storm_thresh_model,model_diffparity_even_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_diffparity_even_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,model_diffparity_even_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
plt.xlabel('$aa_H$ threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  




#early phase, odd cycles
plt1=plt.subplot(322)

plt.fill_between(storm_thresh_model, model_early_odd_mc_phaseamp[:,3],model_early_odd_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_early_odd_mc_phaseamp[:,1],model_early_odd_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_early_odd_mc_oddeven[:,3],model_early_odd_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_early_odd_mc_oddeven[:,1],model_early_odd_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
yval = 0.3
dx=0
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
    plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
                 xy=(np.percentile(aaH_model['aaH']-dx,plot_percentile), yval),
                  ha='center')   
    
plt.plot(storm_prob_earlylate_odd[:,0],storm_prob_earlylate_odd[:,1],'ko',label='Early Active phase (odd cycles)')
#plt.plot(storm_thresh_model,model_early_odd_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,model_early_odd_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,model_early_odd_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
xx=plt.gca().get_xlim()
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   


    
    
#late phase, odd cycles
plt1=plt.subplot(324)

plt.fill_between(storm_thresh_model,model_late_odd_mc_phaseamp[:,3],model_late_odd_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_late_odd_mc_phaseamp[:,1],model_late_odd_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,model_late_odd_mc_oddeven[:,3],model_late_odd_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,model_late_odd_mc_oddeven[:,1],model_late_odd_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')
plt.plot(storm_prob_earlylate_odd[:,0],storm_prob_earlylate_odd[:,2],'ko',label='Late Active phase (odd cycles)')
#plt.plot(storm_thresh_model,model_late_odd_mc_rand[:,0], 'b', label='Random')
plt.plot(storm_thresh_model,model_late_odd_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,model_late_odd_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  




plt1=plt.subplot(326)
plt.fill_between(storm_thresh_model,-model_diffparity_odd_mc_phaseamp[:,3],-model_diffparity_odd_mc_phaseamp[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,-model_diffparity_odd_mc_phaseamp[:,1],-model_diffparity_odd_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_thresh_model,-model_diffparity_odd_mc_oddeven[:,3],-model_diffparity_odd_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_thresh_model,-model_diffparity_odd_mc_oddeven[:,1],-model_diffparity_odd_mc_oddeven[:,2],
                 color='red', alpha=0.3)
#plot the percentiles
for plot_percentile in plot_percentiles:
    Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
    plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[0,yval],'grey')

plt.plot(storm_prob_earlylate_odd[:,0],-storm_prob_earlylate_odd[:,3],'ko',label='Early - Late (odd cycles)')
#plt.plot(storm_thresh_model,model_diffparity_odd_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_thresh_model,-model_diffparity_odd_mc_oddeven[:,0], 'r', label='EarlyLate model')
plt.plot(storm_thresh_model,-model_diffparity_odd_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
plt.xlabel('$aa_H$ threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt.yscale('log')
plt.ylim(0.00003,0.3)
plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  