# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:05:47 2020

@author: mathewjowens

A script to reproduce the anaylsis of Owens et al., Solar Physics 2021,
"Extreme space-weather events and the solar cycle"

"""
import numpy as np
import pandas as pd
import datetime
import os as os
from scipy.stats import pearsonr
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import cycler

import helio_time as htime


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
nbins=15 #Number of phase bins for SPE [30]
Nmc = 1000 #number of Monte Carlo iterations [10000]
#plot_thresholds = [0,10,100,500]
plot_thresholds = [2,4.3,4.7,5.5]
#nthreshbins = 20 #number of storm threshold bins [20]

active_amp = 4 #probability during active phases relative to break
cycle_amp = 6 
d_earlylate = 0.3 #change in probability in early/late active phase

#demoSCs=[12,19,23] #example solar cycles for projecting SC25

phase_lower = 0.18
phase_upper = 0.79

#set colour scale
nc = 4
color = plt.cm.cool(np.linspace(0, 1,nc))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

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

#Data reader functions
def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.txt'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime.datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime.datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)
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
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*12)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    ssn_df['rollingmax'] = ssn_df.rolling(nwindow, center = True).max()['smooth']
    
    #fill the max value at the end of the series
    fillval = ssn_df['rollingmax'].dropna().values[-1]
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
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
    aaH_df['mjd'] = htime.date2mjd(aaH_df['year'].to_numpy(),
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
    solarmintimes_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
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
    df['mjd'] = df['jd'] - 2400000.5
    # # Add in solar cycle number, phase and rise or fall state
    # number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    # df['sc_num'] = number
    # df['sc_phase'] = phase
    # df['sc_state'] = state
    # df['hale_num'] = hale_num
    # df['hale_phase'] = hale_phase
    return df

def load_gle_intensity_list(filepath = 'null'):
    """
        Function to load in a list of GLEs from Avestari et al., A&A, 2020
        :return: DF - DataFrame containing a list of GLE events, including fields:
           time     : Datetime index of GLE onset
           jd       : Julian date of GLE onset
           intensity: % above background
    """
    if filepath == 'null':
        filepath = os.environ['DBOX'] + 'Data\\GLE-list_with_Intensity.csv'
    df = pd.read_csv(filepath)#, sep=',', names=colnam, header=1, dtype=dtype)
    
    # Convert time and loose useless columns. Calculate julian date of each observation.
    df['mjd'] = htime.date2mjd(df['Year'], df['Month'], df['Day']) + 0.5
    df['datetime'] = htime.mjd2datetime(df['mjd'].values)

    
    
    df.drop(['Date','Day','Month','Year'], axis=1, inplace=True)
  
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

#a function to produce a plottable stairs series from x bin edges and y bin centres            
def GetStairsSeries(xedges, ycentres):
    #xedges = wt_histbins
    #ycentres = model_mc_wt_rand[n-1,3,:]
    pltx = []
    plty = []
    #put the first point in
    pltx.append(xedges[0])
    pltx.append(xedges[0])
    plty.append(0.0)
    plty.append(ycentres[0])
    for i in range(1,len(ycentres)):
        pltx.append(xedges[i])
        pltx.append(xedges[i])
        plty.append(ycentres[i-1])
        plty.append(ycentres[i])
    #put the last point in
    pltx.append(xedges[-1])
    pltx.append(xedges[-1])
    plty.append(ycentres[-1])
    plty.append(ycentres[0])
    
    return pltx, plty
    
# <codecell> read in the data
#=============================================================================

#read inthe GLE list
gle_df = load_gle_intensity_list()

smjd = gle_df['mjd'][0]-365
#smjd = htime.datetime2mjd(datetime(1953,1,1))
fmjd = htime.datetime2mjd(datetime.datetime(2019,11,30))#gle_df['mjd'][len(gle_df)-1] - 1

#read in the aaH data file 
aaH_df = pd.DataFrame() #LoadaaH(aaHfilepath)
aaH_df['mjd'] = np.arange(smjd, fmjd,1 )
aaH_df['datetime'] = htime.mjd2datetime(aaH_df['mjd'].values)


aaH_1d = aaH_df.resample('1D', on='datetime').mean() 
aaH_1d['datetime'] = aaH_1d.index
aaH_1d.reset_index(drop=True, inplace=True)
del aaH_df



#=============================================================================
#read in the solar minimum times
solarmintimes_df = LoadSolarMinTimes()

#=============================================================================
#Read in the sunspot data 
ssn_df = LoadSSN()

#interpolate the SSN to daily values
aaH_1d['ssn'] = np.interp(aaH_1d['mjd'],ssn_df['mjd'],ssn_df['ssn'])
aaH_1d['sai'] = np.interp(aaH_1d['mjd'],ssn_df['mjd'],ssn_df['sai'])





gle_binary = np.zeros((len(aaH_1d),1)) -1
for n in range(0,len(gle_df)):
    if (gle_df['mjd'][n] > smjd) & (gle_df['mjd'][n] <fmjd):
        i = np.argmin(np.abs(aaH_1d['mjd'] - gle_df['mjd'][n]))
        #gle_binary [i] = gle_df['I'][n]
        gle_binary [i] = np.log10(gle_df['F1'][n])
aaH_1d['gle'] = gle_binary

#plt.figure()
#plt.plot(ssn_df['datetime'],ssn_df['ssn'])
#plt.plot(ssn_df['datetime'],ssn_df['smooth'])
         
#gles = load_gle_list()


#add the actual aaH data
aaH_df = LoadaaH(aaHfilepath)

#discard data before and after period of interest
mask = (aaH_df['mjd'] >= smjd) & (aaH_df['mjd'] <= fmjd)
aaH_df = aaH_df[mask]
aaH_df.reset_index(drop=True, inplace=True)

mask = (gle_df['mjd'] >= smjd) & (gle_df['mjd'] <= fmjd)
gle_df = gle_df[mask]
gle_df.reset_index(drop=True, inplace=True)

#create a 1-day running smooth
aaH_smooth = aaH_df.rolling('1D', on='datetime').mean()
#centre the smooth
aaH_smooth['datetime'] = aaH_smooth['datetime'] - datetime.timedelta(days = 0.5)
# plt.figure()
# plt.plot(aaH_df['datetime'],aaH_df['aaH'])
# plt.plot(aaH_smooth['datetime'],aaH_smooth['aaH'])
# plt.xlim((10000,10100))

aaH_df = aaH_df.resample('1D', on='datetime').mean() 
aaH_df['datetime'] = aaH_df.index
aaH_df.reset_index(drop=True, inplace=True)


#compute the solar cycle phase at the daily level
aaH_1d['phase'] = np.nan
for i in range(0, len(solarmintimes_df['mjd'])-1):
    smjd = solarmintimes_df['mjd'][i]
    fmjd = solarmintimes_df['mjd'][i+1]
    
    mask = (aaH_1d['mjd'] >= smjd) & (aaH_1d['mjd'] < fmjd)
    
    thiscyclemjd = aaH_1d['mjd'].loc[mask].to_numpy()
    cyclelength = fmjd - smjd
    
    aaH_1d.loc[mask,'phase'] = (thiscyclemjd - smjd)/cyclelength

# <codecell> compare GLEs with max daily smoothed aaH
dt = 4
gle_props = np.empty((len(gle_df),2))
for n in range(0,len(gle_df)):
    gle_date = gle_df['datetime'][n]
    # mask_smooth = (gle_date >= aaH_smooth['datetime'] - datetime.timedelta(days = dt)) & \
    #     (gle_date <= aaH_smooth['datetime'] + datetime.timedelta(days = dt))
    # gle_props[n,0] = aaH_smooth['aaH'].loc[mask_smooth].max()
    
    mask_1d = (gle_date >= aaH_df['datetime'] - datetime.timedelta(days = dt)) & \
        (gle_date <= aaH_df['datetime'] + datetime.timedelta(days = dt))
    gle_props[n,1] = aaH_df['aaH'].loc[mask_1d].max()


#find the top 68 aaH days
largest_aaH = aaH_df.nlargest(len(gle_df), 'aaH', keep='first')
largest_aaH = largest_aaH.reset_index()
aaH_props = np.ones((len(gle_df),1))
aaH_props_nogle = np.ones((len(gle_df),1))*np.nan
for n in range(0,len(gle_df)):
    aaH_date = largest_aaH['datetime'][n]
    mask = (aaH_date >= gle_df['datetime'] - datetime.timedelta(days = dt)) & \
        (aaH_date <= gle_df['datetime'] + datetime.timedelta(days = dt))
    aaH_props[n,0] = gle_df['F1'].loc[mask].max()
    
    if np.isnan(aaH_props[n,0]):
        aaH_props_nogle[n,0] = 0.01
        
    

# plt.figure()
# plt.plot(gle_props[:,0], gle_props[:,1],'o')

plt.figure(figsize=(10, 6))
ax1 = plt.subplot(121)
plt.plot(gle_df['F1'], gle_props[:,1],'ko')
plt.ylabel('Storm magnitude, $<aa_H>_{1d}$ [nT]')
plt.xlabel('GLE fluence, $F(> 1 GV)$ [cm$^{-2}$]')
plt.title('(a) The ' + str(len(gle_df)) +' GLEs')
plt.xscale('log')
xx=plt.gca().get_xlim()
plt.xlim(xx)

pctls = [90, 95, 99, 99.9]
for n in range(0,len(pctls)):
    aa = np.percentile(aaH_df['aaH'],pctls[n])
    n_aa = sum(aaH_df['aaH']>aa)
    
    plt.plot(xx,[aa,aa] ,'k')
    dy = -2
    if n == 0:
        dy = -9
    plt.text(170000,aa+dy, str(pctls[n]) +' PCTL; N = ' +str(n_aa),
             backgroundcolor='w')



ax2 = plt.subplot(122)
plt.plot(aaH_props[:,0],largest_aaH['aaH'], 'ko', label = 'Storm with GLE')
plt.plot(aaH_props_nogle[:,0],largest_aaH['aaH'], 'ro', label = 'Storm without GLE')
yy=plt.gca().get_ylim()
plt.ylim(yy)
plt.xscale('log')
#plt.plot([0,0],yy,'k')
plt.ylabel('Storm magnitude, $<aa_H>_{1d}$ [nT]')
plt.xlabel('GLE fluence, $F(> 1 GV)$ [cm$^{-2}$]')
plt.legend()
plt.title('(b)Top '+ str(len(gle_df)) + ' $aa_H$ days')


plt.tight_layout()

#compute the associations
n_GLEs_with_storms = sum(gle_props[:,1] >= np.percentile(aaH_df['aaH'],99))
n_storms_with_GLEs = sum(aaH_props[:,0] >= 0)

print(str(n_GLEs_with_storms) + ' of the ' + str(len(gle_df)) +' GLEs have storms (99th percentile of aaH)')
print(str(n_storms_with_GLEs) + ' of the top ' + str(len(gle_df)) + ' storms have GLEs ')

#ax1.set_ylim(yy)
# <codecell> time series processing and plots


plotstart = datetime.datetime(1952,1,1)
plotstop = datetime.datetime(2022,12,31)


              

everyyear = mdates.YearLocator(1)   # every year
every5thyear = mdates.YearLocator(5)   # every 5th year

plt.figure()
ax=plt.subplot(311)        
plt.plot(ssn_df['datetime'],ssn_df['ssn'],'k')
plt.xlim(plotstart, plotstop)
plt.ylabel('Sunspot number')
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
plt.show()
#plot alternate cycles
PlotAlternateCycles(solarmintimes_df)
ax.text(0.03, 0.9, r'(a)', transform = ax.transAxes)


ax=plt.subplot(312)
#compute annual storm occurrence at different thresholds
#compute storm occurrence at different thresholds

n_events = np.ones(len(plot_thresholds), dtype ='int')
n = 0
for thresh in plot_thresholds:

    aaH_1d['event'] = 0
    mask = aaH_1d['gle'] >= thresh
    aaH_1d.loc[mask, 'event'] = 1
    
    n_events[n] = int(sum(aaH_1d['event']))
    
    print('Threshold: ' +str(thresh) + '; N = '+str(np.nansum(aaH_1d.loc[mask, 'event'])))
    
    #take the annual mean
    aaH_1y = aaH_1d.resample('1Y', on='datetime').mean() 
    aaH_1y['datetime'] = aaH_1y.index
    aaH_1y.reset_index(drop=True, inplace=True)
    
    #plot it
    #plt.plot(aaH_1y['datetime'],aaH_1y['storm'],'o-',label= '$aa_H$ > {:0.0f} nT'.format(thresh) )
    #plt.step(aaH_1y['datetime'],aaH_1y['event'],'-',
    #         )
    plt.fill_between(aaH_1y['datetime'],aaH_1y['event'], step="pre",
                     label= 'log(F) > ' +str(thresh) + '; N = ' + str(n_events[n]), linewidth =2 )
    n=n+1
    
plt.ylim(0, 0.025)  
plt.xlim(plotstart, plotstop)  
plt.legend(framealpha=0.7,loc='upper right',ncol=1, bbox_to_anchor=(1.07, 1.07),)
plt.ylabel('GLE occurrence prob [day$^{-1}$]')
#plt.yscale('log')
#plt.gca().set_ylim((0,0.03))
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
PlotAlternateCycles(solarmintimes_df)
ax.text(0.03, 0.9, r'(b)', transform = ax.transAxes)




ax=plt.subplot(313)
n=0
for thresh in plot_thresholds:

    aaH_df['event'] = 0
    thresh = np.percentile(aaH_df['aaH'],100*(1- n_events[n]/len(aaH_df['event'])))
    mask = aaH_df['aaH'] >= thresh
    aaH_df.loc[mask, 'event'] = 1
    
    #n_events[n] = int(sum(aaH_1d['event']))
    
    print('Threshold: ' +str(thresh) + '; N = '+str(np.nansum(aaH_df.loc[mask, 'event'])))
    
    #take the annual mean
    aaH_1y = aaH_df.resample('1Y', on='datetime').mean() 
    aaH_1y['datetime'] = aaH_1y.index
    aaH_1y.reset_index(drop=True, inplace=True)
    
    #plot it
    #plt.plot(aaH_1y['datetime'],aaH_1y['storm'],'o-',label= '$aa_H$ > {:0.0f} nT'.format(thresh) )
    #plt.step(aaH_1y['datetime'],aaH_1y['event'],'-',
    #         )
    plt.fill_between(aaH_1y['datetime'],aaH_1y['event'], step="pre",
                     label= '$aa_H$ > ' +str(int(thresh)) +  ' nT; N = ' + str(n_events[n]), linewidth =2 )


#plt.plot(x,y, drawstyle="steps")
    n=n+1
    
plt.ylim(0, 0.025)    
plt.xlim(plotstart, plotstop)  
plt.legend(framealpha=0.7,loc='upper right',ncol=2, bbox_to_anchor=(1.07, 1.07))
plt.ylabel('Storm occurrence prob [day$^{-1}$]')
#plt.yscale('log')
#plt.gca().set_ylim((0,0.03))
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
plt.xlabel('Year')
PlotAlternateCycles(solarmintimes_df)
ax.text(0.03, 0.9, r'(c)', transform = ax.transAxes, )
 

# <codecell> Super-posed epoch plots


bins=np.arange(0,1.00001,1/(nbins)) 
dbin = bins[1]-bins[0]
bin_centres = np.arange(dbin/2, 1-dbin/2 +0.01 ,dbin)

epochs_start = solarmintimes_df['mjd'][:-1].to_numpy()       
epochs_stop = solarmintimes_df['mjd'][1:].to_numpy()        
            
times = aaH_1d['mjd'].to_numpy()  
data = aaH_1d['ssn'].to_numpy()  
spe_ssn = SPE_dualepoch(times, data, epochs_start, epochs_stop, bins)         
      

      


#plt1=plt.subplot(212)
#plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3])

#find the cut off solar cycle phase. Use FWHM
# ssn_thresh = spe_ssn[:,1].max()/2 
# mask_smin = spe_ssn[:,1] < ssn_thresh
# mask_smax = spe_ssn[:,1] >= ssn_thresh

#record teh phase boundaries for solar min/max
# phase_lower = bin_centres[mask_smax].min()
# phase_upper = bin_centres[mask_smax].max()


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
#plt.xlabel('Solar cycle phase')
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

for nsubplot in range(0,3):
    if nsubplot == 0:
        plt1=plt.subplot(222)
        aaHdata = aaH_1d.copy()
        title = '(b) GLEs: All data'
        tcol = 'k'
        ax=plt1
        plt.gca().set_ylim((0,0.01))
    elif nsubplot == 1:
        plt2=plt.subplot(223)
        aaHdata = aaH_1d[aaH_1d['parity'] == 1].copy()
        title = '(c) GLEs: Even cycles'
        tcol = 'r'
        ax=plt2
        plt.xlabel('Solar cycle phase')
        plt.gca().set_ylim((0,0.015))
    elif nsubplot == 2:
        plt3=plt.subplot(224)
        ax=plt3
        aaHdata = aaH_1d[aaH_1d['parity'] == -1].copy()
        title = '(d) GLEs: Odd cycles'
        tcol = 'b'
        plt.xlabel('Solar cycle phase')
        plt.gca().set_ylim((0,0.015))
        
    plt.gca().set_ylim((0,0.015))
    yy=plt.gca().get_ylim()
    #plt.plot([phase_lower,phase_lower],yy,'k--')
    #plt.plot([phase_upper,phase_upper],yy,'k--')
    plt.ylabel('Probability [day$^{-1}$]')
    plt.title(title, color=tcol)
    
    
    
    yy=plt.gca().get_ylim()
    rect = patches.Rectangle((phase_lower, yy[0]), phase_upper-phase_lower, yy[1]-yy[0],
                              edgecolor='none',facecolor='lightgrey', zorder=-1)
    ax.add_patch(rect)
    rect = patches.Rectangle((phase_mid, yy[0]), phase_upper-phase_mid, yy[1]-yy[0],
                              edgecolor='none',facecolor='darkgrey', zorder=-1)
    ax.add_patch(rect)
     
    for i in range(0,len(plot_thresholds)):
        thresh = plot_thresholds[i]
    
        aaHdata.loc[:,'storm'] = 0
        mask = aaHdata['gle'] >= thresh
        aaHdata.loc[mask,'storm'] = 1
        n_events = sum(aaHdata['storm'])
        
        times = aaHdata['mjd'].to_numpy()
        data = aaHdata['storm'].to_numpy()
        print(title, str(np.sum(data==1)))
        
        spe_aaH = SPE_dualepoch(times, data, epochs_start, epochs_stop, bins) 

        #plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
        #              label= '$aa_H$ > {:0.0f} nT'.format(thresh))
        #plt.errorbar(spe_aaH[:,0],spe_aaH[:,1], yerr = spe_aaH[:,3], fmt = 'o-',
        #              label= 'I > ' + str(thresh) + '%')  
        plt.fill_between(spe_aaH[:,0],spe_aaH[:,1], step="pre",
                         label= 'log(F) > ' +str(thresh) + '; N = '
                         + str(n_events), linewidth =2 )
     


        
    
    #plt.yscale('log')
    
   
    
plt1.legend(framealpha=1,loc='upper right',ncol=1)
plt1.yaxis.set_ticks_position("right")
plt3.yaxis.set_ticks_position("right")
plt1.yaxis.set_label_position("right")
plt3.yaxis.set_label_position("right")


# <codecell> Waiting time between GLEs
#from scipy.stats import gaussian_kde


#figure out the histbins to use
wt_histbins = np.array([-0.2767, 0.1002, 0.47712125, 0.854607  , 1.23209275, 1.60957849, 1.98706424,
       2.36454998, 2.74203573, 3.11952147, 3.49700722, 3.87449296,
       4.25197871])



fig, axs = plt.subplots(nrows=2, ncols=2)

n=1
labels = ['(a) ' , '(b) ', '(c) ', '(d) ']
for thresh,ax,lab in zip(plot_thresholds, axs.ravel(), labels):
    #find the events
    mask =  np.log10(gle_df['F1']) >= thresh
    events = gle_df[mask]
    events = events.reset_index()
    #compute the wiating times
    nGLEs = len(events)
    waiting_time = np.empty((nGLEs-1))
    for i in range(0,nGLEs-1):
        waiting_time[i] = np.log10((events['mjd'][i+1] - events['mjd'][i]))

    ax.hist(waiting_time, bins = wt_histbins, histtype = 'step', density = False,
              label = lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs),
              linewidth = 2, color='k')
    #n, bin_edges = np.histogram(waiting_time)
    #bin_centres = bin_edges[:-1] + np.diff(bin_edges)/2
    #plt.step(bin_centres, n/sum(n),  label = 'I > ' + str(thresh) + '%')
    #ax.set_title('I > ' + str(thresh) + '%')

    
    
    # density = gaussian_kde(waiting_time)
    # xs = np.linspace(0,waiting_time.max()*1.2,250)
    # density.covariance_factor = lambda : .10
    # density._compute_covariance()
    # ax.plot(xs,density(xs),label = 'I > ' + str(thresh) + '%',
    # linewidth = 2)
    
    if n==1 or n==3:
        ax.set_ylabel('Occurrence frequency')
    if n == 3 or n==4:
        ax.set_xlabel('Waiting time [days]')
        
    #ax.legend(loc ='upper left')
    
    ax.text(0.03, 0.93, lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs), 
            transform = ax.transAxes, backgroundcolor = 'w')
    ax.set_xlim([-0.3,4.1])
    #ax.set_xticks([np.log10(1), np.log10(30), np.log10(365), np.log10(11*365)])
    ax.set_xticks([0,1,2,3,4])
    #ax.set_xticklabels(['1 day','1 month', '1 year', '11 years'])
    ax.set_xticklabels(['$10^0$','$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
    yy = ax.get_ylim()
    y=[yy[0],yy[1]*1.1]
    ax.set_ylim(y)
    
    dx = 0.20
    
    ax.plot([np.log10(1), np.log10(1)], y, 'r--')
    ax.text(np.log10(1)+dx, y[1]/20, '1 day', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(27), np.log10(27)], y, 'r--')
    ax.text(np.log10(27)+dx, y[1]/20, '27 days', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(365), np.log10(365)], y, 'r--')
    ax.text(np.log10(365)+dx, y[1]/20, '1 yr', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(11*365), np.log10(11*365)], y, 'r--')
    ax.text(np.log10(11*365)+dx, y[1]/20, '11 yrs', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    n=n+1
#plt.legend() 


# <codecell> Compute dt from all events


dt_histbins = wt_histbins

fig, axs = plt.subplots(nrows=2, ncols=2)

n=1
labels = ['(a) ' , '(b) ', '(c) ', '(d) ']

for thresh,ax,lab in zip(plot_thresholds, axs.ravel(), labels):
    mask =  np.log10(gle_df['F1']) >= thresh
    events = gle_df[mask]
    events = events.reset_index()
    #compute the wiating times
    nGLEs = len(events)
    
    #loop through each GLE and compute the dt
    dt_GLE = []
    for i in range(0,nGLEs-1):
        for k  in range(i+1,nGLEs):
            dt_GLE.append(abs(events['mjd'][i] - events['mjd'][k]))
            
    ax.hist(np.log10(dt_GLE), histtype = 'step', bins = dt_histbins,
              density = False,
              label = lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs),
              linewidth = 2, color='k')
    #n, bin_edges = np.histogram(waiting_time)
    #bin_centres = bin_edges[:-1] + np.diff(bin_edges)/2
    #plt.step(bin_centres, n/sum(n),  label = 'I > ' + str(thresh) + '%')
    #ax.set_title('I > ' + str(thresh) + '%')

    
    
    # density = gaussian_kde(waiting_time)
    # xs = np.linspace(0,waiting_time.max()*1.2,250)
    # density.covariance_factor = lambda : .10
    # density._compute_covariance()
    # ax.plot(xs,density(xs),label = 'I > ' + str(thresh) + '%',
    # linewidth = 2)
    
    if n==1 or n==3:
        ax.set_ylabel('Occurrence frequency')
    if n == 3 or n==4:
        ax.set_xlabel(r'$\Delta t$ [days]')
        
    #ax.legend(loc ='upper left')
    
    ax.text(0.03, 0.93, lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs), 
            transform = ax.transAxes, backgroundcolor = 'w')
    ax.set_xlim([-0.1,4.9])
    ax.set_xticks([0,1,2,3,4])
    #ax.set_xticklabels(['1 day','1 month', '1 year', '11 years'])
    ax.set_xticklabels(['$10^0$','$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
    yy = ax.get_ylim()
    y=[yy[0],yy[1]*1.1]
    ax.set_ylim(y)
    
    dx = 0.21
    
    ax.plot([np.log10(1), np.log10(1)], y, 'r--')
    ax.text(np.log10(1)+dx, 10*y[1]/20, '1 day', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(27), np.log10(27)], y, 'r--')
    ax.text(np.log10(27)+dx, 10*y[1]/20, '27 days', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(365), np.log10(365)], y, 'r--')
    ax.text(np.log10(365)+dx, 10*y[1]/20, '1 yr', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(11*365), np.log10(11*365)], y, 'r--')
    ax.text(np.log10(11*365)+dx, y[1]/20, '11 yrs', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    n=n+1
        
    



# <codecell> effect of solar cycle amplitude


storm_thresh = plot_thresholds
meanssn=np.nanmean(aaH_1d['ssn'])

cycle_total = np.empty((len(solarmintimes_df['mjd'])-1, len(storm_thresh)+2))
correl = np.empty((len(storm_thresh),2))
for i in range(0, len(storm_thresh)):
    
    #define storms for this threshold
    aaH_1d['storm'] = 0
    mask = aaH_1d['gle'] >= storm_thresh[i]
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
cycle_total = np.empty((len(solarmintimes_df['mjd'])-1, len(plot_thresholds)+2))
for i in range(0,len(plot_thresholds)):
    thresh = plot_thresholds[i]
    Nstorms =  sum(aaH_1d['gle'] >= thresh)  
    #define storms for this threshold
    aaH_1d['storm'] = 0
    mask = aaH_1d['gle'] > thresh
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
    plt.annotate('I > ' + str(thresh) +' \nN=' 
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

min_mask = ( (aaH_1d['phase'] < phase_lower) | (aaH_1d['phase'] > phase_upper))
max_mask = ( (aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] <= phase_upper))

early_mask = ( (aaH_1d['phase'] < phase_lower) | (aaH_1d['phase'] > phase_mid))
late_mask = ( (aaH_1d['phase'] >= phase_mid) & (aaH_1d['phase'] <= phase_upper))

#assume extreme events perfectly follow the solar cycle


aaH_model = aaH_1d.copy()

#generate the relative probability time series

#random
aaH_model['rel_prob_rand'] = 1

#function of phase only
aaH_model['rel_prob_phase'] = 1
aaH_model.loc[max_mask,'rel_prob_phase'] = active_amp

#phase and amplitude
aaH_model['rel_prob_phaseamp'] = 1 


#phase and amplitude
aaH_model['rel_prob_phaseamp'] = 1 
for n in range(0,len(solarmintimes_df['mjd'])-1 ):
    amp = solarmintimes_df['amplitude'][n]
        
    mask = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
              (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
               ((aaH_1d['phase'] >= phase_lower) & (aaH_1d['phase'] <= phase_upper))
    #mask = (aaH_model['mjd'] >= solarmintimes_df['mjd'][n]) & \
    #         (aaH_model['mjd'] < solarmintimes_df['mjd'][n+1]) & \
    #          ((aaH_1d['sai'] >= 0.5))
    aaH_model.loc[mask,'rel_prob_phaseamp'] =  amp * active_amp/meanssn
    #aaH_model.loc[mask,'rel_prob_phaseamp'] =  1 + active_amp * cycle_amp * (amp/meanssn)
    
   

#odd/even difference
aaH_model['rel_prob_oddeven'] = aaH_model['rel_prob_phaseamp']

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
        
        
        

model_name_list = ['rand','phase','phaseamp', 'oddeven']
#NOrmalise the probability and create CDF for plotting only
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
plt.fill_between(aaH_model['datetime'],aaH_model['rel_prob_phase']*0,aaH_model['rel_prob_phase']*1000,
                 color='red',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_rand']*1000,'b',label='Random model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_phaseamp']*1000,'k',label='Phase+Amp model')
#plt.plot(aaH_model['datetime'],aaH_model['rel_prob_phase'],'r--',label='Phase model')
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_oddeven']*1000,'k--',label='OddEven model')
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
plt.plot(aaH_model['datetime'],aaH_model['rel_prob_cdf_oddeven'],'k--',label='OddEven model')
plt.ylim(0,1)
plt.ylabel('Cumulative probability')
plt.xlabel('Year')
plt.legend()
ax.xaxis.set_minor_locator(every5thyear)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(which='major', length=10, color='k')
PlotAlternateCycles(solarmintimes_df)
plt.xlim(aaH_1d['datetime'][0], aaH_1d['datetime'][len(aaH_1d)-1])




# <codecell> Monte Carlo models
early_mask_odd = ( (aaH_model['phase'] > phase_lower) & (aaH_model['phase'] < phase_mid)) \
                & (aaH_model['parity'] == -1)
late_mask_odd = ( (aaH_model['phase'] >= phase_mid) & (aaH_model['phase'] <= phase_upper)) \
                & (aaH_model['parity'] == -1)
early_mask_even = ( (aaH_model['phase'] > phase_lower) & (aaH_model['phase'] < phase_mid)) \
                & (aaH_model['parity'] == 1)
late_mask_even = ( (aaH_model['phase'] >= phase_mid) & (aaH_model['phase'] <= phase_upper)) \
                & (aaH_model['parity'] == 1)
                
mask_activeparity = early_mask_even | late_mask_odd
mask_quietparity = late_mask_even | early_mask_odd

storm_thresh_model = storm_thresh

for model in model_name_list:


    #model where storm occurrence is function of phase only
    model_r = np.empty((Nmc,len(storm_thresh_model)))
    model_min = np.empty((Nmc,len(storm_thresh_model)))
    model_max = np.empty((Nmc,len(storm_thresh_model)))
    model_diff = np.empty((Nmc,len(storm_thresh_model)))
    #model_ratio = np.empty((Nmc,len(storm_thresh_model)))
    
    model_early_odd = np.empty((Nmc,len(storm_thresh_model)))
    model_late_odd = np.empty((Nmc,len(storm_thresh_model)))
    model_diffparity_odd = np.empty((Nmc,len(storm_thresh_model)))
    
    model_early_even = np.empty((Nmc,len(storm_thresh_model)))
    model_late_even = np.empty((Nmc,len(storm_thresh_model)))
    model_diffparity_even = np.empty((Nmc,len(storm_thresh_model)))
    
    #consider (early-odd and late-even) and (late-odd and early-even) together
    model_parity_min = np.empty((Nmc,len(storm_thresh_model)))
    model_parity_max = np.empty((Nmc,len(storm_thresh_model)))
    model_parity_diff = np.empty((Nmc,len(storm_thresh_model)))
    #model_parity_ratio = np.empty((Nmc,len(storm_thresh_model)))
    
    for nthresh in range(0,len(storm_thresh_model)):
        print('Model:' + model + '; MC simulation of threshold ' +str(nthresh+1) 
              + ' of ' +str(len(storm_thresh_model)) )
        Nstorms =  sum(aaH_model['gle'] >= storm_thresh_model[nthresh]) 
        
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
           
            model_parity_min[Ncount,nthresh] = aaH_model.loc[mask_quietparity,'storm'].mean()
            model_parity_max[Ncount,nthresh] = aaH_model.loc[mask_activeparity,'storm'].mean()
            model_parity_diff[Ncount,nthresh] = model_parity_max[Ncount,nthresh] - model_parity_min[Ncount,nthresh]
            
            
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
    model_parity_min_mc = np.empty((len(storm_thresh_model),5))
    model_parity_max_mc = np.empty((len(storm_thresh_model),5))
    model_parity_diff_mc = np.empty((len(storm_thresh_model),5))
    
    for nthresh in range(0,len(storm_thresh_model)):
        model_mc[nthresh,0] = np.percentile(model_r[:,nthresh],50)
        model_mc[nthresh,1] = np.percentile(model_r[:,nthresh],32)
        model_mc[nthresh,2] = np.percentile(model_r[:,nthresh],68)
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
        
        model_parity_min_mc[nthresh,0] = np.percentile(model_parity_min[:,nthresh],50)
        model_parity_min_mc[nthresh,1] = np.percentile(model_parity_min[:,nthresh],32)
        model_parity_min_mc[nthresh,2] = np.percentile(model_parity_min[:,nthresh],68)
        model_parity_min_mc[nthresh,3] = np.percentile(model_parity_min[:,nthresh],5)
        model_parity_min_mc[nthresh,4] = np.percentile(model_parity_min[:,nthresh],95)
        
        model_parity_max_mc[nthresh,0] = np.percentile(model_parity_max[:,nthresh],50)
        model_parity_max_mc[nthresh,1] = np.percentile(model_parity_max[:,nthresh],32)
        model_parity_max_mc[nthresh,2] = np.percentile(model_parity_max[:,nthresh],68)
        model_parity_max_mc[nthresh,3] = np.percentile(model_parity_max[:,nthresh],5)
        model_parity_max_mc[nthresh,4] = np.percentile(model_parity_max[:,nthresh],95)
        
        model_parity_diff_mc[nthresh,0] = np.percentile(model_parity_diff[:,nthresh],50)
        model_parity_diff_mc[nthresh,1] = np.percentile(model_parity_diff[:,nthresh],32)
        model_parity_diff_mc[nthresh,2] = np.percentile(model_parity_diff[:,nthresh],68)
        model_parity_diff_mc[nthresh,3] = np.percentile(model_parity_diff[:,nthresh],5)
        model_parity_diff_mc[nthresh,4] = np.percentile(model_parity_diff[:,nthresh],95)
    
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
    exec('model_parity_min_mc_' + model +'= model_parity_min_mc') 
    exec('model_parity_max_mc_' + model +'= model_parity_max_mc') 
    exec('model_parity_diff_mc_' + model +'= model_parity_diff_mc') 
 



# <codecell> plot solar active/quietx storm occurrence

plt.figure(figsize = (5,10))
plt1=plt.subplot(311)

storm_x = np.arange(1,5,1)#storm_thresh
storm_x_labs = ['>2','>4.3','>4.7','>5.5']


storm_prob_minmax = np.empty((len(storm_thresh),4))
for i in range(0,len(storm_thresh)):

    aaH_1d['storm'] = 0
    mask = aaH_model['gle']  > storm_thresh[i]
    aaH_1d.loc[mask,'storm'] = 1

    storm_prob_minmax[i,0] = storm_thresh[i]
    storm_prob_minmax[i,1] = aaH_1d.loc[min_mask,'storm'].mean()
    storm_prob_minmax[i,2] = aaH_1d.loc[max_mask,'storm'].mean()
    storm_prob_minmax[i,3] = storm_prob_minmax[i,2] - storm_prob_minmax[i,1]
    
    Nstorms =  sum(aaH_1d['gle'] >= storm_thresh[i]) 
    
    print(storm_thresh[i], Nstorms, storm_prob_minmax[i,2]/storm_prob_minmax[i,1])

plt.fill_between(storm_x, model_min_mc_phase[:,3],model_min_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x, model_min_mc_phase[:,1],model_min_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_x ,model_min_mc_rand[:,3],model_min_mc_rand[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_x, model_min_mc_rand[:,1],model_min_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
yval = 0.3
dx=0
# for thresh in plot_thresholds:
#     Nstorms =  sum(aaH_model['gle'] >= thresh)
    
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')
#     plt.annotate('I > '+ str(thresh) + '\nN=' + str(Nstorms) +'\n',
#                  xy=(thresh, yval),
#                   ha='center')   
    
plt.plot(storm_x, storm_prob_minmax[:,1],'ko',label='Quiet phase')
plt.plot(storm_x, model_min_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_x, model_min_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_min_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
#plt.yscale('log')
xx=plt.gca().get_xlim()
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   
plt1.text(0.01, 0.05, '(a)', transform = plt1.transAxes, )
    
    

plt1=plt.subplot(312)

plt.fill_between(storm_x,model_max_mc_phase[:,3],model_max_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_max_mc_phase[:,1],model_max_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_max_mc_rand[:,3],model_max_mc_rand[:,4],
                 color='blue', alpha=0.5)
plt.fill_between(storm_x,model_max_mc_rand[:,1],model_max_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
# for thresh in plot_thresholds:  
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')
plt.plot(storm_x,storm_prob_minmax[:,2],'ko',label='Active phase')
plt.plot(storm_x,model_max_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_x,model_max_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_max_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
#plt.yscale('log')
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  
plt1.text(0.01, 0.05, '(b)', transform = plt1.transAxes, )




plt1=plt.subplot(313)
plt.fill_between(storm_x,model_diff_mc_phase[:,3],model_diff_mc_phase[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_diff_mc_phase[:,1],model_diff_mc_phase[:,2],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_diff_mc_rand[:,3],model_diff_mc_rand[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_x,model_diff_mc_rand[:,1],model_diff_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
# for thresh in plot_thresholds:  
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')

plt.plot(storm_x,storm_prob_minmax[:,3],'ko',label='Active - quiet phase')
plt.plot(storm_x,model_diff_mc_rand[:,0], 'b', label='Random model')
plt.plot(storm_x,model_diff_mc_phase[:,0], 'r', label='Phase model')
#plt.plot(storm_thresh_model,model_diff_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
plt.xlabel('GLE intensity, log(F)')
plt.ylabel('Probability [day$^{-1}$]')
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
#plt.yscale('log')
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  
plt1.text(0.01, 0.05, '(c)', transform = plt1.transAxes, )
    
plt.tight_layout()    
# <codecell> plot correlation of  cycle amplitude with storm occurrence   
plt.figure(figsize = (6,4))


plt1=plt.subplot(111)

plt.fill_between(storm_x,model_mc_phaseamp[:,3],model_mc_phaseamp[:,4],
                 color='red', alpha=0.5)
plt.fill_between(storm_x,model_mc_phaseamp[:,1],model_mc_phaseamp[:,2],
                 color='red', alpha=0.5)
plt.fill_between(storm_x,model_mc_phase[:,3],model_mc_rand[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_x,model_mc_phase[:,1],model_mc_rand[:,2],
                 color='blue', alpha=0.5)
#plot the percentiles
# yval = 1.05
# dx=0
# for plot_percentile in plot_percentiles:
#     Nstorms =  sum(aaH_model['aaH'] >= np.percentile(aaH_model['aaH'],plot_percentile))
    
#     plt.plot(np.percentile(aaH_model['aaH'],plot_percentile)*np.ones(2),[-.6,yval],'grey')
#     plt.annotate(str(plot_percentile) + 'th \nN=' + str(Nstorms) +'\n',
#                  xy=(np.percentile(aaH_model['aaH']-dx,plot_percentile), yval), ha='center') 
    
xx=plt.gca().get_xlim()
yy=plt.gca().get_ylim()
#plt.plot([xx[0]-10, xx[1]+10],[0,0],'k--')
#plt.gca().set_xlim([xx[0]-10, xx[1]+10])

plt.plot(storm_x,correl[:,0], 'ok', label='Observed')
plt.plot(storm_x,model_mc_phaseamp[:,0], 'r', label='Phase+Amp model')
plt.plot(storm_x,model_mc_phase[:,0], 'b', label='Phase model')
#plt.plot(storm_thresh_model,model_mc_phase[:,0], 'r--', label='Phase model')

plt.xlabel('GLE intensity, log(F)')
plt.ylabel('$r$, correlation of cycle\n amplitude with GLE occurrence')
plt.legend(frameon=True,framealpha=1,loc='lower right')
plt.ylim(-0.6,1.05)
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
#plt.xlim(20,300)

plt.tight_layout()    




# <codecell> plot active parity and quiet parity

storm_prob_parity = np.empty((len(storm_thresh),4))
for i in range(0,len(storm_thresh)):

    aaH_1d['storm'] = 0
    mask = aaH_1d['gle'] >= storm_thresh[i]
    aaH_1d.loc[mask,'storm'] = 1

    storm_prob_parity[i,0] = storm_thresh[i]
    storm_prob_parity[i,1] = aaH_1d.loc[mask_quietparity,'storm'].mean()
    storm_prob_parity[i,2] = aaH_1d.loc[mask_activeparity,'storm'].mean()
    storm_prob_parity[i,3] = storm_prob_parity[i,2] - storm_prob_parity[i,1]
    
    Nstorms =  sum(aaH_1d['gle'] >= storm_thresh[i]) 

plt.figure(figsize = (5,10))

plt1=plt.subplot(311)

plt.fill_between(storm_x,model_parity_min_mc_phaseamp[:,3],model_parity_min_mc_phaseamp[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_x,model_parity_min_mc_phaseamp[:,1],model_parity_min_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_x,model_parity_min_mc_oddeven[:,3],model_parity_min_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_parity_min_mc_oddeven[:,1],model_parity_min_mc_oddeven[:,2],
                 color='red', alpha=0.5)
#plot the percentiles
yval = 0.3
dx=0
# for thresh in plot_thresholds:
#     Nstorms =  sum(aaH_model['gle'] >= thresh)
    
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')
#     plt.annotate('I > '+ str(thresh) + '\nN=' + str(Nstorms) +'\n',
#                  xy=(thresh, yval),
#                   ha='center')   
    
plt.plot(storm_x,storm_prob_parity[:,1],'ko',label='Quiet parity phase')
plt.plot(storm_x,model_parity_min_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
plt.plot(storm_x,model_parity_min_mc_oddeven[:,0], 'r', label='OddEven model')
#plt.plot(storm_thresh_model,model_min_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
#plt.yscale('log')
xx=plt.gca().get_xlim()
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
plt1.text(0.01, 0.05, '(a)', transform = plt1.transAxes, )

    
    
plt1=plt.subplot(312)
plt.fill_between(storm_x,model_parity_max_mc_phaseamp[:,3],model_parity_max_mc_phaseamp[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_x,model_parity_max_mc_phaseamp[:,1],model_parity_max_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_x,model_parity_max_mc_oddeven[:,3],model_parity_max_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_parity_max_mc_oddeven[:,1],model_parity_max_mc_oddeven[:,2],
                 color='red', alpha=0.5)
#plot the percentiles
yval = 0.3
dx=0
# for thresh in plot_thresholds:
#     Nstorms =  sum(aaH_model['gle'] >= thresh)
    
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')
#     plt.annotate('I > '+ str(thresh) + '\nN=' + str(Nstorms) +'\n',
#                  xy=(thresh, yval),
#                   ha='center')   
    
plt.plot(storm_x,storm_prob_parity[:,2],'ko',label='Active parity phase')
plt.plot(storm_x,model_parity_max_mc_phaseamp[:,0], 'b', label='Phase+Amp model')
plt.plot(storm_x,model_parity_max_mc_oddeven[:,0], 'r', label='OddEven model')
#plt.plot(storm_thresh_model,model_min_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
#plt.xlabel('Storm threshold [nT]')
plt.ylabel('Probability [day$^{-1}$]')
#plt.yscale('log')
xx=plt.gca().get_xlim()
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')   
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
plt1.text(0.01, 0.05, '(b)', transform = plt1.transAxes, )


plt1=plt.subplot(313)
plt.fill_between(storm_x,model_parity_diff_mc_phaseamp[:,3],model_parity_diff_mc_phaseamp[:,4],
                 color='blue', alpha=0.3)
plt.fill_between(storm_x,model_parity_diff_mc_phaseamp[:,1],model_parity_diff_mc_phaseamp[:,2],
                 color='blue', alpha=0.5)
plt.fill_between(storm_x,model_parity_diff_mc_oddeven[:,3],model_parity_diff_mc_oddeven[:,4],
                 color='red', alpha=0.3)
plt.fill_between(storm_x,model_parity_diff_mc_oddeven[:,1],model_parity_diff_mc_oddeven[:,2],
                 color='red', alpha=0.5)
#plot the percentiles
# for thresh in plot_thresholds:  
#     plt.plot(thresh*np.ones(2),[0,yval],'grey')

plt.plot(storm_x,storm_prob_parity[:,3],'ko',label='Active - quiet parity phase')
plt.plot(storm_x,model_parity_diff_mc_phaseamp[:,0], 'b', label='Phase+amp model')
plt.plot(storm_x,model_parity_diff_mc_oddeven[:,0], 'r', label='OddEven model')
#plt.plot(storm_thresh_model,model_diff_mc_phaseamp[:,0], 'k--', label='Phase+Amp model')
plt.xlabel('GLE intensity, log(F)')
plt.ylabel('Probability [day$^{-1}$]')
#plt.yscale('log')
#plt.ylim(0.00003,0.3)
#plt.xlim(20,300)
plt.legend(frameon=True,framealpha=1,loc='upper right')  
plt1.set_xticks(storm_x)
plt1.set_xticklabels(storm_x_labs)
plt1.text(0.01, 0.05, '(c)', transform = plt1.transAxes, )


plt.tight_layout()  

# <codecell> Monte Carlo test waiting time significance



Nbin = len(wt_histbins)
bin_centres =( wt_histbins[1:] + wt_histbins[:-1])/2

storm_thresh = plot_thresholds
storm_thresh_model = storm_thresh

model_name_list = ['rand']


for model in model_name_list:


    #model where storm occurrence is function of phase only
    model_wt = np.empty((Nmc,len(storm_thresh_model),Nbin-1))
    model_dt = np.empty((Nmc,len(storm_thresh_model),Nbin-1))    
    
    for nthresh in range(0,len(storm_thresh_model)):
        print('Model:' + model + '; MC simulation of threshold ' +str(nthresh+1) 
              + ' of ' +str(len(storm_thresh_model)) )
        Nstorms =  sum(aaH_model['gle'] >= storm_thresh_model[nthresh]) 
        
        for Ncount in range(0,Nmc):
            #generate the random storm time series
            aaH_model['storm'] = generate_events(aaH_model['rel_prob_'+model].to_numpy(), Nstorms)
            
            #get the events themselves
            mask =  aaH_model['storm'] >= 0.5
            events = aaH_1d[mask]
            events = events.reset_index()
            
            #compute the waiting times
            waiting_time = np.empty((Nstorms-1))
            for i in range(0,Nstorms-1):
                waiting_time[i] = np.log10((events['mjd'][i+1] - events['mjd'][i]))

            counts, bins, patches = ax.hist(waiting_time, bins = wt_histbins, 
                                       histtype = 'step', density = False)
            model_wt[Ncount,nthresh,:] = counts
            
            #loop through each GLE and compute the dt
            dt_GLE = []
            for i in range(0,Nstorms-1):
                for k  in range(i+1,Nstorms):
                    dt_GLE.append(np.log10(abs(events['mjd'][i] - events['mjd'][k])))
            counts, bins, patches = ax.hist(dt_GLE, bins = wt_histbins, 
                                       histtype = 'step', density = False)
            model_dt[Ncount,nthresh,:] = counts
          
           
        
    #find the median and 1- and 2- sigma percentiles of the Monte Carlo runs
    model_mc_wt = np.empty((len(storm_thresh_model),5, Nbin-1))
    model_mc_dt = np.empty((len(storm_thresh_model),5, Nbin-1))
    
    
    for nthresh in range(0,len(storm_thresh_model)):
        for n in range(0,Nbin-1):
            model_mc_wt[nthresh,0,n] = np.percentile(model_wt[:,nthresh,n],50)
            model_mc_wt[nthresh,1,n] = np.percentile(model_wt[:,nthresh,n],32)
            model_mc_wt[nthresh,2,n] = np.percentile(model_wt[:,nthresh,n],68)
            model_mc_wt[nthresh,3,n] = np.percentile(model_wt[:,nthresh,n],5)
            model_mc_wt[nthresh,4,n] = np.percentile(model_wt[:,nthresh,n],95)
            
            model_mc_dt[nthresh,0,n] = np.percentile(model_dt[:,nthresh,n],50)
            model_mc_dt[nthresh,1,n] = np.percentile(model_dt[:,nthresh,n],32)
            model_mc_dt[nthresh,2,n] = np.percentile(model_dt[:,nthresh,n],68)
            model_mc_dt[nthresh,3,n] = np.percentile(model_dt[:,nthresh,n],5)
            model_mc_dt[nthresh,4,n] = np.percentile(model_dt[:,nthresh,n],95)
        
       
    
    #attribute this output to the specific model
    exec('model_mc_wt_' + model +' = model_mc_wt')
    exec('model_mc_dt_' + model +' = model_mc_dt')
    
# <codecell> plot Monte Carlo waiting times

#plot wt
fig, axs = plt.subplots(nrows=2, ncols=2)

n=1
labels = ['(a) ' , '(b) ', '(c) ', '(d) ']
for thresh,ax,lab in zip(plot_thresholds, axs.ravel(), labels):
    
    #construct the "stair" series 
    xx, yylower = GetStairsSeries(wt_histbins, model_mc_wt_rand[n-1,3,:])
    xx, yyupper = GetStairsSeries(wt_histbins, model_mc_wt_rand[n-1,4,:])
    #add the MC reaults
    ax.fill_between(xx,yylower,yyupper,
                     color='red', alpha=0.3)
    xx, yylower = GetStairsSeries(wt_histbins, model_mc_wt_rand[n-1,1,:])
    xx, yyupper = GetStairsSeries(wt_histbins, model_mc_wt_rand[n-1,2,:])
    ax.fill_between(xx,yylower,yyupper,
                     color='red', alpha=0.3)
    xx, yy = GetStairsSeries(wt_histbins, model_mc_wt_rand[n-1,0,:])
    ax.plot(xx,yy,  color='red', label='Random model')
    
    # ax.plot(bin_centres,model_mc_wt_phase[n-1,0,:],
    #                  color='blue', label='Phase model')
    
    # ax.plot(bin_centres,model_mc_wt_oddeven[n-1,0,:],
    #                  'b--', label='OddEven model')
    
    
    
    
    #find the events
    mask =  np.log10(gle_df['F1']) >= thresh
    events = gle_df[mask]
    events = events.reset_index()
    #compute the wiating times
    nGLEs = len(events)
    waiting_time = np.empty((nGLEs-1))
    for i in range(0,nGLEs-1):
        waiting_time[i] = np.log10((events['mjd'][i+1] - events['mjd'][i]))

    ax.hist(waiting_time, bins = wt_histbins, histtype = 'step', density = False,
              label = lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs),
              linewidth = 2, color='k')
    

    
    if n==1 or n==3:
        ax.set_ylabel('Occurrence frequency')
    if n == 3 or n==4:
        ax.set_xlabel('Waiting time [days]')
        
    #ax.legend(loc ='upper left')
    
    ax.text(0.03, 0.93, lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs), 
            transform = ax.transAxes, backgroundcolor = 'w')
    ax.set_xlim([-0.1,4.1])
    #ax.set_xticks([np.log10(1), np.log10(30), np.log10(365), np.log10(11*365)])
    ax.set_xticks([0,1,2,3,4])
    #ax.set_xticklabels(['1 day','1 month', '1 year', '11 years'])
    ax.set_xticklabels(['$10^0$','$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
    yy = ax.get_ylim()
    y=[yy[0],yy[1]*1.1]
    ax.set_ylim(y)
    
    dx = 0.22
    
    ax.plot([np.log10(1), np.log10(1)], y, 'r--')
    ax.text(np.log10(1)+dx, y[1]/2, '1 day', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(27), np.log10(27)], y, 'r--')
    ax.text(np.log10(27)+dx, y[1]/2, '27 days', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(365), np.log10(365)], y, 'r--')
    ax.text(np.log10(365)+dx, y[1]/20, '1 yr', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(11*365), np.log10(11*365)], y, 'r--')
    ax.text(np.log10(11*365)+dx, y[1]/20, '11 yrs', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    
    n=n+1










#plt dt
fig, axs = plt.subplots(nrows=2, ncols=2)

n=1
labels = ['(a) ' , '(b) ', '(c) ', '(d) ']

for thresh,ax,lab in zip(plot_thresholds, axs.ravel(), labels):
    mask =  np.log10(gle_df['F1']) >= thresh
    events = gle_df[mask]
    events = events.reset_index()
    #compute the wiating times
    nGLEs = len(events)
    
    #construct the "stair" series 
    xx, yylower = GetStairsSeries(wt_histbins, model_mc_dt_rand[n-1,3,:])
    xx, yyupper = GetStairsSeries(wt_histbins, model_mc_dt_rand[n-1,4,:])
    #add the MC reaults
    ax.fill_between(xx,yylower,yyupper,
                     color='red', alpha=0.3)
    xx, yylower = GetStairsSeries(wt_histbins, model_mc_dt_rand[n-1,1,:])
    xx, yyupper = GetStairsSeries(wt_histbins, model_mc_dt_rand[n-1,2,:])
    ax.fill_between(xx,yylower,yyupper,
                     color='red', alpha=0.3)
    xx, yy = GetStairsSeries(wt_histbins, model_mc_dt_rand[n-1,0,:])
    ax.plot(xx,yy,  color='red', label='Random model')
    
    #loop through each GLE and compute the dt
    dt_GLE = []
    for i in range(0,nGLEs-1):
        for k  in range(i+1,nGLEs):
            dt_GLE.append(abs(events['mjd'][i] - events['mjd'][k]))
            
    ax.hist(np.log10(dt_GLE), histtype = 'step', bins = dt_histbins,
              density = False,
              label = lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs),
              linewidth = 2, color='k')
    
    if n==1 or n==3:
        ax.set_ylabel('Occurrence frequency')
    if n == 3 or n==4:
        ax.set_xlabel(r'$\Delta t$ [days]')
        
    #ax.legend(loc ='upper left')
    
    ax.text(0.03, 0.93, lab + 'log(F) > ' + str(thresh) + '; N = ' +str(nGLEs), 
            transform = ax.transAxes, backgroundcolor = 'w')
    ax.set_xlim([-0.3,4.9])
    ax.set_xticks([0,1,2,3,4])
    #ax.set_xticklabels(['1 day','1 month', '1 year', '11 years'])
    ax.set_xticklabels(['$10^0$','$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
    yy = ax.get_ylim()
    y=[yy[0],yy[1]*1.1]
    ax.set_ylim(y)
    
    dx = 0.22
    
    ax.plot([np.log10(1), np.log10(1)], y, 'r--')
    ax.text(np.log10(1)+dx, y[1]/2, '1 day', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(27), np.log10(27)], y, 'r--')
    ax.text(np.log10(27)+dx, y[1]/2, '27 days', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(365), np.log10(365)], y, 'r--')
    ax.text(np.log10(365)+dx, y[1]/2, '1 yr', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
    ax.plot([np.log10(11*365), np.log10(11*365)], y, 'r--')
    ax.text(np.log10(11*365)+dx, y[1]/20, '11 yrs', ha = 'center', backgroundcolor = 'w',
            rotation = -90)
    
        #add model values
    # ax.plot(bin_centres, model_mc_dt_rand[n-1,0,:])
    # ax.plot(bin_centres, model_mc_dt_phase[n-1,0,:])
    # ax.plot(bin_centres, model_mc_dt_oddeven[n-1,0,:])
    
    n=n+1   