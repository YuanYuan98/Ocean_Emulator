"""Date-only metadata for a frozen Julian background; no target values involved."""
import cftime
import numpy as np

def absolute_time_features(dates):
    dates=list(dates)
    if not dates or any(getattr(d, 'calendar', None) != 'julian' for d in dates):
        raise ValueError('Anomaly background requires actual Julian target dates')
    units='days since 1958-01-01 00:00:00'
    days=np.asarray(cftime.date2num(dates,units,calendar='julian'),dtype=np.float64)
    phase=[]
    for d in dates:
        start=cftime.DatetimeJulian(d.year,1,1)
        end=cftime.DatetimeJulian(d.year+1,1,1)
        a=float(cftime.date2num(d,units,calendar='julian'))
        b=float(cftime.date2num(start,units,calendar='julian'))
        e=float(cftime.date2num(end,units,calendar='julian'))
        phase.append((a-b)/(e-b))
    p=2*np.pi*np.asarray(phase)
    return np.stack((days/365.25,np.sin(p),np.cos(p),np.sin(2*p),np.cos(2*p)),axis=1)
