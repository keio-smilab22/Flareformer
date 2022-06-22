"""
==============================
Flare times on a GOES XRS plot
==============================

How to plot flare times as provided by the HEK on a GOES XRS plot.
"""
import matplotlib.pyplot as plt

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import parse_time
from sunpy.timeseries import TimeSeries

###############################################################################
# Let's grab GOES XRS data for a particular time of interest and the HEK flare
# data for this time from the NOAA Space Weather Prediction Center (SWPC).

tr = a.Time('2017-06-07 04:00', '2017-06-07 23:00')
results = Fido.search(tr, a.Instrument.xrs & a.goes.SatelliteNumber(15) | a.hek.FL & (a.hek.FRM.Name == 'SWPC'))  # NOQA

###############################################################################
# Then download the XRS data and load it into a TimeSeries.

files = Fido.fetch(results)
goes = TimeSeries(files, concatenate=True)
print(goes.to_dataframe())


###############################################################################
# Next let's retrieve `~sunpy.net.hek.HEKTable` from the Fido result
# and then load the first row from HEK results into ``flares_hek``.

hek_results = results['hek']
flares_hek = hek_results

###############################################################################
# Lets plot everything together.

plt.figure()
goes.plot(columns=["xrsb"])
for f in flares_hek:
    plt.axvline(parse_time(f['event_peaktime']).datetime)
    plt.axvspan(parse_time(f['event_starttime']).datetime,
                parse_time(f['event_endtime']).datetime,
                alpha=0.2, label=f['fl_goescls'])
plt.legend(loc=2)
plt.yscale('log')

plt.show()
