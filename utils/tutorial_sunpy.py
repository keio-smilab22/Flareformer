import numpy as np
import sunpy.data.sample
import sunpy.timeseries as ts


def time_series_tutorial():
    my_timeseries = ts.TimeSeries(sunpy.data.sample.GOES_XRS_TIMESERIES, source='XRS')
    print(sunpy.data.sample.GOES_XRS_TIMESERIES)
    my_timeseries.peek()


if __name__ == '__main__':
    time_series_tutorial()