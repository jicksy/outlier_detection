import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class outlier_detection():
    """ Outlier Detection for Time Series Data
    
    This class helps to identify outliers in a given Time Series data.
    A Data point is identified as an outlier, if it is more than 1.5 standard deviations apart from its calculated moving average. 
    The outcome of this task is a plot, that highlights all found outliers. 
    
    Parameters
    ----------
    method: str
        The method used to smooth data points, in this case we are using "moving_average" as the default parameter
    """
    def __init__(self, method='moving_average'):
        self.method = method
        
    def moving_average(self, data, window_size=7):
        '''Moving average calculated based on the window size and input data
        
        Paramters
        ---------
        data: dataframe
            Data to calculate moving average for
        window_size:
            Window size for which to calculate the moving average
        
        Attributes
        ----------
        rolling_average: 
            Moving average calculated based on window size and input data
        
        Returns
        -------
        A plot that highlights all found outlier points
        '''
        # Calculate Simple moving average (SMA) based on the given window_size
        data['SMA'] = data['ctr'].rolling(window=window_size).mean()
        # Calculate standard deviation. 
        data['sdev'] =  np.std(data['ctr'])
        # Identify outlier values
        # A data point is identified as an outlier if it is 1.5 standard deviation apart from its calculated moving average.
        data['outlier'] = np.where(data['ctr'] > (data['SMA'] + (1.5 * data['sdev'])), 1, \
                            (np.where(data['ctr'] < (data['SMA'] - (1.5 * data['sdev'])), 1, 0)))
        data['outlier_values'] = np.where(data['outlier']==1, data['ctr'], np.nan)
        
        # Plot a time series detecting all found outliers
        plt.figure(figsize=[20,12])
        plt.plot(data['ctr'],label='CTR')
        plt.plot(data['SMA'],label='Simple Moving Average ' + str(window_size) + ' hours')
        plt.plot(data['outlier_values'], 'ko', ms= 10, label='Outlier points')
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=35)
        plt.title('Outlier Detection for CTR data aggregated by hour')
        plt.legend(loc=2)
