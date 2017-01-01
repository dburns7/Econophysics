#!/bin/python
# -----------------------------------------------------------------------------
#  File:        pandas_template.py
#  Usage:       python pandas_template.py 
#  Description: A template file for building projects to analyze historical stock
#               data using pandas
#  Created:     31-Dec-2016 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share


if __name__ == "__main__":

# Define input parameters
ticker     = 'YHOO'
start_date = '2013-04-25'
end_date   = '2014-04-29'

# Get data
share = Share(ticker)
data  = share.get_historical(start_date, end_date)
data  = pd.DataFrame(data)
data.Close = data.Close.astype(float)
data.Date  = pd.to_datetime(data.Date)
data.set_index(data.Date, inplace=True)

# Analyze the data!
# ...

