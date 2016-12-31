#!/bin/python
# -----------------------------------------------------------------------------
#  File:        format_yahoo_historical.py
#  Usage:       python format_yahoo_historical.py 
#  Description: Reformats the historical stock data pulled from the yahoo_finance 
#               API to simple arrays for analysis and visualization.
#  Created:     21-Dec-2016 Dustin Burns
# -----------------------------------------------------------------------------
from yahoo_finance import Share
import time
from datetime import datetime


# Reformat the date string to UNIX time
def format_date_unix(Date):
  # UNIX timestampe in days
  return [time.mktime(datetime.strptime(t, '%Y-%m-%d').timetuple())/(60*60*24) for t in Date]
 

# Reformat the yahoo_finance data structure
def format_share(name, start_date, end_date):
  
  #Retrieve stock data over given time range
  share = Share(name)
  data  = share.get_historical(start_date, end_date)
  
  # Initialize arrays
  Volume    = []
  Symbol    = []
  Adj_Close = []
  High      = []
  Low       = []
  Date      = []
  Close     = []
  Open      = []

  # Loop through the default data structure, an array of dictionaries, filling arrays to return
  for i in range(0, len(data)):
    Volume    .append(data[i]['Volume'])
    Symbol    .append(data[i]['Symbol'])
    Adj_Close .append(data[i]['Adj_Close'])
    High      .append(data[i]['High'])
    Low       .append(data[i]['Low'])
    Date      .append(data[i]['Date'])
    Close     .append(data[i]['Close'])
    Open      .append(data[i]['Open'])
  return (Volume, Symbol, Adj_Close, High, Low, format_date_unix(Date), Close, Open)


# Find Yahoo stock data by default, otherwise, call format_share function from separate project
if __name__ == "__main__":
  data = format_share('YHOO', '2014-04-25', '2014-04-29')
