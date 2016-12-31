#!/bin/python
# -----------------------------------------------------------------------------
#  File:        project_template.py
#  Usage:       python project_template.py 
#  Description: A template file for building projects to analyze historical stock
#               data, calling the format_yahoo_historical.py script to reformat
#               the data structure returned by the yahoo_finance API.
#  Created:     21-Dec-2016 Dustin Burns
# -----------------------------------------------------------------------------
import format_yahoo_historical

# Define input parameters
ticker     = 'YHOO'
start_date = '2014-04-25'
end_date   = '2014-04-29'

# Get data
data = format_yahoo_historical.format_share(ticker, start_date, end_date)

# Analyze the data!
# ...

