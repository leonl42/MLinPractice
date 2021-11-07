# -*- coding: utf-8 -*-

"""
Preprocessor that creates timedeltas for the tweet datetimes.
Output column is a list of 3 timedeltas. Namely one for the year,
one for the date and one for the time

Created on 22.10.2021 6:15 pm
@author: leonl42
"""


from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_DATE, COLUMN_TIME, COLUMN_TIMEZONE, COLUMN_TIMEDELTAS,TWITTER_CREATION_YEAR,TWITTER_CREATION_DATE,TWITTER_CREATION_TIME
from datetime import datetime, timezone


class Timedeltas(Preprocessor):
    """Creates timedeltas for the year,date and time the tweet was posted"""
    
    # constructor
    def __init__(self):
        
        # take all columns with important time information and create a new output column
        super().__init__([COLUMN_DATE, COLUMN_TIME, COLUMN_TIMEZONE], COLUMN_TIMEDELTAS)
        
    def _calc_timedeltas(self, datetimes):
        """return an array of array of 3 timedeltas. One for the year, one for the date and one for the time.
        The reference time is the twitter cration day"""
        
        # convert all datetimes to utc standard time
        datetimes = [datetime.strptime(datetime.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S') for datetime in datetimes]
        
        # isolate the year for its own timedelta
        datetime_years = [datetime.strptime(str(datetime.date().year), '%Y') for datetime in datetimes]
        
        # isolate the date for its own timedelta
        # we still have to include the year here, to avoid a "day is out of range for month" error
        # if for example we have the 29th february, but a year where this day doesnt exist
        datetime_dates = [datetime.strptime(str(datetime.date().year) + "-" + str(datetime.date().month) + "-" + str(datetime.date().day),'%Y-%m-%d') for datetime in datetimes]
        
        # isolate the time for its own timedelta
        datetime_times = [datetime.strptime(str(datetime.time()), '%H:%M:%S')  for datetime in datetimes]
        
        timedeltas = []
        for year,date,time in zip(datetime_years,datetime_dates,datetime_times):
            
            # to give the date its own timedelta, we have to set the year of the TWITTER_CREATION_DATE to the year
            # of the datetime (we had to include the year in the datetime because of the error)
            twitter_creation_date = datetime.strptime(str(year.date().year) + "-" + str(TWITTER_CREATION_DATE.month) + "-" + str(TWITTER_CREATION_DATE.day),'%Y-%m-%d')
            
            # create timedeltas by just subtracting them from the reference time (twitter creation day)
            timedeltas.append([(year-TWITTER_CREATION_YEAR).total_seconds(),(date-twitter_creation_date).total_seconds(),(time-TWITTER_CREATION_TIME).total_seconds()])
            
        return timedeltas
    
    # no need to implement _set_values as no internal variables have to be set
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        """"get timedeltas"""
        
        dates = inputs[0]
        times = inputs[1]
        timezones = inputs[2]
        
        # create datetimes from the column informations
        datetimes = [datetime.strptime(str(date) + " " + str(time) + " " + str(timezone), '%Y-%m-%d %H:%M:%S %z') for date,time,timezone in zip(dates,times,timezones)]
       
        return self._calc_timedeltas(datetimes)
    