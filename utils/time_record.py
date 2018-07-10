# author: viaeou
import pandas as pd


class time_recored(object):

    def __init__(self, appname, starttime, endtime):
        self.appname = appname
        self.starttime = starttime
        self.endtiem = endtime

    def record(self):
        df_time_record = pd.read_csv('../data/time_record.csv')
        df_time_record['appname'].append(self.appname)
        df_time_record['starttime'].append(self.starttime)
        df_time_record['endtime'].append(self.endtiem)
        df_time_record.to_csv('../data/time_record.csv', index=False)
        return df_time_record


