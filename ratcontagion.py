# Copyright 2023 Ilya Starkov. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import zipfile
import os
import datetime
import warnings
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.graph_objects as go
import copy
import pymice as pm
import networkx as nx
import random
import scipy.stats as stats
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import reduce
import plotly.express as px
from plotly.offline import iplot
from math import factorial


class RBCA:
    def __init__(self,
                 input_path: str = None,
                 output_path: str = None,
                 name_animal_file: str = None,
                 dict_names: dict = None,
                 corners: list = None,
                 show_warnings: bool = True):
        """
        ------------ 
        Function:
            Class initialization with definition of the arguments.
        ------------
        Parameters:
            input_path: str
                Path to the folder where are the necessary files.
                You can define zipped or non zipped folders. But if you prefer to work with the IntelliCage archives,
                then your folder need to be non zipped.
                Example: r"C:/Users/...input" (the slashes were flipped for this example)
            output_path: str
                Path to the folder where you want to store the output files.
            name_animal_file: str
                Name of your animal file. It must be in the 'input_path'.
                Example: 'Animal' (it must be a html/csv/xlsx file, but the most preferred is xlsx, because it's explored more)
            dict_names: dict
                Dictionary with lists of file names in the input path.
                Each name in the list is the name of the file where the visit data is located.
                The first word must be the name of a rat group and
                after that it must be space. Example: 'control1 session2' or 'name_group session1 stage 2'.
                It can't be the same group with different stages/condition/cases in the one list. 
                Mistake: {'condition 1':['group1 stage 1', 'group1 stage 2', 'group2 stage 1']}
                Right example: {'condition 1':['group1 stage 1', 'group2 stage 1'],
                                'condition 2':['group1 stage 2']}
            corners: list
                List with original corner labels
                Example: [1,2] or [1]
            show_warnings: bool
                Show warnings (pandas, plotly etc.) or not. Default value is True.
                Example: True or False
        ------------
        Return:
            Class instance of the experiment.
        ------------
        Pay attention:
            There are static methods in this class.
        """

        self.input_path = input_path
        self.output_path = output_path
        self.name_animal_file = name_animal_file
        self.animal_file = None
        self.dict_names = dict_names
        self.corners = corners
        self.log = {'medians': {},
                    'graph_analysis': defaultdict(list),
                    'unique tags (slice)': {},
                    'demonstrators': {},
                    'without_dem': {},
                    'tags_out_bounds': {},
                    'inner_tags':{}}
        self.global_oper = 0

        if not show_warnings:
            warnings.filterwarnings("ignore")

        if input_path is not None:
            if input_path.endswith('.zip'):
                with zipfile.ZipFile(input_path) as z:
                    self.animal_file = self.file_opener(base=[text_file.filename for text_file in z.infolist()],
                                                        file_name=self.name_animal_file,
                                                        zip_=True)
            else:
                self.animal_file = self.file_opener(base=(next(os.walk(self.input_path), (None, None, []))[2]),
                                                    file_name=self.name_animal_file,
                                                    zip_=False)
            self.animal_file['Demonstrator'] = self.animal_file['Demonstrator'].fillna(0).astype(int)
            self.animal_file = self.animal_file.astype(str)


    def to_log(self,
               record):
        """
        ------------
        Function:
            Logging to file (log.txt)
        ------------
        Parameters:
            record: Any
        ------------
        Return:
            None
        ------------
        Side effect:
            Write needed input record to the log
        """

        if self.output_path is not None:
            file_name = f'{self.output_path}\\log.txt'
            with open(file_name, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{datetime.now()}_________________________ {record}\n')


    def file_opener(self,
                    base,
                    file_name: str,
                    zip_: bool = False):
        """
        ------------
        Function:
            Open html/xlsx/csv files (also in zip)
        ------------
        Parameters:
            base: iterable
                List with file names in the directory
            file_name: str
                Name of the necessary file
        ------------
        Return:
            pandas.Dataframe from panel data in the file
        """

        raw_file = None
        for inner_file_name in base:
            if inner_file_name[:len(file_name)] == file_name:
                if inner_file_name[-4:] == 'html':
                    if zip_:
                        raw_file = pd.read_html(z.open(f'{file_name}.html'), header=1)[0]
                    else:
                        raw_file = pd.read_html(f'{self.input_path}\\{file_name}.html', header=1)[0]
                elif inner_file_name[-3:] == 'csv':
                    if zip_:
                        raw_file = pd.read_csv(z.open(f'{file_name}.csv'))
                    else:
                        raw_file = pd.read_csv(f'{self.input_path}\\{file_name}.csv')
                elif inner_file_name[-4:] == 'xlsx':
                    if zip_:
                        raw_file = pd.read_excel(z.open(f'{file_name}.xlsx'))
                    else:
                        raw_file = pd.read_excel(f'{self.input_path}\\{file_name}.xlsx')
                else:
                    self.to_log(
                        f'\nThere is the {file_name}, but it is not the right format (html/csv/xlsx) in the directory {self.input_path}')
        if raw_file is None:
            self.to_log(f'\nThere is no the {file_name} in the directory {self.input_path}')

        return raw_file


    def date_transformer(self,
                         df,
                         column_with_date: str):
        """
        ------------
        Function:
            Correcting dates, because of formats like 'dd.mm' (or 'd.mm','dd.m','d.m') (sometimes it was).
        ------------
        Parameters:
            df: pandas.DataFrame
            column_with_date: str
                Name of the column with needed dates to transform or just check.
        ------------
        Return:
            List with transformed dates
        ------------
        Pay attention:
            If there are the same dates (e.g. '2022-01-13' and '2023-01-13'),
            the function will choose the first date in list_with_whole_dates.
        """

        unique_list = df[column_with_date].unique()
        list_with_whole_dates = [x for x in unique_list if len(x) > 5]

        sup_list = []
        for date in df[column_with_date]:
            if len(date) > 5:
                sup_list.append(date)
            else:
                _date = date.split('.')
                # Can be mistaken in case having 2022-05-02 and 2022-02-05
                if (_date[0] in [i[2] for i in [j.split('-') for j in list_with_whole_dates]] or
                    '0' + _date[0] in [i[2] for i in [j.split('-') for j in list_with_whole_dates]]) and \
                        (_date[1] in [i[1] for i in [j.split('-') for j in list_with_whole_dates]] or
                         '0' + _date[1] in [i[1] for i in [j.split('-') for j in list_with_whole_dates]]):
                    for i in list_with_whole_dates:
                        if i[-2:] == _date[0] or i[-2:] == '0' + _date[0]:
                            sup_list.append(i)
                else:
                    self.to_log(f'\nError (func - date_transformer): date = {date}, sup_list = {sup_list}.')
        return sup_list


    @staticmethod
    def time_operator(time,
                      shift_time,
                      illum: float,
                      shift_illum: float):
        """
        ------------
        Function:
            Helper function to detect changing the illumination
        ------------
        Parameters:
            time: Any
                Time of some moment
            shift_time: Any
                Shifted time of some moment
            illum: float
                Value of illumination from the IntelliCage output archive
            shift_illum: float
                Shifted value of illumination                 
        ------------
        Return:
            shift_time or time or None (regarding the condition)        
        """
        if illum != 0 and shift_illum == 0:
            return shift_time
        elif illum == 0 and shift_illum != 0:
            return time
        else:
            return None


    def intellicage_parser(self,
                           name_base: str,
                           illumination='all_time'):
        """
        ------------
        Function:
            Function to upload data from the IntelliCage output archives (.zip files).
        ------------
        Parameters:
            name_base: str
                Name of the file to upload data with visits.
            illumination: True or False or 'all_time'
                If it is True, you will upload data only with non-zero illumination values (daylight time).
                If it is False, you will upload data only with zero illumination values (nighttime).
                If it is 'all_time', you will upload all available data from the IntelliCage archive.
                Default value is 'all_time'.
        ------------
        Return:
            pandas.DataFrame (with visit data from the IntelliCage)
        ------------
        Pay attention:
            This method include tools from PyMICE:
                Loader,
                getVisits,
                getEnvironment
            Please, read more about the IntelliCage, and it's output files:
            https://www.tse-systems.com/service/intellicage/
            And PyMICE tools: https://github.com/Neuroinflab/PyMICE
        """

        ml = pm.Loader(f'{self.input_path}\\{name_base}.zip',
                       getNp=True,
                       getLog=True,
                       getEnv=True,
                       getHw=True,
                       verbose=False)

        raw_visits = ml.getVisits()

        sup_dict = {'VisitID': [],
                    'Name': [],
                    'Sex': [],
                    'Tag': [],
                    'AntennaDuration': [],
                    'AntennaNumber': [],
                    'Cage': [],
                    'Corner': [],
                    'Left': [],
                    'Right': [],
                    'CornerCondition': [],
                    'Duration': [],
                    '_EndTime': [],
                    'LickContactTime': [],
                    'LickDuration': [],
                    'LickNumber': [],
                    'Module': [],
                    'NosepokeDuration': [],
                    'NosepokeNumber': [],
                    'PlaceError': [],
                    'PresenceDuration': [],
                    'PresenceNumber': [],
                    '_StartTime': [],
                    'VisitSolution': [],
                    'StartDate': [],
                    'EndDate': []}

        for idx, visit in enumerate(raw_visits):
            sup_dict['VisitID'].append(idx)
            sup_dict['Name'].append(visit.Animal.Name)
            sup_dict['Sex'].append(visit.Animal.Sex)
            sup_dict['Tag'].append(int(list(visit.Animal.Tag)[0]))
            sup_dict['AntennaDuration'].append(visit.AntennaDuration.total_seconds())
            sup_dict['AntennaNumber'].append(visit.AntennaNumber)
            sup_dict['Cage'].append(visit.Cage)
            sup_dict['Corner'].append(visit.Corner)
            sup_dict['Left'].append(visit.Corner.Left)
            sup_dict['Right'].append(visit.Corner.Right)
            sup_dict['CornerCondition'].append(visit.CornerCondition)
            sup_dict['Duration'].append(visit.Duration.total_seconds())
            sup_dict['_EndTime'].append(visit.End)
            sup_dict['LickContactTime'].append(visit.LickContactTime.total_seconds())
            sup_dict['LickDuration'].append(visit.LickDuration.total_seconds())
            sup_dict['LickNumber'].append(visit.LickNumber)
            sup_dict['Module'].append(visit.Module)
            sup_dict['NosepokeDuration'].append(visit.NosepokeDuration.total_seconds())
            sup_dict['NosepokeNumber'].append(visit.NosepokeNumber)
            sup_dict['PlaceError'].append(visit.PlaceError)
            sup_dict['PresenceDuration'].append(visit.PresenceDuration.total_seconds())
            sup_dict['PresenceNumber'].append(visit.PresenceNumber)
            sup_dict['_StartTime'].append(visit.Start)
            sup_dict['VisitSolution'].append(visit.VisitSolution)
            sup_dict['StartDate'].append(str(visit.Start.date()))
            sup_dict['EndDate'].append(str(visit.End.date()))

        visits = pd.DataFrame(sup_dict)

        output_visits = None
        if illumination != 'all_time':
            raw_env = ml.getEnvironment(order='DateTime')
            sup_dict = {'DateTime': [],
                        'Illumination': [],
                        'Temperature': [],
                        'Cage': []}
            for moment in raw_env:
                sup_dict['DateTime'].append(moment.DateTime)
                sup_dict['Illumination'].append(moment.Illumination)
                sup_dict['Temperature'].append(moment.Temperature)
                sup_dict['Cage'].append(moment.Cage)

            env = pd.DataFrame(sup_dict)

            env['shift_illumination'] = env['Illumination'].shift()
            env['shift_datetime'] = env['DateTime'].shift()

            env['oper'] = env[['DateTime', 'shift_datetime', 'Illumination', 'shift_illumination']].apply(
                lambda x: self.time_operator(x[0],
                                             x[1],
                                             x[2],
                                             x[3]), axis=1)

            start_i = list(env.loc[(env['Illumination'] == 0) & (env['oper'].isnull() is False)]['oper'].unique())
            end_i = list(env.loc[(env['Illumination'] != 0) & (env['oper'].isnull() is False)]['oper'].unique())

            start_i = sorted(start_i)
            end_i = sorted(end_i)

            if len(start_i) == len(end_i):
                sup_dict = {}

                for start, end in list(zip(start_i, end_i)):
                    sup_dict[f'{start}-{end}'] = visits.loc[
                        (visits['_StartTime'] <= end) & (visits['_StartTime'] >= start)]
                    sup_visits = pd.concat([*sup_dict.values()], ignore_index=True)
                    sup_visits.drop_duplicates(subset=['VisitID'], keep='first', inplace=True)

                    if illumination:
                        visits.index = visits['VisitID']
                        output_visits = visits.drop(sup_visits['VisitID'], axis=0)
                        output_visits.reset_index(drop=True, inplace=True)
                    else:
                        output_visits = sup_visits

            elif len(start_i) != len(end_i):

                sup_dict1 = {}
                for start in start_i:
                    sup_dict1[f'{start}'] = visits.loc[visits['_StartTime'] >= start]
                sup_visits = pd.concat([*sup_dict1.values()], ignore_index=True)
                sup_visits.drop_duplicates(subset=['VisitID'], keep='first', inplace=True)

                sup_dict2 = {}
                for end in end_i:
                    sup_dict2[f'{end}'] = sup_visits.loc[sup_visits['_StartTime'] <= end]
                sup_visits_ = pd.concat([*sup_dict2.values()], ignore_index=True)
                sup_visits_.drop_duplicates(subset=['VisitID'], keep='first', inplace=True)

                if illumination:
                    visits.index = visits['VisitID']
                    output_visits = visits.drop(sup_visits_['VisitID'], axis=0)
                    output_visits.reset_index(drop=True, inplace=True)
                else:
                    output_visits = sup_visits_
            else:
                self.to_log(f'\nError in daylight time intervals: start_dark: {start_i}, end_dark: {end_i}.')
        else:
            output_visits = visits

        return output_visits


    def parser(self,
               name_base: str,
               name_stage: str,
               name_group: str,
               lick: bool = None,
               time: float = None,
               without_dem: bool = False,
               time_start: str = None,
               time_finish: str = None,
               intellicage=None,
               condition=None):
        """
        ------------
        Function:
            Parsing your html-file (that is built via the IntelliCage) or
            parsing original IntelliСage output archives.
        ------------
        Parameters:
            name_base: str
                Name of the file where the visit data is located. The first word must be the name of a rat group and
                after that it must be space. 
                Example: 'control1 session2' or 'name_group session1 stage 2'
            name_stage: str
                Name particular stage gor the group. It defines conditions/environment or other artifacts you wish.
                This name, strictly speaking, goes after group name and space in file name.
                Example: 'stage 1'
            name_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base'.
                Example: 'group1'
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            without_dem: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            time_start: str
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original
                finish time
                in the file, 'start_time' will be equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            condition: dict
                It is dictionary where you put parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
        ------------
        Return:
            DataFrame with behavioral data,
            list with ordinary rat tags in this data (example: [12345678,87654321]), 
            list with demonstrator's tag (example: [12345678])
        ------------
        Pay attention:
            'condition' parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this method.
        """

        if self.input_path.endswith('.zip'):
            with zipfile.ZipFile(self.input_path) as z:
                base = self.file_opener(base=[text_file.filename for text_file in z.infolist()],
                                        file_name=name_base,
                                        zip_=True)
        else:
            if intellicage is not None:
                base = self.intellicage_parser(name_base=name_base,
                                               illumination=intellicage)
            else:
                base = self.file_opener(base=(next(os.walk(self.input_path), (None, None, []))[2]),
                                        file_name=name_base,
                                        zip_=False)

        if intellicage is None:
            base['_StartDate'] = self.date_transformer(base, 'StartDate')
            base['_EndDate'] = self.date_transformer(base, 'EndDate')

            base['_StartTime'] = pd.to_datetime(base['_StartDate'] + '_' + base['StartTime'],
                                                format='%Y-%m-%d_%H:%M:%S.%f')
            base['_EndTime'] = pd.to_datetime(base['_EndDate'] + '_' + base['EndTime'],
                                              format='%Y-%m-%d_%H:%M:%S.%f')

            base['VisitDuration'] = pd.to_numeric(base['VisitDuration'])

        # =========================================================== customizable
        if condition is not None and len(base) > 0:
            for stage in condition.keys():
                if stage == name_stage:
                    for file_name in condition[stage]:
                        if file_name == name_base:
                            if condition[stage][file_name] == 'last_day':
                                last_day = sorted(list(base['StartDate'].unique()))[-1]
                                base = base[base['StartDate'] == last_day]
                            elif condition[stage][file_name] == 'first_day':
                                first_day = sorted(list(base['StartDate'].unique()))[0]
                                base = base[base['StartDate'] == first_day]
                            elif condition[stage][file_name] == 'second_day':
                                second_day = sorted(list(base['StartDate'].unique()))[1]
                                base = base[base['StartDate'] == second_day]
                            elif condition[stage][file_name][:4] == 'time':
                                condition_sup_list = condition[stage][file_name].split('|')

                                if len(condition_sup_list[1:]) % 2 == 0:
                                    sup_starts = sorted(condition_sup_list[1::2])
                                    sup_finishes = sorted(condition_sup_list[2::2])

                                    sup_dict = {}
                                    for start, finish in list(zip(sup_starts, sup_finishes)):

                                        if len(start) > 8 and len(finish) > 8:
                                            begin = pd.to_datetime(start, format='%Y-%m-%d_%H:%M:%S')
                                            end = pd.to_datetime(finish, format='%Y-%m-%d_%H:%M:%S')
                                            sup_dict[f'{start}-{finish}'] = base.loc[
                                                (base['_StartTime'] <= end) & (base['_StartTime'] >= begin)]
                                        else:
                                            self.to_log(
                                                f'\nError when entering time in the condition in {name_base} of {name_stage}.')

                                    base = pd.concat([*sup_dict.values()], ignore_index=True)
                                    base.drop_duplicates(subset=['VisitID'], keep='first', inplace=True)
                                else:
                                    self.to_log(
                                        f'''\nError when entering time in the condition in {name_base} of {name_stage}.
                                            \rIt have to be time pairs.''')

            if 'date' in condition.keys():
                spec_date = condition['date']
                base = base[base['StartDate'] == spec_date]
        # ===========================================================

        dems = list(set(list(self.animal_file.loc[(self.animal_file['Demonstrator'] == '1') &
                                                  (self.animal_file['Group'] == name_group),
                                                  'Animal ID'])))

        if name_stage not in self.log['inner_tags']:
            self.log['inner_tags'][name_stage] = {name_base:{}}
        for i in list(self.animal_file.loc[self.animal_file['Group'] == name_group, 'Animal ID']):
            if name_base not in self.log['inner_tags'][name_stage]:
                self.log['inner_tags'][name_stage][name_base]={}
            self.log['inner_tags'][name_stage][name_base][i] = self.global_oper
            self.global_oper += 1

        base['Tag'] = base['Tag'].astype(str)

        dem_in_base = []
        base_list = list(base['Tag'].unique())

        for i in dems:
            if i in base_list:
                dem_in_base.append(i)
                if without_dem:
                    base_list.remove(i)

        self.log['demonstrators'][name_base] = dem_in_base
        self.to_log(f'\nThe demonstrator in {name_base} - {dem_in_base}')

        base.index = base['Tag']

        if without_dem:
            base_out_dem = base.drop(dem_in_base, axis=0)
        else:
            base_out_dem = base

        if name_stage not in self.log['without_dem']:
            self.log['without_dem'][name_stage] = {name_base: without_dem}
        else:
            self.log['without_dem'][name_stage][name_base] = without_dem

        self.log['unique tags (slice)'][name_base] = list(base_out_dem['Tag'].unique())

        if time is not None and (time_start is None and time_finish is None):
            start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
            finish = start + pd.offsets.DateOffset(minutes=time)
            base_out_dem = base_out_dem.loc[base_out_dem['_StartTime'] <= finish]
            self.to_log(f'\nStart time of downloading {name_base} = {start}, the end of download = {finish}.')

        if time_start is not None and time_finish is not None:

            if len(time_start) > 8 or len(time_finish) > 8:
                start = pd.to_datetime(time_start, format='%Y-%m-%d_%H:%M:%S')
                finish = pd.to_datetime(time_finish, format='%Y-%m-%d_%H:%M:%S')
            else:
                if len(base['StartDate'].unique()) > 1:
                    self.to_log(f'''\nAttention: number of unique dates in "StartDate" more than one.
                        \rData with this exception: {name_base}.
                        \rIt will be used the first date for date analysis and comparison.''')

                start = pd.to_datetime(sorted(list(base['StartDate'].unique()))[0] + '_' + time_start,
                                       format='%Y-%m-%d_%H:%M:%S')
                finish = pd.to_datetime(sorted(list(base['StartDate'].unique()))[0] + '_' + time_finish,
                                        format='%Y-%m-%d_%H:%M:%S')

            try:
                base_out_dem = base_out_dem.loc[
                    (base_out_dem['_StartTime'] <= finish) & (base_out_dem['_StartTime'] >= start)]
            except:
                if sorted(list(base['_StartTime'].unique()))[0] > start:
                    self.to_log(f'\nError during preprocessing of {name_base}:')
                    self.to_log(
                        'Parameter "time_start" is less than the real start time of the data. The first date will be used as a time_start.')
                    start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
                if sorted(list(base['_StartTime'].unique()))[-1] < finish:
                    self.to_log(f'\nError during preprocessing of {name_base}:')
                    self.to_log(
                        'Parameter "time_finish" is bigger than the real finish time of the data. Last date will be used as a time_finish.')
                    finish = pd.Timestamp(sorted(list(base['_StartTime']))[-1])

            base_out_dem = base_out_dem.loc[
                (base_out_dem['_StartTime'] <= finish) & (base_out_dem['_StartTime'] >= start)]

            if time is not None:
                start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
                finish = start + pd.offsets.DateOffset(minutes=time)
                base_out_dem = base_out_dem.loc[base_out_dem['_StartTime'] <= finish]

            self.to_log(f'\nStart time of downloading {name_base} = {start}, the end of download = {finish}.')
        else:
            start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
            finish = pd.Timestamp(sorted(list(base['_StartTime']))[-1])
            self.to_log(f'\nStart time of downloading {name_base} = {start}, the end of download = {finish}.')

        if lick is False:
            base_out_dem = base_out_dem[base_out_dem['LickNumber'] == 0]
        if lick:
            base_out_dem = base_out_dem[base_out_dem['LickNumber'] != 0]

        base_out_dem.sort_values('_StartTime', inplace=True)

        return base_out_dem, base_list, dem_in_base


    @staticmethod
    def inner_analysis(data,
                       data_,
                       tags: list,
                       time_interval: float,
                       special_name: str):
        """
        ------------
        Function:
            Analyzing which rat went into the drinking bowl after which rat, doing it in the form of a dictionary.
            Works only with one situation/bowls/condition/case.
        ------------
        Parameters:
            data: pandas.DataFrame
                Slice of the data where this function will search for needed intervals after each visit.
            data_: pandas.DataFrame
                Slice of the data where this function will search for followers in found intervals.
            tags: list
                List with needed rat tags.
            time_interval: float
                Float number of seconds.
                Example: 12.4
            special_name: str
                Custom string identifier for necessary rat group.
                This name is used in column naming during data processing.
                Example: 'visits in stage1 group1'
        ------------
        Return:
            Dictionary where keys are rat tags and each value is a pandas.DataFrame with followers gathered from one
            case.
        """

        sup_dict = {}

        for tag in tags:

            filter_data = data[data['Tag'] == tag]
            filter_data['EndTime_before'] = filter_data['_EndTime'].shift()
            filter_data['diff'] = (filter_data['_StartTime'] - filter_data['EndTime_before']).dt.seconds
            filter_data['diff'].fillna(value=0, inplace=True)
            filter_data.sort_values('_StartTime', inplace=True)

            _tags = tags.copy()
            _tags.remove(tag)

            name_column = f'Visits in {special_name}'

            df_for_tag = pd.DataFrame(data=0, columns=[name_column], index=_tags)

            for index, row in filter_data.iterrows():

                # Processing the general case
                start = pd.Timestamp(row['EndTime_before'])
                if row['diff'] <= time_interval:
                    finish = row['_StartTime']
                else:
                    finish = start + pd.offsets.DateOffset(seconds=time_interval)

                filtered_df = data_.loc[(data_['Tag'] != tag) & (
                        (data_['_StartTime'] >= start) & (data_['_StartTime'] <= finish))].copy()
                filtered_df.sort_values('_StartTime', inplace=True)

                if len(filtered_df) != 0:
                    filtered_df.sort_values('_StartTime', inplace=True)
                    needed_tag = filtered_df.iloc[0]['Tag']
                    df_for_tag.loc[needed_tag, name_column] = df_for_tag.loc[needed_tag, name_column] + 1

                # Processing the last row
                if row['_StartTime'] == filter_data.iloc[-1]['_StartTime']:

                    start = pd.Timestamp(row['_EndTime'])
                    finish = start + pd.offsets.DateOffset(seconds=time_interval)

                    filtered_df = data_.loc[
                        (data_['Tag'] != tag) & ((data_['_StartTime'] > start) & (data_['_StartTime'] < finish))]

                    if len(filtered_df) != 0:
                        filtered_df.sort_values('_StartTime', inplace=True)
                        needed_tag = filtered_df.iloc[0]['Tag']
                        df_for_tag.loc[needed_tag, name_column] = df_for_tag.loc[needed_tag, name_column] + 1

            sup_dict[tag] = df_for_tag

        return sup_dict


    def combiner(self,
                 dict1: dict,
                 dict2: dict):
        """
        ------------
        Function:
            Combine data about all rat entries in needed corners.
        ------------
        Parameters:
            dict1: dict
                Dictionary where keys are rat tags and values are pandas.DataFrame.
                Indexes in frame are about other rats, which go to a corner after rat in the key.
                In such frames every value is a number of trips.
            dict2: dict
                Exactly like dict1, but another corner.
        ------------
        Return:
            One dictionary with combined data.
        """

        if dict1 is not None and len(dict1) > 0 and dict2 is not None and len(dict2) > 0:
            sup_dict_both = {}
            for k, v in dict1.items():
                if k in dict2.keys():
                    for r, t in dict2.items():
                        if k == r:
                            v = v.reset_index()
                            t = t.reset_index()
                            df = pd.concat([v, t], ignore_index=True)
                            df = df.groupby('index').sum()
                            sup_dict_both[k] = df

                elif k not in dict2.keys():
                    sup_dict_both[k] = v
            for r, t in dict2.items():
                if r not in dict1.keys():
                    sup_dict_both[r] = t
        elif dict1 is None or len(dict1) == 0 and dict2 is not None and len(dict2) > 0:
            sup_dict_both = dict2
        elif dict2 is None or len(dict2) == 0 and dict1 is not None and len(dict1) > 0:
            sup_dict_both = dict2
        else:
            sup_dict_both = None

        groupname_from_dict = list(list(sup_dict_both.values())[0].columns)[0].split('_')[-1].split()[0] # that's right, because in processing these are the names
        file_name = list(list(sup_dict_both.values())[0].columns)[0].split('_')[-1]

        stage_name = None
        for stage in self.dict_names:
            if stage in list(list(sup_dict_both.values())[0].columns)[0]:
                stage_name = stage

        if stage_name is None:
            self.to_log(f'{stage_name} is not founded in {list(list(sup_dict_both.values())[0].columns)[0]}')

        if self.log['without_dem'].get(stage_name):
            without_dem = self.log['without_dem'][stage_name][file_name]
        else:
            without_dem = False

        tags_in_animal = \
            self.animal_file[self.animal_file['Group'] == groupname_from_dict] \
                [self.animal_file['Demonstrator'] == '0']['Animal ID'].to_list() if without_dem else \
                self.animal_file[self.animal_file['Group'] == groupname_from_dict]['Animal ID'].to_list()

        tags_out_bounds = list(set(tags_in_animal)-set(sup_dict_both))
        tags_in_bounds = list(sup_dict_both)

        if stage_name not in self.log['tags_out_bounds']:
            self.log['tags_out_bounds'][stage_name] = {file_name:tags_out_bounds}
        else:
            self.log['tags_out_bounds'][stage_name][file_name] = tags_out_bounds

        for tag in tags_out_bounds:
            if tag not in sup_dict_both:
                sup_dict_both[tag] = pd.DataFrame(columns=[list(list(sup_dict_both.values())[0].columns)[0]],
                                                  index = tags_in_bounds,
                                                  data = [0]*len(tags_in_bounds))
                sup_dict_both[tag].index.name = 'index'

        return sup_dict_both


    def frame_analysis_for_graph(self,
                                 data,
                                 tags: list,
                                 dem_tags: list,
                                 time_interval: float,
                                 net,
                                 corners: list = None,
                                 dynamic: bool = False,
                                 special_name: str = '_'):
        """
        ------------
        Function:
            Analyzing which rat went into the drinking bowl after which rat and add this information to the graph,
            gathering information from all situations/bowls/conditions/cases.
        ------------
        Parameters:
            data: pandas.DataFrame
                Slice of the data where this function will search for needed intervals after each visit.
            tags: list
                List with needed rat tags.
            dem_tags: list
                List with demonstrators' tags
            time_interval: float
                Float number of seconds. The number of seconds after each visit during which the visits of
                followers are counted.
                Example: 12,4 or 30
            net: pyvis.network.Network
                The instance of pyvis.network.Network (the method adds nodes and edges from the input dictionary
                to this structure).
                Example: pyvis.network.Network()
            corners: list
                List with original corner labels
                Example: [1,2] or [1]
                Default: None
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False
                Example: True or False
            special_name: str
                Custom string identifier for necessary rat group.
                This name is used in column naming during data processing.
                Example: 'visits in stage1 group1'
                Default: '_'
        ------------
        Return:
            Dictionary where keys are rat tags and each value is a pandas.DataFrame with followers gathered from
            all cases.
        ------------
        Side effect:
            Adding nodes and edges to the input pyvis.network.Network.
        """

        name_column = f'Visits in {special_name}' #customizable

        corners_dict = dict.fromkeys(corners)
        for corner in corners:
            data_of_corner = data[data['Corner'] == corner]
            corners_dict[corner] = self.inner_analysis(data_of_corner,
                                                       data_of_corner.copy(),
                                                       tags,
                                                       time_interval,
                                                       special_name)

        if len(corners) >= 2:
            final_dict = reduce(self.combiner, list(corners_dict.values()))
        elif len(corners) == 0:
            final_dict = None
        else:
            final_dict = list(corners_dict.values())[0]

        if not dynamic:
            for k, v in final_dict.items():

                file_name = list(v.columns)[0].split('_')[-1]
                stage_name = None
                for stage in self.dict_names:
                    if stage in list(v.columns)[0]:
                        stage_name = stage
                if stage_name is None:
                    self.to_log(f'{stage_name} is not founded in {list(v.columns)[0]}')

                int_k = int(k)
                if int_k in net.get_nodes():
                    int_k = self.log['inner_tags'][stage_name][file_name][k]

                if k in dem_tags:
                    net.add_node(int_k, label=str(k), color='yellow', title=f'{v}', shape='star')
                elif k in str(self.log['tags_out_bounds'][stage_name][file_name]) and v.sum()[0]==0:
                    net.add_node(int_k, label=str(k), color='grey', title=f'{v}', shape = 'diamond')
                else:
                    net.add_node(int_k, label=str(k), color='red', title=f'{v}')

            sup_graph_list = []
            for k, v in final_dict.items():

                file_name = list(v.columns)[0].split('_')[-1]
                stage_name = None
                for stage in self.dict_names:
                    if stage in list(v.columns)[0]:
                        stage_name = stage
                if stage_name is None:
                    self.to_log(f'{stage_name} is not founded in {list(v.columns)[0]}')

                int_k = int(k)
                if self.log['inner_tags'][stage_name][file_name][k] in net.get_nodes():
                    int_k = self.log['inner_tags'][stage_name][file_name][k]

                for tag in v.index:
                    if v.loc[tag, name_column] == 0:
                        sup_graph_list.append([int_k, int(tag)])
                    else:
                        net.add_edge(int_k, int(tag), value=0.01 * int(v.loc[tag, name_column]), color='blue')
            for pair in sup_graph_list:
                net.add_edge(pair[0], pair[1], hidden=True)

        return final_dict


    def graph(self,
              input_data,
              tags: list,
              dem_tags: list,
              net,
              time_interval: float = None,
              dynamic: bool = False,
              special_name: str = '_'):
        """
        ------------
        Function:
            Dividing the data to cases via conditions, evaluating followers and their visits and adding 
            to the given graph nodes and edges. 
        ------------
        Parameters:
            input_data: pandas.DataFrame
                Frame with visit data.
            tags: list
                List with needed rat tags.
            dem_tags:list
                List with demonstrators' tags.
            net: pyvis.network.Network
                The instance of pyvis.network.Network (the method adds nodes and edges from the input dictionary to this
                structure).
                Example: pyvis.network.Network()
            time_interval:
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is None.
                Example: 12,4 or 30
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False
            special_name: str
                Custom string identifier for necessary rat group.
                This name is used in column naming during data processing.
                Example: 'visits in stage1 group1'
                Default: '_'
        ------------
        Return:
            Dictionary where keys are rat tags and each value is a pandas.DataFrame with followers from all cases.
        ------------
        Side effect:
            Adding nodes and edges to the input pyvis.network.Network.
        """

        corners_list = []
        for corner in self.corners:
            if corner not in list(input_data['Corner'].unique()):
                self.to_log(f'''\nThere is no data on the {corner} corner in the next slice:
                                \rName of the group: {special_name}
                                \rData: {input_data.head()}
                                \rTags: {tags}
                                \nThe analysis will be performed without this corner.''')
            else:
                corners_list.append(corner)

        if len(corners_list) > 0:
            graph_dict = self.frame_analysis_for_graph(data=input_data,
                                                       tags=tags,
                                                       dem_tags=dem_tags,
                                                       time_interval=time_interval,
                                                       net=net,
                                                       corners=corners_list,
                                                       dynamic=dynamic,
                                                       special_name=special_name)
        else:
            graph_dict = None

        return graph_dict


    def work(self,
             title_of_stage: str,
             names_of_files: list,
             lick: bool = None,
             time: float = None,
             time_start: str = None,
             time_finish: str = None,
             time_start_median: str = None,
             time_finish_median: str = None,
             replacing_value: float = None,
             median_special_time: bool = True,
             delete_zero_in_intervals_for_median: bool = True,
             input_time_interval=None,
             without_dem_base: bool = False,
             dynamic: bool = False,
             intellicage=None,
             parser_condition: dict = None):
        """
        ------------
        Function:
            It builds pyvis graph and provides the information about visit data (including followers' visits) 
        ------------
        Parameters:
            title_of_stage: str
                Title of one list with file names where visit data is. It is a key in the 'dict_names' in __init__(). 
                Nominally, this is the name of a cluster of files that differ in some way.
                Example: 'Stage 1'
            names_of_files: list
                List with file names where visit data is.
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
            time: float
                See the parameter 'time' of the 'parser' method.
            time_start: str
                See the parameter 'time_start' of the 'parser' method.
            time_finish: str
                See the parameter 'time_finish' of the 'parser' method.
            time_start_median: str
                Specific start time (format is like time_start) to evaluate median for statistics
                Default is None
            time_finish_median: str
                Specific finish time (format is like time_finish) to evaluate median for statistics
                Default is None
            replacing_value: float
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is None.
                Example: 12,4 or 30
            median_special_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False
            input_time_interval: float or 'auto'
                Float number of seconds or flag 'auto'. The number of seconds after each visit during which the visits
                of followers are counted. If 'auto', time interval for observing followers' visits will be evaluated
                automatically. Default value is 'auto'.
                Example: 'auto' or 12.5 or 50
            without_dem_base: bool
                See the parameter 'without_dem' of the 'parser' method.
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False 
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
        ------------
        Return:
             Tuple with:
                 1) pandas.DataFrame where indexes are rat tags and values are lists with intervals (seconds) between
                 visits;
                 2) list with dictionaries, each dictionary is a relation between rats-initiators and followers;
                 3) list with demonstrator's tags.
             Example: (pandas.DataFrame, list(dict1(), dict2(), dict3()), list)
        """

        self.to_log(f'\n======= {title_of_stage} =======')

        bases = {}
        bases_tags = {}
        bases_dems = {}

        names_base = names_of_files
        for name_base in names_base:
            name_group = name_base.split(' ')[0]
            bases[name_base], \
            bases_tags[name_base], \
            bases_dems[name_base] = self.parser(name_base=name_base,
                                                name_stage=title_of_stage,
                                                name_group=name_group,
                                                lick=lick,
                                                time=time,
                                                without_dem=without_dem_base,
                                                time_start=time_start,
                                                time_finish=time_finish,
                                                intellicage=intellicage,
                                                condition=parser_condition)

        if median_special_time:  # special computing space
            self.to_log('Special calculations (median over the whole time, for each group).')

            bases_all = {}
            bases_tags_all = {}
            bases_dems_all = {}

            for name_base in names_base:
                name_group = name_base.split(' ')[0]
                bases_all[name_base], \
                bases_tags_all[name_base], \
                bases_dems_all[name_base] = self.parser(name_base=name_base,
                                                        name_stage=title_of_stage,
                                                        name_group=name_group,
                                                        lick=lick,
                                                        time=None,
                                                        without_dem=False,  # attention
                                                        time_start=time_start_median,
                                                        time_finish=time_finish_median,
                                                        intellicage=intellicage,
                                                        condition=parser_condition)
            sup_bases_all = []
            for i in bases_tags_all.values():
                for j in i:
                    sup_bases_all.append(j)

            base_all = pd.concat([*bases_all.values()], ignore_index=True)

            tag_list_all = base_all.Tag.unique()
            base_all_ = pd.DataFrame(columns=[f'{title_of_stage}_intervals'])
            for tag in tag_list_all:
                new_df = base_all[base_all.Tag == tag].sort_values(by=['_StartTime'])
                new_df['shift_EndTime'] = new_df['_EndTime'].shift()
                diff = list((new_df['_StartTime'] - new_df['shift_EndTime']).dt.seconds)[1:]
                if len(diff) == 0:
                    base_all_.loc[tag, f'{title_of_stage}_intervals'] = replacing_value
                else:
                    base_all_.loc[tag, f'{title_of_stage}_intervals'] = diff

            base_all = base_all_

            for tag in sup_bases_all:
                if tag not in base_all.index:
                    base_all.loc[tag] = replacing_value
            if replacing_value is not None:
                base_all = base_all.fillna(
                    value=replacing_value)  # situation len(new_df) == 1 or == 0

            base_all = base_all[f'{title_of_stage}_intervals']

        # ------- after special computing space
        all_dems = []
        for i in bases_dems.values():
            if len(i) > 0 and i not in all_dems:
                all_dems.append(i[0])

        sup_bases = []
        for i in bases_tags.values():
            for j in i:
                sup_bases.append(j)

        base = pd.concat([*bases.values()], ignore_index=True)

        tag_list = base.Tag.unique()
        base_ = pd.DataFrame(columns=[f'{title_of_stage}_intervals'])
        for tag in tag_list:
            new_df = base[base.Tag == tag].sort_values(by=['_StartTime'])
            new_df['shift_EndTime'] = new_df['_EndTime'].shift()
            diff = list((new_df['_StartTime'] - new_df['shift_EndTime']).dt.seconds)[1:]
            if len(diff) == 0:
                base_.loc[tag, f'{title_of_stage}_intervals'] = replacing_value
            else:
                base_.loc[tag, f'{title_of_stage}_intervals'] = diff

            self.to_log(f'{tag} : {diff}')

        base = base_

        for tag in sup_bases:
            if tag not in base.index:
                base.loc[tag] = replacing_value
        if replacing_value is not None:
            base = base.fillna(
                value=replacing_value)  # situation len(new_df) == 1 or == 0

        base = base[f'{title_of_stage}_intervals']

        net = Network('1000px', '1000px')

        list_with_dicts = {}

        for k, v in bases.items():
            tags = bases_tags[k]

            time_interval_list = []
            for tag in tags:
                if median_special_time:
                    i = base_all[tag]
                else:
                    i = base[tag]
                if type(i) is int:
                    if i == 0:
                        if not delete_zero_in_intervals_for_median:
                            time_interval_list.append(i)
                    else:
                        time_interval_list.append(i)
                else:
                    for j in i:
                        if j == 0:
                            if not delete_zero_in_intervals_for_median:
                                time_interval_list.append(j)
                        else:
                            time_interval_list.append(j)
            time_interval = np.median(time_interval_list)
            self.log['medians'][f'{k} ({title_of_stage})'] = time_interval

            self.to_log(f'Median in {k} = {time_interval}')

            v.sort_values(by=['_StartTime'], inplace=True)

            if input_time_interval == 'auto':
                graph_dict = self.graph(v,
                                        bases_tags[k],
                                        bases_dems[k],
                                        net,
                                        time_interval=time_interval,
                                        dynamic=dynamic,
                                        special_name=title_of_stage + '_' + k)
            else:
                graph_dict = self.graph(v,
                                        bases_tags[k],
                                        bases_dems[k],
                                        net,
                                        time_interval=input_time_interval,
                                        dynamic=dynamic,
                                        special_name=title_of_stage + '_' + k)

            list_with_dicts[k] = graph_dict

        if not dynamic:
            net.show_buttons(filter_=['physics',
                                      # 'nodes'
                                      ])
            net.save_graph(f'{self.output_path}\\{title_of_stage}_graph.html')

        self.to_log('\n------- Rat tags in each group -------')
        for k, v in bases_tags.items():
            self.to_log(f'{k, v}')
        self.to_log(base)

        return base, list_with_dicts, all_dems


    def eda_graph(self,
                  lick: bool = None,
                  time: float = None,
                  replacing_value='auto',
                  without_dem_base: bool = False,
                  input_time_interval='auto',
                  time_start: str = None,
                  time_finish: str = None,
                  time_start_median: str = None,
                  time_finish_median: str = None,
                  delete_zero_in_intervals_for_median: bool = False,
                  median_special_time: bool = False,
                  dynamic: bool = False,
                  intellicage=None,
                  parser_condition: dict = None):
        """
        ------------ 
        Function:
            This method creates a graph (based on puvis) for each stage (keys in the dictionary 'dict_names'),
            and also prints information about processed files, individuals, etc.
        ------------
        Parameters:
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            replacing_value: float or 'auto'
                Float number of seconds. The number of seconds after each visit during which the visits of followers
                are counted. Default value is 'auto'.
                Example: 12,4 or 30
            without_dem_base: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            input_time_interval: float or 'auto'
                Float number of seconds or flag 'auto'. The number of seconds after each visit during which the
                visits of followers are counted. If 'auto', time interval for observing followers' visits will be
                evaluated automatically. Default value is 'auto'.
                Example: 'auto' or 12.5 or 50                
            time_start: str
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original finish
                time in the file, 'time_finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            time_start_median: str
                Specific start time (format is like time_start) to evaluate median for statistics
                Default is None
            time_finish_median: str
                Specific finish time (format is like time_finish) to evaluate median for statistics
                Default is None
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is False.
                Example: True of False            
            median_special_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is False.
                Example: True or False
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False            
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
        ------------
        Return:
            List with tuples.
                Each tuple with:
                     1) pandas.DataFrame where indexes are rat tags and values are lists with intervals (seconds)
                     between visits;
                     2) list with dictionaries, each dictionary is a relation between rats-initiators and followers;
                     3) list with demonstrator's tags.
                     Example: (pandas.DataFrame, list(dict1(),dict2(),dict3()), list)
                Example: [(frame1, dict1, list1),
                          (frame2, dict2, list2)]
        """

        self.to_log('\n======= Work with graphs (eda_graph) =======')

        if replacing_value == 'auto':
            if time is None:
                if time_start is not None and time_finish is not None:
                    if len(time_start) > 8 or len(time_finish) > 8:
                        time_start_for_replacing_value = pd.to_datetime(time_start, format='%Y-%m-%d_%H:%M:%S')
                        time_finish_for_replacing_value = pd.to_datetime(time_finish, format='%Y-%m-%d_%H:%M:%S')
                    else:
                        time_start_for_replacing_value = pd.to_datetime(time_start, format='%H:%M:%S')
                        time_finish_for_replacing_value = pd.to_datetime(time_finish, format='%H:%M:%S')
                    replacing_value = (time_finish_for_replacing_value - time_start_for_replacing_value).seconds
                else:
                    replacing_value = 24 * 60 * 60  # whole day
            else:
                replacing_value = time * 60

        output_list = []
        for title, names_base in self.dict_names.items():
            output_list.append(self.work(title_of_stage=title,
                                         names_of_files=names_base,
                                         lick=lick,
                                         time=time,
                                         without_dem_base=without_dem_base,
                                         dynamic=dynamic,
                                         input_time_interval=input_time_interval,
                                         time_start=time_start,
                                         time_finish=time_finish,
                                         time_start_median=time_start_median,
                                         time_finish_median=time_finish_median,
                                         delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                         median_special_time=median_special_time,
                                         replacing_value=replacing_value,
                                         intellicage=intellicage,
                                         parser_condition=parser_condition
                                         )
                               )
        return output_list


    @staticmethod
    def graph_plotly(data: dict,
                     dem_tags: list,
                     coords='default',
                     special_name: str = '_'):
        """
        ------------
        Function:
            This method makes an interaction graph via plotly.graph_objects.Scatter.
        ------------
        Parameters:
            data: dict   
                Dictionary with the interaction data. Keys are rat tags of followers, values are pandas.DataFrames where
                indexes are rat tags of followers and frame values are numbers of visits after initiator.                
                Example: {123:pandas.DataFrame({456:[2],789:[4]}),
                          111:pandas.DataFrame({222:[1],333:[0]})}
            dem_tags: list
                List with demonstrators' tags.
                Example: [123, 456, 789]
            coords: list with tuples
                Coordinates to draw nodes as rats in the cage. Note pairs of coordinates should be no less than rats.
                There is no maximum pairs, tags are assigned simply in order to each pair. Default value is 'default'.
                Example: [(3, 5), (4, 1), (6, 1), (7, 5), (5, 8)]
            special_name: str
                Custom string identifier for necessary rat group.
                This name is used in column naming during data processing.
                Example: 'visits in stage1 group1'
                Default: '_'
        ------------
        Return:
            List with trace with nodes and traces with edges for plotly.graph_objects.Scatter.
            Example: [node_trace,*edge_traces]
            or
            plotly.graph_objects.Scatter with the label 'No data on drinking bowls'.
        """

        name_column = f'Visits in {special_name}' # customizable

        if coords == 'default':
            coords = [(3, 5), (4, 1), (6, 1), (7, 5), (5, 8)]

        sup_dict_both = data
        oper = 0
        nodes = {}
        color_list = []

        if sup_dict_both is not None:
            for tag in sup_dict_both.keys():
                if tag in dem_tags:
                    # idx = copy.deepcopy(oper)
                    color_list.append('yellow')
                nodes[tag] = coords[oper]
                color_list.append('red')
                oper += 1

            edges = []

            for tag, ties in sup_dict_both.items():
                for i in ties.index:
                    val = ties.loc[i, name_column]
                    if val != 0:
                        edge = (tag, i, val)
                        edges.append(edge)

            edge_traces = []

            for i in edges:
                influencer = nodes[i[0]]
                dependent = nodes[i[1]]
                edge_trace = go.Scatter(x=[influencer[0], dependent[0]],
                                        y=[influencer[1], dependent[1]],
                                        line=dict(width=5 * i[2], color='blue'),
                                        hoverinfo=None,
                                        mode='lines',
                                        showlegend=False)
                edge_traces.append(edge_trace)

            node_trace = go.Scatter(x=[x[0] for x in nodes.values()],
                                    y=[x[1] for x in nodes.values()],
                                    mode='markers+text',
                                    hoverinfo='all',
                                    hovertext=[f'{i[0]}: {i[1]}-->{i[2][0]}' for i in
                                               [(k, x.index.values, x.T.values) for k, x in sup_dict_both.items()]],
                                    marker=dict(showscale=False,
                                                reversescale=False,
                                                color=color_list,
                                                size=50),
                                    text=[x for x in nodes.keys()],
                                    textposition='bottom center',
                                    showlegend=False)

            # go.Figure(data=[node_trace,*edge_traces])
            return [node_trace, *edge_traces]
        else:
            return [go.Scatter(x=[5],
                               y=[5],
                               mode='markers+text',
                               marker=dict(size=100),
                               text='No data on drinking bowls')]


    def dynamic_graph(self,
                      name: str,
                      dict_data: dict):
        """        
        ------------
        Function:
            This method makes a dynamic graph via the set of input slides in form of dictionary.
        ------------
        Parameters:
            name: str
                Name of the file where the visit data is located. The first word must be the name of a rat group and
                after that it must be space. 
                Example: 'control1 session2' or 'name_group session1 stage 2'
            dict_data: dict
                Dictionary where keys are time (format 'HH:MM') and values are returns from the 'graph_plotly' method.
                Example: {'12:30': graph_plotly()}
        ------------
        Return:
              None
        ------------
        Side effect:
            html-file with a dynamic graph.
        """

        fig_dict = {"data": [],
                    "layout": {},
                    "frames": []}

        fig_dict["layout"]["xaxis"] = {'range': [1, 9],
                                       'autorange': False,
                                       'showgrid': False,
                                       'zeroline': False,
                                       'visible': False}
        fig_dict["layout"]["yaxis"] = {'range': [-1, 10],
                                       'autorange': False,
                                       'showgrid': False,
                                       'zeroline': False,
                                       'visible': False}
        fig_dict["layout"]['title'] = f"{name}"
        fig_dict["layout"]["hovermode"] = "closest"

        sliders_dict = {"active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {"font": {"size": 20},
                                         "prefix": "Time: ",
                                         "visible": True,
                                         "xanchor": "right"},
                        "transition": {"duration": 300,
                                       "easing": "cubic-in-out"},
                        "pad": {"b": 10,
                                "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": []}

        trace_list = []
        for i in list(dict_data.values()):
            for j in i:
                trace_list.append(j)

        fig_dict['data'] = trace_list

        steps = []
        i = 0

        times = list(dict_data.keys())
        oper_times = 0

        while i < len(trace_list):
            step = {'method': 'update',
                    'args': [{"visible": [False] * len(trace_list)}, ],
                    'label': str(times[oper_times]),
                    'value': str(times[oper_times])}
            inner_oper = 0
            for j in range(len(trace_list)):
                if i < len(trace_list):
                    if trace_list[i]['mode'] == 'markers+text':
                        if inner_oper == 0:
                            step['args'][0]['visible'][i] = True
                            trace_list[i]['showlegend'] = False
                            i += 1
                            inner_oper += 1
                        elif inner_oper != 0:
                            break
                    else:
                        step['args'][0]['visible'][i] = True
                        trace_list[i]['showlegend'] = False
                        i += 1
                else:
                    break

            steps.append(step)
            oper_times += 1

        sliders_dict['steps'] = steps
        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)
        fig.write_html(f"{self.output_path}\\dg_{name}.html")


    def dynamic_graphs(self,
                       start: str,
                       finish: str,
                       slide: float,
                       step: float,
                       time_start_median: str = None,
                       time_finish_median: str = None,
                       lick: bool = None,
                       time: float = None,
                       replacing_value='auto',
                       without_dem_base: bool = False,
                       input_time_interval='auto',
                       delete_zero_in_intervals_for_median: bool = True,
                       median_special_time: bool = True,
                       dynamic: bool = False,
                       intellicage=None,
                       parser_condition: dict = None,
                       division_coef: float = None):
        """
        ------------
        Function:
            Processing the visit data, slicing, labeling by time and providing the data to the 'dynamic_graph' method to
            make graphs with time slider.
        ------------
        Parameters:
            start: str
                Particular start time to download the data (if your 'start' is less than the original start time 
                in the file, 'start' will equal to the original start time).
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time).
                Example: '12:30:45' or '2022-01-01_00:00:00'
            slide: float
                Slice (minute time slide) to make one frame for dynamic graph.
                Example: 30 or 35.5
            step: float
                Step (minutes) to move time slide of the graph.
                Example: 10 or 15.5
            time_start_median: str
                Specific start time (format is like time_start) to evaluate median for statistics
                Default is None
            time_finish_median: str
                Specific finish time (format is like time_finish) to evaluate median for statistics
                Default is None
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            replacing_value: float or 'auto'
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted.
                Default value is 'auto'.
                Example: 12,4 or 30
            without_dem_base: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            input_time_interval: float or 'auto'
                Float number of seconds or flag 'auto'. The number of seconds after each visit during which the visits
                of followers are counted. If 'auto', time interval for observing followers' visits will be evaluated
                automatically. Default value is 'auto'.
                Example: 'auto' or 12.5 or 50
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False
            median_special_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False 
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
            division_coef: float
                Float number to divide (normalize) metrics (node susceptibility, credibility, graph_weight).
                Default value is None.
                Example: 48 or 15.5
        ------------
        Return:
            List with file names and dictionary about graphs (where keys are time and 
            values are dictionaries (keys are file names and values are returns from 'graph_plotly' method)).
                Example: [names, dynamic_graph_dict]
        ------------
        Side effect:
            html-files with graphs.
        """

        if len(start) > 8 or len(finish) > 8:
            start = pd.to_datetime(start, format='%Y-%m-%d_%H:%M:%S')
            end_of_all = pd.to_datetime(finish, format='%Y-%m-%d_%H:%M:%S')
        else:
            start = pd.to_datetime(start, format='%H:%M:%S')
            end_of_all = pd.to_datetime(finish, format='%H:%M:%S')

        slide = pd.offsets.DateOffset(minutes=slide)
        step = pd.offsets.DateOffset(minutes=step)

        new_start = start
        new_finish = start + slide

        time_dict = {}
        while new_start <= end_of_all - slide:

            str_new_start = str(new_start).replace(' ', '_')
            str_new_finish = str(new_finish).replace(' ', '_')

            if str_new_start[:4] == '1900' or str_new_finish[:4] == '1900':
                str_new_start = str_new_start[-8:]
                str_new_finish = str_new_finish[-8:]

            sup_args = dict(lick=lick,
                            time=time,
                            replacing_value=replacing_value,
                            without_dem_base=without_dem_base,
                            input_time_interval=input_time_interval,
                            time_start=str_new_start,
                            time_finish=str_new_finish,
                            time_start_median=time_start_median,
                            time_finish_median=time_finish_median,
                            delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                            median_special_time=median_special_time,
                            dynamic=dynamic,
                            intellicage=intellicage,
                            parser_condition=parser_condition)

            if division_coef:
                data, _ = self.graph_analysis(division_coef=division_coef,
                                              **sup_args)
            else:
                data = self.eda_graph(**sup_args)

            time_dict[str_new_finish] = data

            new_start += step
            new_finish += step

        final_time_dict = {}
        names = []
        dems = []

        for time, data in time_dict.items():
            sup_dict = {}

            stage_names = self.dict_names.keys()
            data_for_graph_zipped = list(zip(stage_names, data))

            for stage_name, cortege in data_for_graph_zipped:
                dems += cortege[2]

                for name, dict_relations in cortege[1].items():
                    sup_name = stage_name + '_' + name
                    if sup_name not in names:
                        names.append(sup_name)
                    sup_dict[sup_name] = self.graph_plotly(dict_relations,
                                                           cortege[2],
                                                           special_name=sup_name)
            final_time_dict[time] = sup_dict

        for name in names:
            new_sup_dict = {}
            for time, sup_dict in final_time_dict.items():
                for inner_name, inner_graph in sup_dict.items():
                    if inner_name == name:
                        new_sup_dict[time] = inner_graph
            self.dynamic_graph(name, new_sup_dict)

        return [names, final_time_dict]


    def graph_analysis(self,
                       division_coef: float = None,
                       lick: bool = None,
                       time: float = None,
                       replacing_value='auto',
                       without_dem_base: bool = False,
                       input_time_interval='auto',
                       time_start: str = None,
                       time_finish: str = None,
                       time_start_median: str = None,
                       time_finish_median: str = None,
                       delete_zero_in_intervals_for_median: bool = True,
                       median_special_time: bool = True,
                       dynamic: bool = False,
                       intellicage=None,
                       parser_condition: dict = None):
        """
        ------------
        Function:
            Calculating graph statistics. Metrics: eccentricity, diameter, radius, periphery, center, node
            credibility, node susceptibilities, graph weight. Additionally, it prints median for graph analysis.
        ------------
        Parameters:
            division_coef: float
                Float number to divide (normalize) metrics (node susceptibility, credibility, graph_weight).
                Default value is None.
                Example: 48 or 15.5
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            replacing_value: float or 'auto'
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is 'auto'.
                Example: 12,4 or 30
            without_dem_base: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            input_time_interval: float or 'auto'
                Float number of seconds or flag 'auto'. The number of seconds after each visit during which the visits
                of followers are counted. If 'auto', time interval for observing followers' visits will be evaluated
                automatically. Default value is 'auto'.
                Example: 'auto' or 12.5 or 50
            time_start: str
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original finish
                time in the file, 'time_finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            time_start_median: str
                Specific start time (format is like time_start) to evaluate median for statistics
                Default is None
            time_finish_median: str
                Specific finish time (format is like time_finish) to evaluate median for statistics
                Default is None
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False            
            median_special_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False            
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is false.
                Example: True or False            
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
        ------------
        Return:
            Dictionary, where keys are names of metrics and values are metrics.
        ------------
        Side effect:
            Making graph and printing metrics.
        """

        data_for_graph = self.eda_graph(lick=lick,
                                        time=time,
                                        replacing_value=replacing_value,
                                        without_dem_base=without_dem_base,
                                        input_time_interval=input_time_interval,
                                        time_start=time_start,
                                        time_finish=time_finish,
                                        time_start_median=time_start_median,
                                        time_finish_median=time_finish_median,
                                        delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                        median_special_time=median_special_time,
                                        dynamic=dynamic,
                                        intellicage=intellicage,
                                        parser_condition=parser_condition)

        stage_names = self.dict_names.keys()
        data_for_graph_zipped = list(zip(stage_names, data_for_graph))

        for stage_name, cortege in data_for_graph_zipped:
            for name_file, graph_dict in cortege[1].items():

                self.log['graph_analysis']['Time start'].append(time_start)
                self.log['graph_analysis']['Time finish'].append(time_finish)
                self.log['graph_analysis']['Stage name'].append(stage_name)
                self.log['graph_analysis']['Name file'].append(name_file)

                median = self.log['medians'][f'{name_file} ({stage_name})']
                self.log['graph_analysis']['Median for graph analysis'].append(median)
                self.log['graph_analysis']['Input replacing value'].append(input_time_interval)

                edges_list = []
                node_credibilities = {}
                node_susceptibilities = {}

                if graph_dict is not None:

                    self.log['graph_analysis']['Data available'].append(True)
                    self.log['graph_analysis']['Unique tags (sample)'].append(list(graph_dict))

                    sup_dem_list = []
                    for key_tag, v in graph_dict.items():
                        for index_tag, relations in list(zip(v.index, v.values)):
                            if relations[0] != 0:
                                for tag in [index_tag] * relations[0]:
                                    edges_list.append((key_tag, tag))
                            if index_tag in node_susceptibilities:
                                node_susceptibilities[index_tag] += relations[0]
                            else:
                                node_susceptibilities[index_tag] = relations[0]

                        node_credibilities[key_tag] = v.values.sum()

                        if key_tag in self.log['demonstrators'][name_file]:
                            sup_dem_list.append((key_tag, node_credibilities[key_tag]))

                    self.log['graph_analysis']['Edges from dem'].append(sup_dem_list)

                    for tag in node_susceptibilities.keys():
                        node_susceptibilities[
                            tag] = f'count = {node_susceptibilities[tag]}, coef = {round(node_susceptibilities[tag] / division_coef, 3)}'

                    graph_weight = 0
                    for tag in node_credibilities.keys():
                        graph_weight += node_credibilities[tag]

                    for tag in node_credibilities.keys():
                        node_credibilities[
                            tag] = f'count = {node_credibilities[tag]}, coef = {round(node_credibilities[tag] / division_coef, 3)}'

                    g = nx.Graph()
                    g.add_edges_from(edges_list)

                    self.to_log(
                        f'\n======= Graph metrics of {name_file} ({stage_name}) ({time_start} - {time_finish}) =======')
                    try:
                        eccentricity = [(x, y) for x, y in nx.eccentricity(g).items()]
                        self.to_log(f'Eccentricity: {eccentricity}')
                        self.log['graph_analysis']['Eccentricity'].append(eccentricity)
                    except:
                        self.to_log('There is no values to evaluate eccentricity.')
                        self.log['graph_analysis']['Eccentricity'].append(None)
                    try:
                        diameter = nx.diameter(g)
                        self.to_log(f'Diameter: {diameter}')
                        self.log['graph_analysis']['Diameter'].append(diameter)
                    except:
                        self.to_log('There is no values to evaluate diameter.')
                        self.log['graph_analysis']['Diameter'].append(None)
                    try:
                        radius = nx.radius(g)
                        self.to_log(f'Radius: {radius}')
                        self.log['graph_analysis']['Radius'].append(radius)
                    except:
                        self.to_log('There is no values to evaluate radius.')
                        self.log['graph_analysis']['Radius'].append(None)
                    try:
                        periphery = list(nx.periphery(g))
                        self.to_log(f'Periphery: {periphery}')
                        self.log['graph_analysis']['Periphery'].append(periphery)
                    except:
                        self.to_log('There is no values to evaluate periphery.')
                        self.log['graph_analysis']['Periphery'].append(None)
                    try:
                        center = list(nx.center(g))
                        self.to_log(f'Center: {center}')
                        self.log['graph_analysis']['Center'].append(center)
                    except:
                        self.to_log('There is no values to evaluate center.')
                        self.log['graph_analysis']['Center'].append(None)
                    try:
                        number_of_edges = len(nx.edges(g))
                        self.to_log(f'Number of edges: {number_of_edges}')
                        self.log['graph_analysis']['Number of edges'].append(number_of_edges)
                    except:
                        self.to_log('There are no edges.')
                        self.log['graph_analysis']['Number of edges'].append(None)

                    self.to_log(f'Node credibility: f{node_credibilities}')
                    self.log['graph_analysis']['Node credibility'].append(node_credibilities)
                    self.to_log(f'Node susceptibilities: {node_susceptibilities}')
                    self.log['graph_analysis']['Node susceptibilities'].append(node_susceptibilities)
                    self.to_log(f'Graph weight: {graph_weight}')
                    self.log['graph_analysis']['Graph weight'].append(graph_weight)

                    self.to_log(f'Median for graph analysis: {median}')
                    self.to_log(f'Input replacing value: {input_time_interval}')

                    number_of_nodes = len(list(graph_dict))
                    self.to_log(f'Number of nodes: {number_of_nodes}')
                    self.log['graph_analysis']['Number of nodes'].append(number_of_nodes)

                    number_of_connected_nodes_list = []
                    for tag_from, tag_to in edges_list:
                        if tag_from not in number_of_connected_nodes_list:
                            number_of_connected_nodes_list.append(tag_from)
                        if tag_to not in number_of_connected_nodes_list:
                            number_of_connected_nodes_list.append(tag_to)

                    if len(number_of_connected_nodes_list) == 0:
                        number_of_connected_nodes = 0
                    else:
                        number_of_connected_nodes = len(number_of_connected_nodes_list)

                    self.to_log(f'Number of connected nodes: {number_of_connected_nodes}')
                    self.log['graph_analysis']['Number of connected nodes'].append(number_of_connected_nodes)

                else:
                    self.to_log(f'\nIn {stage_name} {name_file} there is no information')

                    none_list = ['Eccentricity', 'Diameter', 'Radius', 'Periphery', 'Center', 'Node credibility',
                                 'Node susceptibilities', 'Graph weight', 'Number of nodes',
                                 'Number of connected nodes',
                                 'Number of edges', 'Unique tags (sample)', 'Edges from dem']
                    for key in none_list:
                        self.log['graph_analysis'][key].append(None)
                    self.log['graph_analysis']['Data available'].append(False)

        return data_for_graph, self.log['graph_analysis']


    @staticmethod
    def label_densityhist(ax,
                          n,
                          bins,
                          x=4,
                          y=0.01,
                          r=2):
        """
        ------------
        Function:
            Special labeling for histograms
            (source: https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin).
        ------------
        Parameters:
            ax: Object axe of matplotlib
                The axis to plot.
            n: list, array of int, float
                The values of the histogram bins.
            bins: list, array of int, float
                The edges of the bins.
            x: int, float
                Related the x position of the bin labels. The higher, the lower the value on the x-axis.
                Default: 4
            y: int, float
                Related the y position of the bin labels. The higher, the greater the value on the y-axis.
                Default: 0.01
            r: int
                Number of decimal places.
                Default: 2        
        ------------
        Return:
            Only side effect. Modifying histograms for better labeling.
        """
        k = []

        for i in range(0, len(n)):
            k.append((bins[i + 1] - bins[i]) * n[i])

        k = [round(x, r) for x in k]

        for i in range(0, len(n)):
            x_pos = (bins[i + 1] - bins[i]) / x + bins[i]
            y_pos = n[i] + (n[i] * y)
            label = str(k[i])
            ax.text(x_pos, y_pos, label)


    def eda_count(self,
                  lick: bool = None,
                  time: float = None,
                  verbose: bool = False,
                  without_dem: bool = False,
                  time_start: str = None,
                  time_finish: str = None,
                  intellicage=None,
                  density: bool = True,
                  bins: int = 12,
                  size: tuple = (5, 5),
                  condition: dict = None,
                  parser_condition: dict = None):
        """
        ------------
        Function:
            Method for computing visit statistics on rat groups. 
        ------------
        Parameters:
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False
            without_dem: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            time_start: str
                Particular start time to download the data (if your 'start' is less than the original start time 
                in the file, 'start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            density: bool
                Parameter density for the histogram. Default value is True.
                Example: True or False
            bins: int
                The number of bins in the histogram. Default value is 12.
                Example: 10 or 20
            size: tuple
                Tuple with the size parameters in form of (width, height). Default value is (5,5).
                Example: (10,10)
            condition: dict
                Dictionary where keys are stage names and values are parameters for the 'parser' method
                    (customizable, see in this method code).
                Example: {'stage1': True} ('without_dem' = True)
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
        ------------
        Return:
            Dictionary where keys are stage names and values are pandas.DataFrames 
            (where indexes are rat tags and values are numbers of visits).
            Example: {'stage1': frame1,
                      'stage2': frame2}        
        """

        return_dict = {}

        for stage, file_names in self.dict_names.items():
            bases = {}
            bases_tags = {}
            bases_dem_tags = {}
            tags_in_animal = {}

            # =========================================================== customizable
            if condition is not None and stage in list(condition.keys()):
                final_without_dem = condition[stage]
            else:
                final_without_dem = without_dem
            # ===========================================================

            for name_base in file_names:
                name_group = name_base.split(' ')[0]

                tags_in_animal[name_base] = \
                    self.animal_file[self.animal_file['Group']==name_group]\
                    [self.animal_file['Demonstrator']=='0']['Animal ID'].to_list() if final_without_dem else \
                    self.animal_file[self.animal_file['Group'] == name_group]['Animal ID'].to_list()

                bases[name_base], \
                bases_tags[name_base], \
                bases_dem_tags[name_base] = self.parser(name_base=name_base,
                                                        name_stage=stage,
                                                        name_group=name_group,
                                                        lick=lick,
                                                        time=time,
                                                        without_dem=final_without_dem,
                                                        time_start=time_start,
                                                        time_finish=time_finish,
                                                        intellicage=intellicage,
                                                        condition=parser_condition)

            base = pd.concat([*bases.values()], ignore_index=True)
            base = base.groupby(['Tag']).count()

            for tags in tags_in_animal.values():
                for tag in tags:
                    if tag not in base.index and tag is not None:
                        base.loc[tag] = 0
            base = base['VisitID']

            return_dict[stage] = base

            self.to_log(f'\n======= Rat tags in groups =======')
            for k, v in bases_tags.items():
                self.to_log(f'{k, v}')
            self.to_log(f'\n======= {stage} =======')

            for k, v in bases.items():
                v.index = v['VisitID']
                v = v.groupby(['Tag']).count()
                v['The number of visits of each rat'] = v['VisitID']

                self.to_log(f'\n{k}\n{v["The number of visits of each rat"]}')

            self.to_log(f'\n======= Information on the whole {stage} =======')
            self.to_log(base)

            if verbose:
                fig, ax = plt.subplots(figsize=size)
                n, _bins, _ = ax.hist(base, bins=bins, log=False, density=density)

                if density:
                    self.label_densityhist(ax, n, bins=_bins, x=4, y=0.01, r=2)
                    ax.axes.yaxis.set_ticks([])

                ax.set_title(stage)
                self.to_log(f'\nBin boundaries: {[int(x) for x in list(_bins)]}')

        return return_dict


    def eda_intervals(self,
                      lick: bool = None,
                      replacing_value='auto',
                      verbose: bool = False,
                      verbose_detailed: bool = False,
                      without_dem: bool = False,
                      time: float = None,
                      time_start: str = None,
                      time_finish: str = None,
                      intellicage=None,
                      condition: dict = None,
                      parser_condition: dict = None,
                      density: bool = True,
                      bins: int = 12,
                      size: tuple = (5, 5),
                      hist_range: tuple = (0, 1000),
                      hist_range_detailed: tuple = (0, 1000)):
        """        
        ------------
        Function:
            Method for computing intervals (between visits) statistics on rat groups.
        ------------
        Parameters:
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            replacing_value: float or 'auto'
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is 'auto'.
                Example: 12,4 or 30
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False
            verbose_detailed: bool
                Displaying additional information while the algorithm is running. Default value is False.
                Example: True or False
            without_dem: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            time_start: str
                Particular start time to download the data (if your 'start' is less than the original start time 
                in the file, 'start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            condition: dict
                Dictionary where keys are stage names and values are parameters for the 'parser' method
                    (customizable, see in this method code).
                Example: {'stage1': True} ('without_dem' = True)
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
            density: bool
                Parameter density for the histogram. Default value is True.
                Example: True or False
            bins: int
                The number of bins in the histogram. Default value is 12.
                Example: 10 or 20
            size: tuple
                Tuple with the size parameters in form of (width, height). Default value is (5,5).
                Example: (10,10)
            hist_range: tuple
                The range for the horizontal axis in the histogram.
                Default value is (0,1000)
                Example: (50,100)
            hist_range_detailed: tuple
                The range for the horizontal axis in the additional histogram (for each rat tag).
                Default value is (0,1000)
                Example: (50,100)
        ------------
        Return:
            Dictionary where keys are stage names, and values are pandas.Series with visit intervals for each tag.
        """

        if replacing_value == 'auto':
            if time is None:
                if time_start is not None and time_finish is not None:

                    if len(time_start) > 8 or len(time_finish) > 8:
                        time_start_for_replacing_value = pd.to_datetime(time_start, format='%Y-%m-%d_%H:%M:%S')
                        time_finish_for_replacing_value = pd.to_datetime(time_finish, format='%Y-%m-%d_%H:%M:%S')
                    else:
                        time_start_for_replacing_value = pd.to_datetime(time_start, format='%H:%M:%S')
                        time_finish_for_replacing_value = pd.to_datetime(time_finish, format='%H:%M:%S')

                    replacing_value = (time_finish_for_replacing_value - time_start_for_replacing_value).seconds
                else:
                    replacing_value = 24 * 60 * 60  # whole day
            else:
                replacing_value = time * 60

        return_dict = {}

        for stage, file_names in self.dict_names.items():
            _replacing_value = replacing_value
            bases = {}
            bases_tags = {}
            bases_dem_tags = {}

            # =========================================================== customizable
            if condition is not None and stage in list(condition.keys()):
                final_without_dem = condition[stage]
            else:
                final_without_dem = without_dem

            if condition is not None and 'replacing_value' in list(condition.keys()):
                if stage in list(condition['replacing_value']):
                    _replacing_value = condition['replacing_value'][stage]
            # ===========================================================

            for name_base in file_names:
                name_group = name_base.split(' ')[0]
                bases[name_base], \
                bases_tags[name_base], \
                bases_dem_tags[name_base] = self.parser(name_base=name_base,
                                                        name_stage=stage,
                                                        name_group=name_group,
                                                        lick=lick,
                                                        time=time,
                                                        without_dem=final_without_dem,
                                                        time_start=time_start,
                                                        time_finish=time_finish,
                                                        intellicage=intellicage,
                                                        condition=parser_condition)
            all_dems = []
            for i in bases_dem_tags.values():
                if len(i) > 0 and i not in all_dems:
                    all_dems.append(i[0])

            sup_bases = []
            for i in bases_tags.values():
                for j in i:
                    if j is not None and j not in sup_bases:
                        sup_bases.append(j)

            base = pd.concat([*bases.values()], ignore_index=True)

            tag_list = base.Tag.unique()
            base_ = pd.DataFrame(columns=[f'{stage}_intervals'])
            for tag in tag_list:
                new_df = base[base.Tag == tag].sort_values(by=['_StartTime'])
                new_df['shift_EndTime'] = new_df['_EndTime'].shift()
                diff = list((new_df['_StartTime'] - new_df['shift_EndTime']).dt.seconds)[1:]
                if len(diff) == 0:
                    base_.loc[tag, f'{stage}_intervals'] = _replacing_value
                else:
                    base_.loc[tag, f'{stage}_intervals'] = np.median(diff)

                if verbose_detailed:
                    self.to_log(f'{tag} : {diff}')

                    if len(diff) > 0:
                        fig, ax = plt.subplots()
                        n, _bins, _ = plt.hist(diff,
                                              bins=None,
                                              log=False,
                                              density=density,
                                              range=hist_range_detailed)
                        self.label_densityhist(ax, n, bins=_bins, x=4, y=0.01, r=2)
                        plt.title(f'{tag} : {diff} \nBin boundaries: {[int(x) for x in list(_bins)]}')
                        ax.axes.yaxis.set_ticks([])

            base = base_

            for tag in sup_bases:
                if tag not in base.index:
                    base.loc[tag] = _replacing_value
            if _replacing_value is not None:
                base = base.fillna(value=_replacing_value)  # situation len(new_df) == 1 or == 0

            base = base[f'{stage}_intervals']

            return_dict[stage] = base

            self.to_log(f'\n======= Rat tags in groups =======')
            for k, v in bases_tags.items():
                self.to_log(f'{k, v}')

            self.to_log(f'\n======= Information on the whole {stage} =======')
            self.to_log(base)

            if verbose:
                fig, ax = plt.subplots(figsize=size)
                n, _bins, _ = ax.hist(base,
                                     bins=bins,
                                     log=False,
                                     density=density,
                                     range=hist_range)
                if density:
                    self.label_densityhist(ax, n, bins=_bins, x=4, y=0.01, r=2)
                    ax.axes.yaxis.set_ticks([])

                ax.set_title(stage)
                self.to_log(f'\nBin boundaries: {[int(x) for x in list(_bins)]}')

        return return_dict


    def eda_target(self,
                   target_column_name: str,
                   column_name_for_group: str,
                   group_func = 'sum',
                   lick: bool = None,
                   time: float = None,
                   verbose: bool = False,
                   without_dem: bool = False,
                   time_start: str = None,
                   time_finish: str = None,
                   intellicage=None,
                   condition: dict = None,
                   parser_condition: dict = None,
                   density: bool = True,
                   bins: int = 12,
                   size: tuple = (5, 5),
                   hist_range: tuple = (0, 1000)):
        """
        ------------
        Function:
            Method to analyze specific column in Intellicage data about visits.
        ------------
        Parameters:
            column_name_for_group: str
                The column that the grouping is based on (in pandas.DataFrame from Intellicage data)
                Example: 'column1'
            target_column_name:
                The name of the column that we want to analyze
                Example: 'column2'
            group_func:
                Grouping function, via str pandas notation or any function for dataframes.
                Example: 'mean','sum'
                Default: 'sum'
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False
            without_dem: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            time_start: str
                Particular start time to download the data (if your 'start' is less than the original start time
                in the file, 'start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' 
            time_finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            condition: dict
                Dictionary where keys are stage names and values are parameters for the 'parser' method
                    (customizable, see in this method code).
                Example: {'stage1': True} ('without_dem' = True)
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
            density: bool
                Parameter density for the histogram. Default value is True.
                Example: True or False
            bins: int
                The number of bins in the histogram. Default value is 12.
                Example: 10 or 20
            size: tuple
                Tuple with the size parameters in form of (width, height). Default value is (5,5).
                Example: (10,10)
            hist_range: tuple
                The range for the horizontal axis in the histogram.
                Default value is (0,1000)
                Example: (50,100)
        ------------
        Return:
            Dictionary where keys - stages, values - pandas.DataFrame with needed data
        """

        return_dict = {}

        for stage, file_names in self.dict_names.items():
            bases = {}
            bases_tags = {}
            bases_dem_tags = {}
            tags_in_animal = {}

            # =========================================================== customizable
            if condition is not None and stage in list(condition.keys()):
                final_without_dem = condition[stage]
            else:
                final_without_dem = without_dem
            # ===========================================================

            for name_base in file_names:
                name_group = name_base.split(' ')[0]

                tags_in_animal[name_base] = \
                    self.animal_file[self.animal_file['Group']==name_group]\
                    [self.animal_file['Demonstrator']=='0']['Animal ID'].to_list() if final_without_dem else \
                    self.animal_file[self.animal_file['Group'] == name_group]['Animal ID'].to_list()

                bases[name_base], \
                bases_tags[name_base], \
                bases_dem_tags[name_base] = self.parser(name_base=name_base,
                                                        name_stage=stage,
                                                        name_group=name_group,
                                                        lick=lick,
                                                        time=time,
                                                        without_dem=final_without_dem,
                                                        time_start=time_start,
                                                        time_finish=time_finish,
                                                        intellicage=intellicage,
                                                        condition=parser_condition)

            base = pd.concat([*bases.values()], ignore_index=True)
            base = base.groupby([column_name_for_group]).agg({target_column_name: group_func})

            # for tags in tags_in_animal.values():
            #     for tag in tags:
            #         if tag not in base.index and tag is not None:
            #             base.loc[tag] = 0

            return_dict[stage] = base

            self.to_log(f'\n======= Rat tags in groups =======')
            for k, v in bases_tags.items():
                self.to_log(f'{k, v}')
            self.to_log(f'\n======= Information on the whole {stage} =======')
            self.to_log(base)

            if verbose:
                fig, ax = plt.subplots(figsize=size)
                n, _bins, _ = ax.hist(base.values,
                                     bins=bins,
                                     log=False,
                                     density=density,
                                     range=hist_range)
                if density:
                    self.label_densityhist(ax, n, bins=_bins, x=4, y=0.01, r=2)
                    ax.axes.yaxis.set_ticks([])
                ax.set_title(stage)
                self.to_log(f'\nBin boundaries: {[int(x) for x in list(_bins)]}')

        return return_dict


    @staticmethod
    def series_changer(x: pd.Series,
                       y: pd.Series,
                       value_instead_of_none=None,
                       funk_to_print=print,
                       verbose: bool = True):
        """
        ------------
        Function:
            Method to compare two pandas.Series for statistical testing.
            Adding empty indexes or deleting unnecessary.
        ------------
        Parameters:
            x: pandas.Series
                First series with values.
            y: pandas.Series
                Second series to compare with the first one.
            value_instead_of_none: Any
                Value to add for empty rat tag.
                Example: {'x':None, 'y':2}
                Default is None.
            funk_to_print:
                Function to print or record events.
                Default is print.
            verbose: bool
                Default is True.
        ------------
        Return:
        ------------
        Pay attention:
            If value_instead_of_none is None than the behaviour of this method is deleting.
        """

        _x = x.copy()
        _y = y.copy()

        if value_instead_of_none is None:
            for i in _y.index:
                if i not in _x.index:
                    if verbose:
                        funk_to_print(f'There is no tag {i} in the first selection. Deleting this tag.')
                    _y = _y.loc[_y.index != i]

            for i in _x.index:
                if i not in _y.index:
                    if verbose:
                        funk_to_print(f'There is no tag {i} in the second selection. Deleting this tag.')
                    _x = _x.loc[_x.index != i]
        else:
            for i in _y.index:
                if i not in _x.index:
                    if verbose:
                        funk_to_print(
                            f'There is no tag {i} in the first selection. Adding the tag to the first selection with a {value_instead_of_none}.')
                    _x.loc[i] = value_instead_of_none['x']

            for i in _x.index:
                if i not in _y.index:
                    if verbose:
                        funk_to_print(
                            f'There is no tag {i} in the second selection. Adding the tag to the second selection with a {value_instead_of_none}.')
                    _y.loc[i] = value_instead_of_none['y']

        _x.sort_index(inplace=True)
        _y.sort_index(inplace=True)

        return _x, _y


    def permutation(self,
                    x: pd.Series,
                    y: pd.Series,
                    statistic=np.nanmedian,
                    permutation_type: str = 'independent',
                    n_resamples: int = None,
                    alternative: str = 'two-sided',
                    title: str = None,
                    verbose: bool = False,
                    **kwargs):
        """
        ------------
        Function:
            Method to perform the permutation test.
            See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
        ------------
        Parameters:
            x,y: pandas.Series
                Serieses to compare.
            statistic:
                Statistics to check via permutation test between two groups.
                Default is np.nanmedian.
            permutation_type: str
                Interaction type between two samples.
                Default is 'independent'.
            n_resamples: int
                Number of permutation.
                If None than evaluate automatically.
                Default is None.
            alternative: str
                Statistical alternative.
                Default is 'two-sided'.
            title: str
                Title for printing.
                Default is None.
            verbose: bool
                Printing.
                Default is False.
            **kwargs:
                Remaining arguments for series_changer. See series_changer method.
        ------------
        Return:
            Tuple with transformed x,y and result (return by scipy.stats.permutation_test)
        ------------
        Pay attention:
            Maximum two samples can be compared.
        """

        _x, _y = x.copy(), y.copy()
        if permutation_type == 'samples' or permutation_type == 'pairing':
            _x, _y = self.series_changer(x,y,verbose=verbose,**kwargs)
        if n_resamples is None:
            if permutation_type == 'independent':
                # https://en.wikipedia.org/wiki/Binomial_coefficient
                n = len(x) + len(y)
                k = len(x)
                n_resamples = factorial(n)/(factorial(k)*factorial(n-k))
            elif permutation_type == 'samples':
                n_resamples = factorial(2)**len(_x)   # only two samples
            else:
                n_resamples = factorial(len(_x))

        result = stats.permutation_test(data=(_x, _y),
                                  statistic=statistic,
                                  permutation_type=permutation_type,
                                  n_resamples=n_resamples,
                                  alternative=alternative)

        rec = f"{title}: {result}"
        if verbose:
            print(rec)
        self.to_log(rec)

        return _x, _y, result


    def wilcoxon(self,
                 x: pd.Series,
                 y: pd.Series,
                 alternative: str = 'two-sided',
                 zero_method: str = 'zsplit',
                 title: str = None,
                 verbose: bool = False,
                 **kwargs):
        """
        ------------
        Function:
            Method to perform the wilcoxon test.
            See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        ------------
        Parameters:
            x,y: pandas.Series
                Serieses to compare.
            alternative: str
                Statistical alternative.
                Default is 'two-sided'.
            zero_method: str
                Method to handle with zero data in series.
                Default is 'zsplit'.
            title: str
                Title for printing.
                Default is None.
            verbose: bool
                Printing.
                Default is False.
            **kwargs:
                Remaining arguments for series_changer. See series_changer method.
        ------------
        Return:
            Tuple with transformed x,y and result (return by scipy.stats.wilcoxon)
        ------------
        Pay attention:
            Maximum two samples can be compared.
        """

        _x, _y = self.series_changer(x,y,**kwargs)

        rec1 = f'\n{title}'
        rec2 = f'\nMethod - {zero_method}, alternative - {alternative}'
        result = stats.wilcoxon(_x, _y, alternative=alternative, zero_method=zero_method)
        if verbose:
            print(rec1)
            print(rec2)
            print(result)
        self.to_log(rec1)
        self.to_log(rec2)
        self.to_log(result)

        return _x, _y, result


    def timeline(self,
                 metric: str,
                 start: str,
                 finish: str,
                 slide: float,
                 step: float,
                 time_start_median: str = None,
                 time_finish_median: str = None,
                 lick: bool = None,
                 time: float = None,
                 replacing_value='auto',
                 without_dem_base: bool = False,
                 input_time_interval='auto',
                 delete_zero_in_intervals_for_median: bool = True,
                 median_special_time: bool = True,
                 dynamic: bool = True,
                 intellicage=None,
                 parser_condition: dict = None,
                 division_coef: float = 1,
                 plotly_verbose: bool = True,
                 output_file: str = 'html'):
        """
        ------------
        Function:
            Form some useful metrics of rat behaviour.
            Metrics:
            gcc (giant cluster component)
            dem_power (number of the edges from a dem)
            visit_density (gantt chart on visits time)
            free_bowls_time (gantt chart on non-visit time)
        ------------
        Parameters:
            metric: str
                Values: 'gcc','dem_power','visit_density','free_bowls_time'
            start: str
                Particular start time to download the data (if your 'start' is less than the original start time
                in the file, 'start' will equal to the original start time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00'
            finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45' or '2022-01-01_00:00:00' or '2022-01-01_00:00:00'
            slide: float
                Slice (minute time slide) to make one frame for dynamic graph.
                Example: 30 or 35.5
            step: float
                Step (minutes) to move time slide of the graph (used for gcc and dem_power in dynamic charts).
                Example: 10 or 15.5
            time_start_median: str
                Specific start time (format is like start) to evaluate median for statistics
                Default is None.
            time_finish_median: str
                Specific finish time (format is like finish) to evaluate median for statistics
                Default is None.
            lick: bool or None
                True - gather only data where a rat perform licks.
                False - take only data without drinking.
                None - all data upload regardless drinking.
                Default is None.
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            replacing_value: float or 'auto'
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is 'auto'.
                Example: 12,4 or 30
            without_dem_base: bool
                Removing rows where demonstrator is. Default value is False.
                Example: True or False
            input_time_interval: float or 'auto'
                Float number of seconds or flag 'auto'. The number of seconds after each visit during which the visits
                of followers are counted. If 'auto', time interval for observing followers' visits will be evaluated
                automatically. Default value is 'auto'.
                Example: 'auto' or 12.5 or 50
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False
            median_special_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is false.
                Example: True or False
            intellicage: bool or 'all_time'
                See 'illumination' parameter in intellicage_parser signature.
                Example: True, False or 'all_time'
            parser_condition: dict
                It is dictionary where you put parser's parameters (See the 'parser' method).
                Default value is None.
                Example: {'illumination':'all_time',
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
                This parameter is fully customizable and can be hardcoded as you wish, see 'customizable' in this class
                    or parser's code.
            division_coef: float
                Float number to divide (normalize) metrics (node susceptibility, credibility, graph_weight).
                Default value is None.
                Example: 48 or 15.5
            plotly_verbose: bool
                Visualizing all graphs in notebook.
                Default is True
            output_file: str
                Format of the output file. See iplot method for plotly. Can be 'png','html' and so on.
                Default is 'html'.
        ------------
        Return:
            None
        ------------
        Side effect:
            Files with charts.
        """

        if metric == 'gcc' or metric == 'dem_power':
            _ = self.dynamic_graphs(start=start,
                                    finish=finish,
                                    time_start_median=time_start_median,
                                    time_finish_median=time_finish_median,
                                    slide=slide,
                                    step=step,
                                    lick=lick,
                                    time=time,
                                    replacing_value=replacing_value,
                                    without_dem_base=without_dem_base,
                                    input_time_interval=input_time_interval,
                                    delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                    median_special_time=median_special_time,
                                    dynamic=dynamic,
                                    intellicage=intellicage,
                                    parser_condition=parser_condition,
                                    division_coef=division_coef)

            data = pd.DataFrame(self.log['graph_analysis'])

            for stage, file_names in self.dict_names.items():
                for file_name in file_names:
                    data_for_plot = data.loc[(data['Stage name'] == stage) & (
                            data['Name file'] == file_name)]

                    all_tags = list(set(list(self.animal_file.loc[
                                                 (self.animal_file['Group'] == file_name.split(' ')[0]),
                                                 'Animal ID'])))

                    fig = go.Figure(layout=go.Layout(
                        title=go.layout.Title(text=f'Metric: {metric}. Stage: {stage}. File: {file_name}')))

                    if metric == 'gcc':
                        fig.add_trace(go.Scatter(x=data_for_plot['Time finish'],
                                                 y=100 * data_for_plot['Number of connected nodes'] / len(all_tags),
                                                 name='Number of connected nodes'))
                        if plotly_verbose:
                            fig.show()

                        if output_file == 'html':
                            fig.write_html(f"{self.output_path}\\{metric}-{stage}-{file_name}.html")
                        else:
                            iplot(fig,
                                  validate=False,
                                  filename=f"{metric}-{stage}-{file_name}",
                                  image_width=1000,
                                  image_height=600,
                                  image=output_file)

                    elif metric == 'dem_power':
                        for dem in self.log['demonstrators'][file_name]:
                            sup_plot_array = []
                            for list_ in list(data['Edges from dem']):
                                if list_ is not None:
                                    for cortege in list_:
                                        if cortege[0] == dem:
                                            sup_plot_array.append(cortege[1])
                                else:
                                    sup_plot_array.append(0)

                            fig.add_trace(go.Scatter(x=data_for_plot['Time finish'],
                                                     y=np.array(sup_plot_array) / len(all_tags),
                                                     name=f'Number of edges from {dem}'))
                        if plotly_verbose:
                            fig.show()

                        if output_file == 'html':
                            fig.write_html(f"{self.output_path}\\{metric}-{stage}-{file_name}.html")
                        else:
                            iplot(fig,
                                  validate=False,
                                  filename=f"{metric}-{stage}-{file_name}",
                                  image_width=1000,
                                  image_height=600,
                                  image=output_file)

        elif metric == 'visit_density':
            for stage, file_names in self.dict_names.items():
                for file_name in file_names:
                    data_from_parser = self.parser(name_base=file_name,
                                                   name_stage=stage,
                                                   name_group=file_name.split(' ')[0],
                                                   lick=lick,
                                                   time=time,
                                                   without_dem=without_dem_base,
                                                   time_start=start,
                                                   time_finish=finish,
                                                   intellicage=intellicage,
                                                   condition=parser_condition)

                    all_tags = list(set(list(self.animal_file.loc[
                                                 (self.animal_file['Group'] == file_name.split(' ')[0]),
                                                 'Animal ID'])))

                    unique_tags = list(data_from_parser[0]['Tag'].unique())
                    start_of_slice = sorted(list(data_from_parser[0]['_StartTime']))[0]
                    finish_for_empty = pd.Timestamp(start_of_slice) + pd.offsets.DateOffset(microseconds=1)

                    for tag in all_tags:
                        for corner in self.corners:
                            if tag not in unique_tags:
                                data_from_parser[0].loc[f'{tag}_({corner})',
                                                        ['Tag',
                                                         'Corner',
                                                         '_StartTime',
                                                         '_EndTime']] = [tag,
                                                                         corner,
                                                                         start_of_slice,
                                                                         finish_for_empty]
                            elif corner not in list(
                                    data_from_parser[0][data_from_parser[0]['Tag'] == tag]['Corner'].unique()):
                                data_from_parser[0].loc[f'{tag}_({corner})',
                                                        ['Tag',
                                                         'Corner',
                                                         '_StartTime',
                                                         '_EndTime']] = [tag,
                                                                         corner,
                                                                         start_of_slice,
                                                                         finish_for_empty]

                    data_from_parser[0]['Tag_Corner'] = data_from_parser[0]['Tag']. \
                                                            apply(lambda x: str(int(x)) \
                        if x not in self.log['demonstrators'][file_name] else str(int(x))+' (dem) '
                                                                  ) + \
                                                        ' (' + data_from_parser[0]['Corner']. \
                                                            apply(lambda x: str(int(x))) + \
                                                        ')'

                    sup_list_for_plot = []
                    for index, row in data_from_parser[0].iterrows():
                        sup_list_for_plot.append(dict(Visit='',
                                                      Start=row['_StartTime'],
                                                      Finish=row['_EndTime'],
                                                      Tag_Corner=row['Tag_Corner']))

                    frame_for_plot = pd.DataFrame(sup_list_for_plot)
                    frame_for_plot.sort_values('Tag_Corner', inplace=True, ascending=False)

                    first_part = frame_for_plot.loc[~frame_for_plot['Tag_Corner'].str.contains('dem')]
                    second_part = frame_for_plot.loc[frame_for_plot['Tag_Corner'].str.contains('dem')]
                    frame_for_plot = pd.concat([first_part, second_part], ignore_index=True)

                    fig = px.timeline(data_frame=frame_for_plot,
                                      x_start="Start",
                                      x_end="Finish",
                                      y="Tag_Corner",
                                      color='Visit',
                                      opacity=1,
                                      title=f'Metric: {metric}. Stage: {stage}. File: {file_name}')
                    fig.update_layout(xaxis=dict(showgrid=False),
                                      yaxis=dict(showgrid=False))

                    # =========================================================== customizable
                    start_for_plot, finish_for_plot = None, None
                    if parser_condition and stage in parser_condition:
                        for stage_, files_dict in parser_condition.items():
                            if files_dict.get(file_name):
                                condition_str_divided = files_dict[file_name].split('|')
                                if len(condition_str_divided )==3: # стандартное условие с time
                                    start_for_plot = datetime.strptime(condition_str_divided[1],'%Y-%m-%d_%H:%M:%S')
                                    finish_for_plot = datetime.strptime(condition_str_divided[2],'%Y-%m-%d_%H:%M:%S')
                                    break

                    if not start_for_plot and not finish_for_plot:
                        start_for_plot = datetime.strptime(start, '%Y-%m-%d_%H:%M:%S')
                        finish_for_plot = datetime.strptime(finish, '%Y-%m-%d_%H:%M:%S')

                    fig.update_layout(xaxis=dict(
                        fixedrange=True,
                        range=[start_for_plot, finish_for_plot],
                        tickformat='%H:%M'))
                    # ===========================================================

                    fig.update_layout(template='plotly_white')
                    fig.update_traces(marker=dict(color='black'))
                    # fig.update_xaxes(linecolor='black')               # interfered with densitometry
                    # fig.update_yaxes(linecolor='black')

                    if plotly_verbose:
                        fig.show()

                    if output_file == 'html':
                        fig.write_html(f"{self.output_path}\\{metric}-{stage}-{file_name}.html")
                    else:
                        iplot(fig,
                              validate=False,
                              filename=f"{metric}-{stage}-{file_name}",
                              image_width=1000,
                              image_height=600,
                              image=output_file)

        elif metric == 'free_bowls_time':
            for stage, file_names in self.dict_names.items():
                for file_name in file_names:
                    sup_list_for_plot = []
                    data_from_parser = self.parser(name_base=file_name,
                                                   name_stage=stage,
                                                   name_group=file_name.split(' ')[0],
                                                   lick=lick,
                                                   time=time,
                                                   without_dem=without_dem_base,
                                                   time_start=start,
                                                   time_finish=finish,
                                                   intellicage=intellicage,
                                                   condition=parser_condition)
                    for index, row in data_from_parser[0].iterrows():
                        sup_list_for_plot.append(dict(File=f'{stage}-{file_name}',
                                                      Start=row['_StartTime'],
                                                      Finish=row['_EndTime'],
                                                      Visits=''))

                    frame_for_plot = pd.DataFrame(sup_list_for_plot)
                    fig = px.timeline(data_frame=frame_for_plot,
                                      x_start="Start",
                                      x_end="Finish",
                                      y="File",
                                      color='Visits',
                                      opacity=1,
                                      title=f'Metric: {metric}. Stage: {stage}. File: {file_name}'
                                      )

                    fig.update_layout(xaxis=dict(showgrid=False),
                                      yaxis=dict(showgrid=False))

                    # =========================================================== customizable
                    if parser_condition is not None:
                        for name_stage in parser_condition.keys():
                            if stage == name_stage:
                                for name_base in parser_condition[stage]:
                                    if file_name == name_base:
                                        if parser_condition[stage][file_name][:4] == 'time':
                                            condition_sup_list = parser_condition[stage][file_name].split('|')
                                            if len(condition_sup_list[1:]) > 2 and len(condition_sup_list[1:])%2 == 0:
                                                starts = a[1:-1:2]
                                                ends = a[2:-1:2]
                                                sup_time_list = list(zip(starts, ends))

                                                sup_list_for_breaks = []
                                                for slice in sup_time_list:
                                                    sup_list_for_breaks.append(dict(bounds=[slice[0],slice[1]]))
                                                fig.update_xaxes(rangebreaks=sup_list_for_breaks)

                    fig.update_layout(template='plotly_white')
                    fig.update_traces(marker=dict(color='black'))
                    # fig.update_xaxes(linecolor='black')
                    # fig.update_yaxes(linecolor='black')

                    if plotly_verbose:
                        fig.show()

                    if output_file == 'html':
                        fig.write_html(f"{self.output_path}\\{metric}-{stage}-{file_name}.html")
                    else:
                        iplot(fig,
                              validate=False,
                              filename=f"{metric}-{stage}-{file_name}",
                              image_width=1000,
                              image_height=600,
                              image=output_file)