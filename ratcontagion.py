# Copyright 2023 Starkov Ilya. All Rights Reserved.
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

class RatExperiment:
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 name_animal_file: str,
                 dict_names: dict,
                 show_warnings: bool = True):
        """
        ------------ 
        Function:
            Class initialization with definition of the arguments.
        ------------
        Parameters:
            input_path: str
                Path to the folder where are the necessary files.
                You can define zipped or non zipped folders. But if you prefer to work with the Intellicage archives,
                then your folder need to be non zipped.
                Example: r"C:/Users/...input" (the slashes were flipped for this example)
            output_path: str
                Path to the folder where you want to store the output files.
            name_animal_file: str
                Name of your animal file (.html). It must be in the 'input_path'.
                Example: 'Animal' (it must be a html-file)
            dict_names: dict
                Dictionary with lists of file names in the input path.
                Each name in the list is the name of the file where the visit data is located.
                The first word must be the name of a rat group and
                after that it must be space. Example: 'control1 session2' or 'name_group session1 stage 2'.
                It can't be the same group with different stages/condition/cases in the one list. 
                Mistake: {'condition 1':['group1 stage 1', 'group1 stage 2', 'group2 stage 1']}
                Right example: {'condition 1':['group1 stage 1', 'group2 stage 1'],
                                'condition 2':['group1 stage 2']}
            show_warnings: bool
                Show warnings (pandas, plotly etc.) or not. Default value is False.
                Example: True or False
        ------------
        Return:
            Class instance of the experiment. 
        ------------
        Pay attention:
            There are no static or class methods in this class, all methods are instance methods.
            
        """

        self.input_path = input_path
        self.output_path = output_path
        self.name_animal_file = name_animal_file
        self.dict_names = dict_names
        self.dict_with_medians = {}

        if not show_warnings:
            warnings.filterwarnings("ignore")

    @staticmethod
    def date_transformer(df,
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
                    print(f'\nError (func - date_transformer): date = {date}, sup_list = {sup_list}.')
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
                Value of illumination from the Intellicage output archive
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
                           input_path: str,
                           name_base: str,
                           illumination='all_time',
                           condition: dict = None):
        """
        ------------
        Function:
            Function to upload data from the Intellicage output archives (.zip files).
        ------------
        Parameters:
            input_path: str
                Path to the folder where are the necessary files.
                You can define zipped or non zipped folders. But if you prefer to work with the Intellicage archives,
                then your folder need to be non zipped.
                Example: r"C:/Users/...input" (the slashes were flipped for this example)
            name_base: str
                Name of the file to upload data with visits.
            illumination: True or False or 'all_time'
                If it is True, you will upload data only with non-zero illumination values (daylight time).
                If it is False, you will upload data only with zero illumination values (nighttime).
                If it is 'all_time', you will upload all available data from the Intellicage archive.
                Default value is 'all_time'.
            condition: dict
                Dictionary where keys are parameters (triggers, flags etc.), and values are
                changing values for the parameter
                (see customizable part of this method).
                Example: {'session1':'last_day','session2':'last_day','session3':'first_6_hours'}
                You can customize conditions (keys, values and logic).
                Default value is None.
                If you do not want to define any conditions, just do not change the default value.  
        ------------
        Return:
            pandas.DataFrame (with visit data from the Intellicage)
        ------------
        Pay attention:
            This method include tools from PyMICE:
                Loader,
                getVisits,
                getEnvironment
            Please, read more about the Intellicage, and it's output files:
            https://www.tse-systems.com/service/intellicage/
            And PyMICE tools: https://github.com/Neuroinflab/PyMICE
        """

        ml = pm.Loader(f'{input_path}\\{name_base}.zip',
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
                    sup_visits.drop_duplicates(subset=['VisitID'], keep=False, inplace=True)

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
                sup_visits.drop_duplicates(subset=['VisitID'], keep=False, inplace=True)

                sup_dict2 = {}
                for end in end_i:
                    sup_dict2[f'{end}'] = sup_visits.loc[sup_visits['_StartTime'] <= end]
                sup_visits_ = pd.concat([*sup_dict2.values()], ignore_index=True)
                sup_visits_.drop_duplicates(subset=['VisitID'], keep=False, inplace=True)

                if illumination:
                    visits.index = visits['VisitID']
                    output_visits = visits.drop(sup_visits_['VisitID'], axis=0)
                    output_visits.reset_index(drop=True, inplace=True)
                else:
                    output_visits = sup_visits_
            else:
                print(f'\nError in daylight time intervals: start_dark: {start_i}, end_dark: {end_i}.')
        else:
            output_visits = visits
        # =========================================================== customizable
        if condition is not None and len(output_visits) > 0:
            if name_base.split()[-1] in list(condition.keys()):
                for key in list(condition.keys()):
                    if name_base.split()[-1] == key:
                        if condition[key] == 'last_day':
                            last_day = sorted(list(output_visits['StartDate'].unique()))[-1]
                            output_visits = output_visits[output_visits['StartDate'] == last_day]
                        elif condition[key] == 'first_day':
                            first_day = sorted(list(output_visits['StartDate'].unique()))[0]
                            output_visits = output_visits[output_visits['StartDate'] == first_day]
            if 'date' in list(condition.keys()):
                spec_date = condition['date']
                output_visits = output_visits[output_visits['StartDate'] == spec_date]
        # ===========================================================

        return output_visits

    def parser(self,
               name_base: str,
               name_group: str,
               name_animal: str,
               input_path: str,
               without_lik: bool = False,
               only_with_lick: bool = False,
               time: float = None,
               verbose: bool = False,
               without_dem: bool = False,
               time_start: str = None,
               time_finish: str = None,
               intellicage: dict = None):
        """
        ------------
        Function:
            Parsing your html-file (that is built via the IntelliCage) or
            parsing original Intellicage output archives.
        ------------
        Parameters:
            name_base: str
                Name of the file where the visit data is located. The first word must be the name of a rat group and
                after that it must be space. 
                Example: 'control1 session2' or 'name_group session1 stage 2'
            name_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base'.
                Example: 'group1'
            name_animal: str
                Name of the file where animal data is located.
                Example: 'Animal' (it must be a html-file)
            input_path: str
                Path to the folder where are the necessary files.
                You can define zipped or non zipped folders. But if you prefer to work with the Intellicage archives,
                then your folder need to be non zipped.
                Example: r"C:/Users/...input" (the slashes were flipped for this example)
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
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
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original
                finish time
                in the file, 'start_time' will be equal to the original start time). Default value is None.
                Example: '12:30:45'
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
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
        """

        if input_path.endswith('.zip'):
            with zipfile.ZipFile(input_path) as z:
                for inner_file_name in [text_file.filename for text_file in z.infolist()]:
                    if inner_file_name[:len(name_base)] == name_base:
                        if inner_file_name[-4:] == 'html':
                            base = pd.read_html(z.open(f'{name_base}.html'), header=1)[0]
                        else:
                            print(f'\nThere is no file {name_base}.html in directory {input_path}')
                    elif inner_file_name[:len(name_animal)] == name_animal:
                        if inner_file_name[-4:] == 'html':
                            raw_animal = pd.read_html(z.open(f'{name_animal}.html'), header=1)[0]
                        else:
                            print(f'\nThere is no file {name_animal}.html in directory {input_path}')
        else:
            for inner_file_name in (next(os.walk(input_path), (None, None, []))[2]):
                if inner_file_name[:len(name_base)] == name_base:
                    if intellicage is not None:
                        base = self.intellicage_parser(input_path=input_path,
                                                       name_base=name_base,
                                                       illumination=intellicage['illumination'],
                                                       condition=intellicage['condition'])
                    elif inner_file_name[-4:] == 'html':
                        base = pd.read_html(f'{input_path}\\{name_base}.html', header=1)[0]
                    else:
                        print(f'\nThere is no file {name_base} in directory {input_path}')

                elif inner_file_name[:len(name_animal)] == name_animal:
                    if inner_file_name[-4:] == 'html':
                        raw_animal = pd.read_html(f'{input_path}\\{name_animal}.html', header=1)[0]
                    else:
                        print(f'\nThere is no file {name_animal}.html in directory {input_path}')

        if intellicage is None:
            base['_StartDate'] = self.date_transformer(base, 'StartDate')
            base['_EndDate'] = self.date_transformer(base, 'EndDate')

            base['_StartTime'] = pd.to_datetime(base['_StartDate'] + '_' + base['StartTime'],
                                                format='%Y-%m-%d_%H:%M:%S.%f')
            base['_EndTime'] = pd.to_datetime(base['_EndDate'] + '_' + base['EndTime'], format='%Y-%m-%d_%H:%M:%S.%f')

            base['VisitDuration'] = pd.to_numeric(base['VisitDuration'])

        dems = list(set(list(raw_animal.loc[(raw_animal['Protocol'] == 'new') & (raw_animal['Demostrator'] == 1) & (
                raw_animal['Group'] == name_group), 'Animal ID'])))

        dem_in_base = []
        base_list = list(base['Tag'].unique())

        for i in dems:
            if i in base_list:
                dem_in_base.append(i)
                if without_dem:
                    base_list.remove(i)
        if verbose:
            print(f'\nThe demonstrator in {name_base} - {dem_in_base}')

        base.index = base['Tag']

        if without_dem:
            base_out_dem = base.drop(dem_in_base, axis=0)
        else:
            base_out_dem = base

        if time is not None and (time_start is None and time_finish is None):
            start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
            finish = start + pd.offsets.DateOffset(minutes=time)
            base_out_dem = base_out_dem.loc[base_out_dem['_StartTime'] <= finish]
            if verbose:
                print(f'\nStart time of downloadling {name_base} = {start}, the end of download = {finish}.')

        if time_start is not None and time_finish is not None:
            if len(base['StartDate'].unique()) > 1:
                print(f'''\nAttention: number of unique dates in "StartDate" more than one.\
                    \rData with this exception: {name_base}.\
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
                    print(f'\nError during preprocessing of {name_base}:')
                    print(
                        'Parameter "time_start" is less than the real start time of the data. The first date will be used as a time_start.')
                    start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
                if sorted(list(base['_StartTime'].unique()))[-1] < finish:
                    print(f'\nError during preprocessing of {name_base}:')
                    print(
                        'Parameter "time_finish" is bigger than the real finish time of the data. Last date will be used as a time_finish.')
                    finish = pd.Timestamp(sorted(list(base['_StartTime']))[-1])

            base_out_dem = base_out_dem.loc[
                (base_out_dem['_StartTime'] <= finish) & (base_out_dem['_StartTime'] >= start)]

            if time is not None:
                start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
                finish = start + pd.offsets.DateOffset(minutes=time)
                base_out_dem = base_out_dem.loc[base_out_dem['_StartTime'] <= finish]

            if verbose:
                print(f'\nStart time of downloadling {name_base} = {start}, the end of download = {finish}.')
        else:
            if verbose:
                start = pd.Timestamp(sorted(list(base['_StartTime']))[0])
                finish = pd.Timestamp(sorted(list(base['_StartTime']))[-1])
                print(f'\nStart time of downloadling {name_base} = {start}, the end of download = {finish}.')

        if without_lik:
            base_out_dem = base_out_dem[base_out_dem['LickNumber'] == 0]
        if only_with_lick:
            base_out_dem = base_out_dem[base_out_dem['LickNumber'] != 0]

        base_out_dem.sort_values('_StartTime', inplace=True)

        return base_out_dem, base_list, dem_in_base

    @staticmethod
    def inner_analysis(data,
                       data_,
                       tags: list,
                       time_interval: float,
                       name_of_group: str):
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
            name_of_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base'.
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

            name_column = f'Visits in {name_of_group}'

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

    def frame_analysis_for_graph(self,
                                 data,
                                 data_,
                                 tags: list,
                                 dem_tags: list,
                                 time_interval: float,
                                 net,
                                 regime='both',
                                 data2=None,
                                 data_2=None,
                                 dynamic: bool = False,
                                 name_of_group: str = '_'):
        """
        ------------
        Function:
            Analyzing which rat went into the drinking bowl after which rat and add this information to the graph,
            gathering information from all situations/bowls/conditions/cases.
        ------------
        Parameters:
            data: pandas.DataFrame
                Slice of the data where this function will search for needed intervals after each visit.
            data_: pandas.DataFrame
                Slice of the data where this function will search for followers in found intervals.
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
            regime:
                Number of drinking bowls to analyze. Default value is 'both'.
                Example: 1 or 2 or 'both'
            data2: pandas.DataFrame
                Slice of the data where this function will search for needed intervals after each visit. 
                This parameter is used when you have several drinkers, locations, etc. For example, in our case
                there were two drinking bowls, so we need two frames per rat. Default value is None.
            data_2: pandas.DataFrame
                Slice of the data where this function will search for followers in found intervals.
                This parameter is used when you have several drinkers, locations, etc.
                Default value is None.
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False
                Example: True or False
            name_of_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base'. Default value is '_'.
        ------------
        Return:
            Side effect is adding nodes and edges to the input pyvis.network.Network.
            Dictionary where keys are rat tags and each value is a pandas.DataFrame with followers gathered from
            all cases.
        """

        name_column = f'Visits in {name_of_group}'
        sup_dict = self.inner_analysis(data, data_, tags, time_interval, name_of_group)

        if regime != 'both':
            if not dynamic:
                for k, v in sup_dict.items():
                    if k in dem_tags:
                        net.add_node(int(k), label=str(k), color='yellow', title=f'{v}')
                    else:
                        net.add_node(int(k), label=str(k), color='red', title=f'{v}')

                sup_graph_list = []
                for k, v in sup_dict.items():
                    for tag in v.index:
                        if v.loc[tag, name_column] == 0:
                            sup_graph_list.append([k, tag])
                        else:
                            net.add_edge(int(k), int(tag), value=0.01 * int(v.loc[tag, name_column]), color='blue')
                for pair in sup_graph_list:
                    net.add_edge(int(pair[0]), int(pair[1]), hidden=True)
            return sup_dict

        else:
            sup_dict2 = self.inner_analysis(data2, data_2, tags, time_interval, name_of_group)

            sup_dict_both = {}
            for k, v in sup_dict.items():
                if k in sup_dict2.keys():
                    for r, t in sup_dict2.items():
                        if k == r:
                            v = v.reset_index()
                            t = t.reset_index()
                            df = pd.concat([v, t], ignore_index=True)
                            df = df.groupby('index').sum()
                            sup_dict_both[k] = df

                elif k not in sup_dict2.keys():
                    sup_dict_both[k] = v
            for r, t in sup_dict2.items():
                if r not in sup_dict.keys():
                    sup_dict_both[r] = t

            # --------------------------------------------------
            if not dynamic:
                for k, v in sup_dict_both.items():
                    if k in dem_tags:
                        net.add_node(int(k), label=str(k), color='yellow', title=f'{v}')
                    else:
                        net.add_node(int(k), label=str(k), color='red', title=f'{v}')

                sup_graph_list = []
                for k, v in sup_dict_both.items():
                    for tag in v.index:
                        if v.loc[tag, name_column] == 0:
                            sup_graph_list.append([k, tag])
                        else:
                            net.add_edge(int(k), int(tag), value=0.01 * int(v.loc[tag, name_column]), color='blue')
                for pair in sup_graph_list:
                    net.add_edge(int(pair[0]), int(pair[1]), hidden=True)

            return sup_dict_both

    def graph(self,
              input_data,
              tags: list,
              dem_tags: list,
              net,
              corners='both',
              time_interval: float = None,
              dynamic: bool = False,
              name_of_group: str = '_'):
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
            corners:
                Number of drinking bowls (bowls were in corners) to analyse. Default value is 'both'. 
                Example: 1 or 2 or 'both'
            time_interval:
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is None.
                Example: 12,4 or 30
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False
            name_of_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base'. Default value is '_'.
        ------------
        Return:
            Side effect is adding nodes and edges to the input pyvis.network.Network.
            Dictionary where keys are rat tags and each value is a pandas.DataFrame with followers from all cases.
        """

        data = input_data.copy()
        graph_dict = None

        if corners == 'both':
            if len(list(input_data['Corner'].unique())) == 2:
                data1 = data[data['Corner'] == 1]
                data_1 = data1.copy()
                data2 = data[data['Corner'] == 2]
                data_2 = data2.copy()

                graph_dict = self.frame_analysis_for_graph(data1,
                                                           data_1,
                                                           tags,
                                                           dem_tags,
                                                           time_interval,
                                                           net,
                                                           regime='both',
                                                           data2=data2,
                                                           data_2=data_2,
                                                           dynamic=dynamic,
                                                           name_of_group=name_of_group)

            elif 1 in list(input_data['Corner'].unique()) and 2 not in list(input_data['Corner'].unique()):
                print(f'''\nThere is no data on the second drinking bowl in the next slice:\
                \rName of the group: {name_of_group}\
                \rData: {input_data.head()}\
                \rTags: {tags}\
                \nThe first drinking bowl will be selected.''')

                data1 = data[data['Corner'] == 1]
                data_1 = data1.copy()

                graph_dict = self.frame_analysis_for_graph(data1,
                                                           data_1,
                                                           tags,
                                                           dem_tags,
                                                           time_interval,
                                                           net,
                                                           dynamic=dynamic,
                                                           name_of_group=name_of_group)

            elif 1 not in list(input_data['Corner'].unique()) and 2 in list(input_data['Corner'].unique()):
                print(f'''\nThere is no data on the first drinking bowl in the next slice:\
                \rName of the group: {name_of_group}\
                \rData: {input_data.head()}\
                \rTags: {tags}\
                \nThe second drinking bowl will be selected.''')

                data2 = data[data['Corner'] == 2]
                data_2 = data2.copy()

                graph_dict = self.frame_analysis_for_graph(data2,
                                                           data_2,
                                                           tags,
                                                           dem_tags,
                                                           time_interval,
                                                           net,
                                                           dynamic=dynamic,
                                                           name_of_group=name_of_group)

            elif 1 not in list(input_data['Corner'].unique()) and 2 not in list(input_data['Corner'].unique()):
                print(f'''\nThere is no data on both drinking bowls in the next slice:\
                \rName of the group: {name_of_group}\
                \rData: {input_data.head()}\
                \rTags: {tags}''')

        elif corners == 1:
            if 1 in list(data['Corner'].unique()):
                data1 = data[data['Corner'] == 1]
                data_1 = data1.copy()

                graph_dict = self.frame_analysis_for_graph(data1,
                                                           data_1,
                                                           tags,
                                                           dem_tags,
                                                           time_interval,
                                                           net,
                                                           dynamic=dynamic,
                                                           name_of_group=name_of_group)
            else:
                print(f'''\nThere is no data on the first drinking bowl in the next slice:\
                \rName of the group: {name_of_group}\
                \rData: {input_data.head()}\
                \rTags: {tags}''')

        elif corners == 2:
            if 2 in list(data['Corner'].unique()):
                data2 = data[data['Corner'] == 2]
                data_2 = data2.copy()

                graph_dict = self.frame_analysis_for_graph(data2,
                                                           data_2,
                                                           tags,
                                                           dem_tags,
                                                           time_interval,
                                                           net,
                                                           dynamic=dynamic,
                                                           name_of_group=name_of_group)
            else:
                print(f'''\nThere is no data on the second drinking bowl in the next slice:\
                \rName of the group: {name_of_group}\
                \rData: {input_data.head()}\
                \rTags: {tags}''')

        return graph_dict

    def work(self,
             title_of_stage: str,
             names_of_files: list,
             without_lik: bool = False,
             only_with_lick: bool = False,
             time: float = None,
             time_start: str = None,
             time_finish: str = None,
             replacing_value: float = None,
             median_all_time: bool = True,
             delete_zero_in_intervals_for_median: bool = True,
             input_time_interval=None,
             corners='both',
             without_dem_base: bool = False,
             verbose: bool = True,
             dynamic: bool = False,
             intellicage: dict = None):
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
            without_lik: bool
                See the parameter 'without_lik' of the 'parser' method.
            only_with_lick: bool
                See the parameter 'only_with_lik' of the 'parser' method.
            time: float
                See the parameter 'time' of the 'parser' method.
            time_start: str
                See the parameter 'time_start' of the 'parser' method.
            time_finish: str
                See the parameter 'time_finish' of the 'parser' method.
            replacing_value: float
                Float number of seconds. The number of seconds after each visit during which the visits of followers are
                counted. Default value is None.
                Example: 12,4 or 30
            median_all_time: bool
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
            corners:
                Number of drinking bowls (bowls were in corners) to analyse. Default value is 'both'. 
                Example: 1 or 2 or 'both'
            without_dem_base: bool
                See the parameter 'without_dem' of the 'parser' method.
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False 
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False 
            intellicage: dict
                See the parameter 'intellicage' of the 'parser' method.              
        ------------
        Return:
             Tuple with:
                 1) pandas.DataFrame where indexes are rat tags and values are lists with intervals (seconds) between
                 visits;
                 2) list with dictionaries, each dictionary is a relation between rats-initiators and followers;
                 3) list with demonstrator's tags.
             Example: (pandas.DataFrame, list(dict1(), dict2(), dict3()), list)
        """

        if verbose:
            print(f'\n======= {title_of_stage} =======')

        bases = {}
        bases_tags = {}
        bases_dems = {}

        names_base = names_of_files
        for name_base in names_base:
            name_group = name_base.split(' ')[0]
            bases[name_base], \
            bases_tags[name_base], \
            bases_dems[name_base] = self.parser(name_base=name_base,
                                                name_animal=self.name_animal_file,
                                                input_path=self.input_path,
                                                name_group=name_group,
                                                without_lik=without_lik,
                                                only_with_lick=only_with_lick,
                                                time=time,
                                                verbose=verbose,
                                                without_dem=without_dem_base,
                                                time_start=time_start,
                                                time_finish=time_finish,
                                                intellicage=intellicage)

        if median_all_time:  # special computing space
            if verbose:
                print('Special calculations (median over the whole time, for each group).')
            bases_all = {}
            bases_tags_all = {}
            bases_dems_all = {}

            for name_base in names_base:
                name_group = name_base.split(' ')[0]
                bases_all[name_base], \
                bases_tags_all[name_base], \
                bases_dems_all[name_base] = self.parser(name_base=name_base,
                                                        name_animal=self.name_animal_file,
                                                        input_path=self.input_path,
                                                        name_group=name_group,
                                                        without_lik=without_lik,  # attention
                                                        only_with_lick=only_with_lick,  # attention
                                                        time=None,
                                                        verbose=False,
                                                        without_dem=False,  # attention
                                                        time_start=None,
                                                        time_finish=None,
                                                        intellicage=intellicage)
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

            if verbose:
                print(f'{tag} : {diff}')

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
                if median_all_time:
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
            self.dict_with_medians[f'{k} ({title_of_stage})'] = time_interval

            if verbose:
                print(f'Median in {k} = {time_interval}')

            v.sort_values(by=['_StartTime'], inplace=True)
            if input_time_interval == 'auto':
                graph_dict = self.graph(v,
                                        bases_tags[k],
                                        bases_dems[k],
                                        net,
                                        time_interval=time_interval,
                                        corners=corners,
                                        dynamic=dynamic,
                                        name_of_group=k)
            else:
                graph_dict = self.graph(v,
                                        bases_tags[k],
                                        bases_dems[k],
                                        net,
                                        time_interval=input_time_interval,
                                        corners=corners,
                                        dynamic=dynamic,
                                        name_of_group=k)

            list_with_dicts[k] = graph_dict

        if not dynamic:
            net.show_buttons(filter_=['physics',
                                      # 'nodes'
                                      ])
            net.save_graph(f'{self.output_path}\\{title_of_stage}_graph.html')

        if verbose:
            print('\n------- Rat tags in each group -------')
            for k, v in bases_tags.items():
                print(k, v)
            print(base)

        return base, list_with_dicts, all_dems

    def eda_graph(self,
                  without_lik: bool = False,
                  only_with_lick: bool = False,
                  time: float = None,
                  verbose: bool = True,
                  replacing_value='auto',
                  without_dem_base: bool = False,
                  input_time_interval='auto',
                  corners='both',
                  time_start: str = None,
                  time_finish: str = None,
                  delete_zero_in_intervals_for_median : bool = False,
                  median_all_time: bool = False,
                  dynamic: bool = False,
                  intellicage: dict = False):
        """
        ------------ 
        Function:
            This method creates a graph (based on puvis) for each stage (keys in the dictionary 'dict_names'),
            and also prints information about processed files, individuals, etc.
        ------------
        Parameters:
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False 
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
            corners:
                Number of drinking bowls (bowls were in corners) to analyse. Default value is 'both'. 
                Example: 1 or 2 or 'both'
            time_start: str
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original finish
                time in the file, 'time_finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45'            
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is False.
                Example: True of False            
            median_all_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is False.
                Example: True or False
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False            
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
                Default value is None.
                Example: {'illumination':'all_time', 
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
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

        if verbose:
            print('\n======= Work with graphs (eda_graph) =======')

        if replacing_value == 'auto':
            if time is None:
                if time_start is not None and time_finish is not None:
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
                                         without_lik=without_lik,
                                         only_with_lick=only_with_lick,
                                         time=time,
                                         verbose=verbose,
                                         without_dem_base=without_dem_base,
                                         dynamic=dynamic,
                                         input_time_interval=input_time_interval,
                                         corners=corners,
                                         time_start=time_start,
                                         time_finish=time_finish,
                                         delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                         median_all_time=median_all_time,
                                         replacing_value=replacing_value,
                                         intellicage=intellicage
                                         )
                               )
        return output_list

    @staticmethod
    def graph_plotly(data: dict,
                     dem_tags: list,
                     coords='default',
                     name_of_group: str = '_'):
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
            name_of_group: str
                Name of the particular rat group in the visit data. It must be the first word (with space after that)
                in the 'name_base' (See the parameter 'name_base' of the 'parser' method). Default value is '_'.
                Example: 'group1'
        ------------
        Return:
            List with trace with nodes and traces with edges for plotly.graph_objects.Scatter.
            Example: [node_trace,*edge_traces]
            or
            plotly.graph_objects.Scatter with the label 'No data on drinking bowls'.
        """

        name_column = f'Visits in {name_of_group}'

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
            Only side effect. It makes a html-file with a dynamic graph.
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
                                         "prefix": "Time:",
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
                    'label': str(times[oper_times])[-8:],
                    'value': str(times[oper_times])[-8:]}
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
        fig.write_html(f"{self.output_path}\\{name}.html")

    def dynamic_graphs(self,
                       start: str,
                       finish: str,
                       slide: float,
                       step: float,
                       without_lik: bool = False,
                       only_with_lick: bool = False,
                       time: float = None,
                       verbose: bool = False,
                       replacing_value='auto',
                       without_dem_base: bool = False,
                       input_time_interval='auto',
                       corners='both',
                       delete_zero_in_intervals_for_median: bool = True,
                       median_all_time: bool = True,
                       dynamic: bool = False,
                       intellicage: dict = None):
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
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time).
                Example: '12:30:45'
            slide: float
                Slice (minute time slide) to make one frame for dynamic graph.
                Example: 30 or 35.5
            step: float
                Step (minutes) to move time slide of the graph.
                Example: 10 or 15.5
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False 
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
            corners: 
                Number of drinking bowls (bowls were in corners) to analyse. Default value is 'both'. 
                Example: 1 or 2 or 'both'
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False
            median_all_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is False.
                Example: True or False 
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
                Default value is None.
                Example: {'illumination':'all_time', 
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}    
        ------------
        Return:
            List with file names and dictionary about graphs (where keys are time and 
            values are dictionaries (keys are file names and values are returns from 'graph_plotly' method)).
                Example: [names, dynamic_graph_dict]
            Side effects are html-files with graphs.
        """

        start = pd.to_datetime(start, format='%H:%M:%S')
        end_of_all = pd.to_datetime(finish, format='%H:%M:%S')
        slide = pd.offsets.DateOffset(minutes=slide)
        step = pd.offsets.DateOffset(minutes=step)

        new_start = start
        new_finish = start + slide

        time_dict = {}
        while new_start <= end_of_all - slide:
            str_new_start = str(new_start)[-8:]
            str_new_finish = str(new_finish)[-8:]

            data = self.eda_graph(without_lik=without_lik,
                                  only_with_lick=only_with_lick,
                                  time=time,
                                  verbose=verbose,
                                  replacing_value=replacing_value,
                                  without_dem_base=without_dem_base,
                                  input_time_interval=input_time_interval,
                                  corners=corners,
                                  time_start=str_new_start,  # pay attention
                                  time_finish=str_new_finish,  # pay attention
                                  delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                  median_all_time=median_all_time,
                                  dynamic=dynamic,
                                  intellicage=intellicage)

            time_dict[new_finish] = data

            new_start += step
            new_finish += step

        final_time_dict = {}
        names = []
        dems = []

        for time, data in time_dict.items():
            sup_dict = {}
            for cortege in data:
                dems += cortege[2]

                for name, dict_relations in cortege[1].items():
                    if name not in names:
                        names.append(name)
                    sup_dict[name] = self.graph_plotly(dict_relations,
                                                       cortege[2],
                                                       name_of_group=name)
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
                       without_lik: bool = False,
                       only_with_lick: bool = False,
                       time: float = None,
                       verbose: bool = True,
                       replacing_value='auto',
                       without_dem_base: bool = False,
                       input_time_interval='auto',
                       corners='both',
                       time_start: str = None,
                       time_finish: str = None,
                       delete_zero_in_intervals_for_median: bool = True,
                       median_all_time: bool = True,
                       dynamic: bool = False,
                       intellicage: dict = None):
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
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
            time: float
                Time (minutes only) since the beginning of the file. Default value is None.
                Example: 30 or 30.5
            verbose: bool
                Displaying information while the algorithm is running. Default value is False.
                Example: True or False 
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
            corners:
                Number of drinking bowls (bowls were in corners) to analyse. Default value is 'both'. 
                Example: 1 or 2 or 'both'
            time_start: str
                Particular start time to download the data (if your 'time_start' is less than the original start time 
                in the file, 'time_start' will equal to the original start time). Default value is None.
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            time_finish: str
                Particular finish time to download the data (if your 'time_finish' is higher than the original finish
                time in the file, 'time_finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45'            
            delete_zero_in_intervals_for_median: bool
                Deleting zeros in the median calculation or not. Default value is True.
                Example: True of False            
            median_all_time: bool
                Way to calculate the median of visits. If True, it is evaluated on all data connected with the
                particular group. Default value is True.
                Example: True or False            
            dynamic: bool
                Using 'dynamic' mode, when you want to make a graph with Plotly Time Slider. Default value is false.
                Example: True or False            
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
                Default value is None.
                Example: {'illumination':'all_time', 
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
        ------------
        Return:
            Only side effect. Printing all metrics.
        """

        data_for_graph = self.eda_graph(without_lik=without_lik,
                                        only_with_lick=only_with_lick,
                                        time=time,
                                        verbose=verbose,
                                        replacing_value=replacing_value,
                                        without_dem_base=without_dem_base,
                                        input_time_interval=input_time_interval,
                                        corners=corners,
                                        time_start=time_start,
                                        time_finish=time_finish,
                                        delete_zero_in_intervals_for_median=delete_zero_in_intervals_for_median,
                                        median_all_time=median_all_time,
                                        dynamic=dynamic,
                                        intellicage=intellicage)

        stage_names = self.dict_names.keys()
        data_for_graph = list(zip(stage_names, data_for_graph))
        for stage_name, cortege in data_for_graph:
            for name_file, graph_dict in cortege[1].items():
                adges_list = []
                node_credibilities = {}
                node_susceptibilities = {}
                if graph_dict is not None:
                    for key_tag, v in graph_dict.items():
                        for index_tag, relations in list(zip(v.index, v.values)):
                            if relations[0] != 0:
                                for tag in [index_tag] * relations[0]:
                                    adges_list.append((key_tag, tag))
                            if index_tag in node_susceptibilities:
                                node_susceptibilities[index_tag] += relations[0]
                            else:
                                node_susceptibilities[index_tag] = relations[0]
                        node_credibilities[key_tag] = v.values.sum()

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
                    g.add_edges_from(adges_list)

                    print(f'\n======= Graph metrics of {name_file} ({stage_name}) =======')
                    try:
                        print("Eccentricity: ", [(x, y) for x, y in nx.eccentricity(g).items()])
                    except:
                        print('There is no values to evaluate eccentricity')
                    try:
                        print("Diameter: ", nx.diameter(g))
                    except:
                        print('There is no values to evaluate diameter')
                    try:
                        print("Radius: ", nx.radius(g))
                    except:
                        print('There is no values to evaluate radius')
                    try:
                        print("Periphery: ", list(nx.periphery(g)))
                    except:
                        print('There is no values to evaluate periphery')
                    try:
                        print("Center: ", list(nx.center(g)))
                    except:
                        print('There is no values to evaluate center')

                    print('Node credibility: ', node_credibilities)
                    print('Node susceptibilities: ', node_susceptibilities)
                    print('Graph weight: ', graph_weight)
                    print('Median for graph analysis: ', self.dict_with_medians[f'{name_file} ({stage_name})'])
                else:
                    print(f'\nIn {stage_name} {name_file} there is no information')

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
                  without_lik: bool = False,
                  only_with_lick: bool = False,
                  time: float = None,
                  verbose: bool = False,
                  without_dem: bool = False,
                  time_start: str = None,
                  time_finish: str = None,
                  intellicage: dict = None,
                  density: bool = True,
                  bins: int = 12,
                  size: tuple = (5, 5),
                  condition: dict = None):
        """
        ------------
        Function:
            Method for computing visit statistics on rat groups. 
        ------------
        Parameters:
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
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
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            time_finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45'
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
                Default value is None.              
                Example: {'illumination':'all_time', 
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
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
                Dictionary where keys are stage names and values are parameters for the 'parser' method (customizable).
                Example: {'stage1': True} ('without_dem' = True)
        ------------
        Return:
            Dictionary where keys are stage names and values are pandas.DataFrames 
            (where indexes are rat tags and values are numbers of visits).
            Example: {'stage1': frame1,
                      'stage2': frame2}        
        """

        return_dict = {}
        input_without_dem = copy.deepcopy(without_dem)

        for stage, file_names in self.dict_names.items():
            bases = {}
            bases_tags = {}
            bases_dem_tags = {}

            # =========================================================== customizable
            if stage in list(condition.keys()):
                without_dem = condition[stage]
            else:
                without_dem = input_without_dem
            # ===========================================================

            for name_base in file_names:
                name_group = name_base.split(' ')[0]
                bases[name_base], \
                bases_tags[name_base], \
                bases_dem_tags[name_base] = self.parser(name_base=name_base,
                                                        name_group=name_group,
                                                        name_animal=self.name_animal_file,
                                                        input_path=self.input_path,
                                                        without_lik=without_lik,
                                                        only_with_lick=only_with_lick,
                                                        time=time,
                                                        verbose=verbose,
                                                        without_dem=without_dem,
                                                        time_start=time_start,
                                                        time_finish=time_finish,
                                                        intellicage=intellicage)

            base = pd.concat([*bases.values()], ignore_index=True)
            base = base.groupby(['Tag']).count()

            sup_bases = []
            for i in bases_tags.values():
                for j in i:
                    if j is not None and j not in sup_bases:
                        sup_bases.append(j)

            for tag in sup_bases:
                if tag not in base.index:
                    base.loc[tag] = 0
            base = base['VisitID']

            return_dict[stage] = base

            if verbose:
                print(f'\n======= Rat tags in groups =======')
                for k, v in bases_tags.items():
                    print(k, v)

                print(f'\n======= {stage} =======')
                for k, v in bases.items():
                    v.index = v['VisitID']
                    v = v.groupby(['Tag']).count()
                    v['The number of visits of each rat'] = v['VisitID']

                    print('\n', k, '\n', v['The number of visits of each rat'])

                print(f'\n======= Information on the whole {stage} =======')
                print(base)

                fig, ax = plt.subplots(figsize=size)
                n, bins, _ = ax.hist(base, bins=bins, log=False, density=density)
                self.label_densityhist(ax, n, bins, x=4, y=0.01, r=2)
                ax.set_title(stage)
                ax.axes.yaxis.set_ticks([])
                print(f'\nBin boundaries: {[int(x) for x in list(bins)]}')

        return return_dict

    def eda_intervals(self,
                      without_lik: bool = False,
                      only_with_lick: bool = False,
                      replacing_value='auto',
                      verbose: bool = False,
                      verbose_detailed: bool = False,
                      without_dem: bool = False,
                      time: float = None,
                      time_start: str = None,
                      time_finish: str = None,
                      intellicage: dict = None,
                      condition: dict = None,
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
            without_lik: bool
                Filter the data so that only rats that didn't drink remain. It raises error if you define 'without_lik'
                and 'only_with_lik' as True. Both True means all data will be downloaded. Default value is False.
                Example: True or False
            only_with_lick: bool
                Filter the data so that only rats that did drink remain. Default value is False.
                Example: True or False
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
                Example: '12:30:45' (12 hours 30 minutes 45 seconds)
            time_finish: str
                Particular finish time to download the data (if your 'finish' is higher than the original finish time 
                in the file, 'finish' will be equal to the original finish time). Default value is None.
                Example: '12:30:45'
            intellicage: dict
                It is dictionary where you put intellicage_parser's parameters (See the 'intellicage_parser' method).
                Default value is None.              
                Example: {'illumination':'all_time', 
                          'condition': {'session1':'last_day',
                                        'session2':'last_day',
                                        'session3':'first_day',
                                        'date':time}}
            condition: dict
                Dictionary where keys are stage names and values are parameters for the 'parser' method (customizable).
                Example: {'stage1': True} ('without_dem' = True)
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
        """

        if replacing_value == 'auto':
            if time is None:
                if time_start is not None and time_finish is not None:
                    time_start_for_replacing_value = pd.to_datetime(time_start, format='%H:%M:%S')
                    time_finish_for_replacing_value = pd.to_datetime(time_finish, format='%H:%M:%S')
                    replacing_value = (time_finish_for_replacing_value - time_start_for_replacing_value).seconds
                else:
                    replacing_value = 24 * 60 * 60  # whole day
            else:
                replacing_value = time * 60

        return_dict = {}
        input_without_dem = copy.deepcopy(without_dem)

        for stage, file_names in self.dict_names.items():
            bases = {}
            bases_tags = {}
            bases_dem_tags = {}

            # =========================================================== customizable
            if condition is not None and stage in list(condition.keys()):
                without_dem = condition[stage]
            else:
                without_dem = input_without_dem
            # ===========================================================

            for name_base in file_names:
                name_group = name_base.split(' ')[0]
                bases[name_base], \
                bases_tags[name_base], \
                bases_dem_tags[name_base] = self.parser(name_base=name_base,
                                                        name_group=name_group,
                                                        name_animal=self.name_animal_file,
                                                        input_path=self.input_path,
                                                        without_lik=without_lik,
                                                        only_with_lick=only_with_lick,
                                                        time=time,
                                                        verbose=verbose,
                                                        without_dem=without_dem,
                                                        time_start=time_start,
                                                        time_finish=time_finish,
                                                        intellicage=intellicage)
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
                    base_.loc[tag, f'{stage}_intervals'] = replacing_value
                else:
                    base_.loc[tag, f'{stage}_intervals'] = np.median(diff)

                if verbose_detailed:
                    print(f'{tag} : {diff}')

                    if len(diff) > 0:
                        fig, ax = plt.subplots()
                        n, bins, _ = plt.hist(diff,
                                              bins=None,
                                              log=False,
                                              density=density,
                                              range=hist_range_detailed)
                        self.label_densityhist(ax, n, bins, x=4, y=0.01, r=2)
                        plt.title(f'{tag} : {diff} \nBin boundaries: {[int(x) for x in list(bins)]}')
                        ax.axes.yaxis.set_ticks([])

            base = base_

            for tag in sup_bases:
                if tag not in base.index:
                    base.loc[tag] = replacing_value
            if replacing_value is not None:
                base = base.fillna(value=replacing_value)  # situation len(new_df) == 1 or == 0

            base = base[f'{stage}_intervals']

            return_dict[stage] = base

            if verbose:
                print(f'\n======= Rat tags in groups =======')
                for k, v in bases_tags.items():
                    print(k, v)

                print(f'\n======= Information on the whole {stage} =======')
                print(base)

                fig, ax = plt.subplots(figsize=size)
                n, bins, _ = ax.hist(base,
                                     bins=bins,
                                     log=False,
                                     density=density,
                                     range=hist_range)
                self.label_densityhist(ax, n, bins, x=4, y=0.01, r=2)
                ax.set_title(stage)
                ax.axes.yaxis.set_ticks([])
                print(f'\nBin boundaries: {[int(x) for x in list(bins)]}')

        return return_dict