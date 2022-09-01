from utils.preprocess.dbconnector import DBConnector, data_path
from utils.functions import *
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Preprocessor(DBConnector) :
    def __init__(self):
        super().__init__()

    def df_preprocess(self):
        view_df, msg_df, mate_df, basic_df = self.save_dataframe()

        # calculate age with mem_birth_year
        mate_df['age'] = int(str(datetime.today())[:4]) - mate_df['mem_birth_year']
        df = mate_df[['mem_no', 'mem_sex', 'age']]

        list_name = ['view_list', 'viewed_list', 'send_list', 'sent_list']
        for i, lname in enumerate(list_name) : # make columns in list_name
            if i <= 1 :
                print('view - df -')
                df = self.get_list(lname, df, view_df)
            else :
                print('msg - df - ')
                df = self.get_list(lname, df, msg_df)

        return df


    def get_list(self, list_name, df, list_df):
        # member list viewed by x
        df[list_name] = None # initialize view_list
        for i in range(len(df)) :
            if list_name in ['sent_list', 'viewed_list'] :
                value_list = list_df[list_df['ptr_mem_no'] == df.loc[i, 'mem_no']].mem_no.values.tolist()
            else :
                value_list = list_df[list_df['mem_no'] == df.loc[i, 'mem_no']].ptr_mem_no.values.tolist()
            df.loc[i, list_name] = [[value_list]]

        df[list_name] = df[list_name].apply(lambda x : x[0])
        df[f'{list_name}_count'] = df[list_name].apply(lambda x : len(x)) # total view amount

        df = df[df[f'{list_name}_count'] > 0]
        df.reset_index(drop=True, inplace=True)
        return df

    # make feature df for calculate PR score
    def make_feature_df(self, df, basic_df, mate_df):
        temp = pd.merge(df[['mem_no', 'mem_sex', 'age', 'view_list_count', 'viewed_list_count', 'send_list_count',
                            'sent_list_count']], basic_df[['mem_no','login_cnt','tot_stay_time']])
        feature_df = pd.merge(temp, mate_df[['mem_no', 'photo_cnt', 'upd_cnt']])
        return feature_df


    # make correlation df for explanation
    def make_corr_df(self, mate_df):
        # ['location', 'smoke', 'height', 'career', 'salary', 'health','hobby', 'religion', 'possess', 'drink']
        mate_df = mate_df[['mem_no', 'mem_sex', 'mem_loc', 'mate_religion', 'mate_car', 'mate_job', 'mate_ann_salary',
                           'mate_career', 'mate_style', 'mate_charc', 'mate_hobby', 'favor_food', 'possess_property',
                            'smoke_slct', 'drink_slct','health_slct', 'mate_height', 'mate_slct']]
        bins = [0, 150, 160, 170, 180, 200]
        mate_df['mate_height'] = pd.cut(mate_df['mate_height'], bins=bins,
                                            labels=['under_150', '150', '160', '170', 'over_180'])
        mate_df['drink_slct'] = mate_df['drink_slct'].apply(drink_code)
        mate_df['smoke_slct'] = mate_df['smoke_slct'].apply(lambda x: 1 if len(str(x)) > 0 and str(x)[0] == 'c' else 0)
        mate_df['health_slct'] = mate_df['health_slct'].apply(health_code)

        mate_df = char_transform(mate_df)
        mate_df = food_transform(mate_df)
        mate_df = hobby_transform(mate_df)

        mate_df['mate_religion'] = mate_df['mate_religion'].apply(religion_code)
        mate_df['mate_job'] = mate_df['mate_job'].apply(job_code)
        mate_df['mate_career'] = mate_df['mate_career'].apply(career_code)
        mate_df['mate_car'] = mate_df['mate_car'].apply(lambda x: 0 if x == 9 else 1)
        mate_df['mate_ann_salary'] = mate_df['mate_ann_salary'].apply(salary_code)
        mate_df['possess_property'] = mate_df['possess_property'].apply(pos_code)
        mate_df['mate_career'] = mate_df['mate_career'].apply(career_code)
        mate_df['mate_ann_salary'] = mate_df['mate_ann_salary'].apply(salary_code)
        mate_df['health_slct'] = mate_df['health_slct'].apply(health_code)
        mate_df['smoke_slct'] = mate_df['smoke_slct'].apply(lambda x: 1 if len(str(x)) > 0 and str(x)[0] == 'c' else 0)

        del mate_df['mate_charc'], mate_df['favor_food'], mate_df['mate_hobby'], mate_df['mem_sex']

        return mate_df
