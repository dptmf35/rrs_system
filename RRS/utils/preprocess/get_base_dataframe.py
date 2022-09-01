import pandas as pd
from utils.config.common import data_path
from datetime import datetime
from functions import *

mate_df = pd.read_pickle(f'{data_path}mate_df.pkl')
basic_df = pd.read_pickle(f'{data_path}basic_df.pkl')
view_df = pd.read_pickle(f'{data_path}view_df.pkl')
msg_df = pd.read_pickle(f'{data_path}msg_df.pkl')

class getBaseDataFrame() :
    def __init__(self):
        pass

    def pr_base_dataframe(self, mate_df=mate_df):
        bins = [0, 150, 160, 170, 180, 200]
        mate_df['mate_height'] = pd.cut(mate_df['mate_height'], bins=bins,
                                        labels=['under_150', '150', '160', '170', 'over_180'])
        mate_df = mate_df[['mem_no', 'mem_sex', 'mem_loc', 'mate_religion', 'mate_car', 'mate_job', 'mate_ann_salary',
                           'mate_career', 'mate_style', 'mate_charc', 'mate_hobby', 'favor_food', 'possess_property',
                            'smoke_slct', 'drink_slct','health_slct', 'mate_height', 'mate_slct']]

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


    def get_cf_base_dataframe(self):
        mate_df['age'] = int(str(datetime.today())[:4]) - mate_df['mem_birth_year']
        df = mate_df[['mem_no', 'mem_sex', 'age']]
        # pf view / pf viewed
        df = self.get_member_list(df, view_df, 'view_list', 'viewed_list')
        # msg send / msg sent
        df = self.get_member_list(df, msg_df, 'send_list', 'sent_list')

        return df

    def corr_base_transform(self, mate_df=mate_df):
        # ['location', 'smoke', 'height', 'career', 'salary', 'health','hobby', 'religion', 'possess', 'drink']
        mate_df = mate_df[['mem_no', 'mem_sex', 'mem_loc', 'mate_religion', 'mate_car', 'mate_job', 'mate_ann_salary',
                           'mate_career', 'mate_style', 'mate_charc', 'mate_hobby', 'favor_food', 'possess_property',
                           'smoke_slct', 'drink_slct', 'health_slct', 'mate_height', 'mate_slct']]
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

    def get_member_list(self, df, df2, list_name, list_name2):
        df[list_name] = None
        # member list x viewed
        for i in range(len(df)) :
            temp = df2[df2['mem_no']==df.loc[i, 'mem_no']]['ptr_mem_no'].values.tolist()
            df.loc[i, list_name] = [[temp]]

        df[list_name2] = None
        # member list view x
        for i in range(len(df)) :
            temp = df2[df2['ptr_mem_no']==df.loc[i, 'mem_no']]['mem_no'].values.tolist()
            df.loc[i, list_name] = [[temp]]

        df[list_name] = df[list_name].apply(lambda x : x[0])
        df[f'{list_name}_count'] = df[list_name].apply(lambda x : len(x))

        df[list_name2] = df[list_name2].apply(lambda x : x[0])
        df[f'{list_name2}_count'] = df[list_name2].apply(lambda x : len(x))

        return df