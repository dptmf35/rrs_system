import pymysql
import pandas as pd
import os
from datetime import datetime, timedelta
data_path = './utils/data/'

class DBConnector :
    def __init__(self):
        self.conn = pymysql.connect(
            user='my_srv',
            passwd='wkrldi@duqhdi12',
            host='125.141.223.156',
            db='m_yeoboya_ai_dv',
            charset='utf8',
            port=33141
        )

    def read_sql(self, table_name, *column_name):
        cursor = self.conn.cursor(pymysql.cursors.DictCursor)
        today = str(datetime.today())[:10]
        yesterday = str(datetime.today() - timedelta(1))[:10]
        if column_name :
            sql = f"SELECT * FROM m_yeoboya.{table_name} WHERE DATE({column_name[0]}) BETWEEN '{yesterday}' and '{today}'"
        else :
            sql = f"SELECT * FROM m_yeoboya.{table_name}"
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        df = pd.DataFrame(result)
        return df


    def save_dataframe(self):
        if not os.path.exists(f'{data_path}view_df.csv') :
            view_df = self.read_sql('member_pf_view_all_log', 'ins_date')
        else :
            view_df = pd.read_csv(f'{data_path}view_df.csv')
        if not os.path.exists(f'{data_path}msg_df.csv') :
            msg_df = self.read_sql('member_msg_log_back', 'ins_date')
        else :
            msg_df = pd.read_csv(f'{data_path}msg_df.csv')
        if not os.path.exists(f'{data_path}mate_df.csv') :
            mate_df = self.read_sql('member_mate')
        else :
            mate_df = pd.read_csv(f'{data_path}mate_df.csv')
        if not os.path.exists(f'{data_path}basic_df.csv') :
            basic_df = self.read_sql('member_basic')
        else :
            basic_df = pd.read_csv(f'{data_path}basic_df.csv')

        # delete duplications
        view_df.drop_duplicates(inplace=True, subset=['mem_no', 'ptr_mem_no'])
        view_df = view_df[view_df['mem_no'].isin(mate_df['mem_no'].values)]
        view_df = view_df[view_df['ptr_mem_no'].isin(mate_df['mem_no'].values)]
        view_df.reset_index(drop=True, inplace=True)
        print(f'pf view dataframe len : {len(view_df)}')


        # delete duplications
        msg_df.drop_duplicates(inplace=True, subset=['mem_no', 'ptr_mem_no'])
        msg_df = msg_df[msg_df['mem_no'].isin(mate_df['mem_no'].values)]
        msg_df = msg_df[msg_df['ptr_mem_no'].isin(mate_df['mem_no'].values)]
        msg_df.reset_index(drop=True, inplace=True)
        print(f'msg dataframe len : {len(msg_df)}')

        # save to csv file
        view_df.to_csv(f'{data_path}view_df.csv', index=None)
        msg_df.to_csv(f'{data_path}msg_df.csv', index=None)
        mate_df.to_csv(f'{data_path}mate_df.csv', index=None)
        basic_df.to_csv(f'{data_path}basic_df.csv', index=None)
        return view_df, msg_df, mate_df, basic_df

