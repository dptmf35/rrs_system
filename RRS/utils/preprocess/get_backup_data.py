from utils.config.db_configs import *
import pymysql
import pandas as pd


class getBackUpData() :
    def __init__(self):
        self.conn = conn

    def save_sql(self):
        mate_df = self.read_sql('member_mate')
        basic_df = self.read_sql('member_basic')
        view_df = self.read_sql('member_pf_view_all_log', 'ins_date')
        msg_df = self.read_sql('member_msg_log_back', 'ins_date')

        view_df = self.check_duplications(view_df, mate_df)
        msg_df = self.check_duplications(msg_df, mate_df)

        mate_df.to_pickle(f'{data_path}mate_df.pkl')
        basic_df.to_pickle(f'{data_path}basic_df.pkl')
        view_df.to_pickle(f'{data_path}view_df.pkl')
        msg_df.to_pickle(f'{data_path}msg_df.pkl')

    def read_sql(self, table_name, *column_name) :
        cursor = self.conn.cursor(pymysql.cursors.DictCursor)
        if column_name :
            sql = f"SELECT * FROM m_yeoboya.{table_name} WHERE DATE({column_name[0]}) BETWEEN '{yesterday}' AND '{today}'"
        else :
            sql = f"SELECT * FROM m_yeoboya.{table_name}"
        print('Processed SQL Query :', sql)
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        df = pd.DataFrame(result)
        print(f'DataFrame ({table_name}) Length : {len(df):,}')
        return df

    def check_duplications(self, df,mate_df):
        df.drop_duplicates(inplace=True, subset=['mem_no', 'ptr_mem_no'])
        df = df[(df['mem_no'].isin(mate_df['mem_no'].values)) & (df['ptr_mem_no'].isin(mate_df['mem_no'].values))]
        df = df.reset_index(drop=True)
        print(f'Deduplicated DataFrame Length : {len(df):,}')
        return df


if __name__ == '__main__' :
    getB = getBackUpData()
    getB.save_sql()