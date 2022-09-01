from utils.preprocess.preprocess import Preprocessor
from utils.preprocess.dbconnector import data_path
import pandas as pd

p = Preprocessor()


if __name__ == '__main__' :
    df = p.df_preprocess()
    df.to_csv(f'{data_path}df.csv', index=None)

    df = pd.read_csv(f'{data_path}df.csv')
    basic_df = pd.read_csv(f'{data_path}basic_df.csv')
    mate_df = pd.read_csv(f'{data_path}mate_df.csv')

    feature_df = p.make_feature_df(df, basic_df, mate_df)
    feature_df.to_csv(f'{data_path}feature_df.csv', index=None)

    corr_df = p.make_corr_df(mate_df)
    corr_df.to_csv(f'{data_path}corr_df.csv', index=None)

    df = pd.read_csv(f'{data_path}df.csv')
    print(df.sort_values(by='send_list_count', ascending=False)['mem_no'].values[:20])
    print(df[df['mem_sex']=='f'].sort_values(by='send_list_count', ascending=False)['mem_no'].values[:20])