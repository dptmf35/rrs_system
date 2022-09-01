import pandas as pd
from utils.preprocess.dbconnector import data_path
import time

df = pd.read_csv(f'{data_path}df.csv')
corr_df = pd.read_csv(f'{data_path}corr_df.csv')


class ReciprocalExplanation() :

    def __init__(self):
        # self.df = pd.read_csv(f'{data_path}df.csv')
        # self.corr_df = pd.read_csv(f'{data_path}corr_df.csv')
        pass

    def get_reciprocal_explanation(self, x, y, k=3):
        e_x_y = self.get_explanation(x, y, k) # e_x,y
        e_y_x = self.get_explanation(y, x, k) # e_y,x

        # re_x_y = f'''most important feature of {y} for {x} : {e_x_y} //// most important feature of {x} for {y} : {e_y_x}'''

        re_x_y = [e_x_y, e_y_x]
        return re_x_y

    def get_explanation(self, x, y, k=3): # input user x

        view_list = eval(df[df['mem_no']==x].view_list.values[0])
        send_list = eval(df[df['mem_no']==x].send_list.values[0])

        # m_x
        m_x = [1 if i in send_list else 0 for i in view_list]
        av_df = pd.DataFrame([m_x], columns=view_list, index=[x]).T
        # S_x_a_v
        for col in corr_df.columns[1:] :
            s_x_a_v = [corr_df[corr_df['mem_no']==y][col].values[0] for y in av_df.index]
            av_df[col] = s_x_a_v
        # print(av_df.head())
        av_df = pd.get_dummies(av_df)

        y_av = pd.get_dummies(corr_df[corr_df['mem_no']==y])
        for y_col in y_av.columns :
            if y_col in av_df.columns and y_av[y_col].values[0] != 1 :
                av_df.drop(y_col,inplace=True,axis=1)
        av_df.drop([av_col for av_col in av_df.columns if av_col not in y_av.columns and av_col != x], axis=1, inplace=True)

        result = av_df.corr()[x].sort_values(ascending=False) # correlation(Pearson)
        # print(result)
        # print(f'highest correlation feature for {x} : {result.index[1]}, correlation value : {result.values[1]}')

        return result.index[1:k+1].tolist()

if __name__ == '__main__' :
    start = time.time()
    rpc = ReciprocalExplanation()
    x = 1697018
    y = 1789272
    result = rpc.get_reciprocal_explanation(x, y, k=3)
    print(result)
    print('=' * 40)
    print(f'elapsed time : {time.time() - start}')