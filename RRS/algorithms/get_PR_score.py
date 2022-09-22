
import joblib
import pandas as pd
from utils.config.common import conf

df = pd.read_pickle(f"{conf.dataPath}pr_base_df.pkl")
pr_model = joblib.load(f"{conf.dataPath}pr_score_model.pkl")


class PR_score():
    def __init__(self):
        pass

    def predict_positive_reply(self, x, top_k=30):
        x_info = df[df['mem_no']==x].reset_index(drop=True)
        x_age = x_info.loc[0, 'age']
        x_gender = x_info.loc[0, 'mem_sex']
        x_view = x_info.loc[0, 'view_list']
        x_send = x_info.loc[0, 'send_list']


        y_range = df[(abs(x_age - df['age']) <= 5) & (df['mem_sex']!=x_gender)].mem_no.values
        y_range = [y for y in y_range if y not in x_view and y not in x_send]
        y_info = df[df['mem_no'].isin(y_range)].reset_index(drop=True)

        if x_gender == 'm' :
            y_info.columns = ['ptr_' + c for c in y_info.columns]
            y_info['mem_no'] = x
        else :
            x_info.columns = ['ptr_' + c for c in x_info.columns]
            y_info['ptr_mem_no'] = x

        x_y_info = pd.merge(x_info, y_info)
        x_y_info.drop(['mem_no', 'ptr_mem_no', 'mem_sex', 'ptr_mem_sex'], axis=1, inplace=True)
        x_y_info['age_gap'] = x_y_info['age'] - x_y_info['ptr_age']

        x_y_info['loc_like'] = x_y_info.apply(lambda x : 1 if x['mem_loc'] == x['ptr_mem_loc'] else 0, axis =1)
        x_y_info['marriage_like'] = x_y_info.apply(lambda x : 1 if x['mem_loc'] == x['ptr_mem_loc'] else 0, axis =1)
        x_y_info.drop(['mem_loc', 'mate_slct', 'ptr_mem_loc', 'ptr_mate_slct', 'view_list', 'viewed_list',
                       'send_list', 'sent_list', 'ptr_view_list', 'ptr_viewed_list', 'ptr_send_list',
                       'ptr_sent_list','ptr_smoke_slct'], axis=1, inplace=True)

        x_y_info = pd.get_dummies(x_y_info)
        for col in pr_model.feature_names_ :
            if col not in x_y_info.columns :
                x_y_info[col] = 0

        pred = pr_model.predict_proba(x_y_info)
        pred_result = pd.DataFrame(pred[:, 1], columns=['score'], index=y_range)
        pred_result = pred_result.sort_values('score', ascending=False).head(top_k)

        return pred_result