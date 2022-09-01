
import joblib
import pandas as pd
from utils.config.common import data_path

df = pd.read_pickle(f"{data_path}pr_score_base_df.pkl")
pr_model = joblib.load(f"{data_path}pr_score_model.pkl")


class PR_score():
    def __init__(self):
        pass

    def predict_positive_reply(self, x, top_k=50):
        x_info = df[df['mem_no']==x]
        x_age = x_info.age.values[0]
        x_gender = x_info.mem_sex.values[0]
        x_view = x_info.view_list.values[0]
        x_send = x_info.send_list.values[0]

        y_range = df[(abs(x_age - df['age']) <= 5) & (df['mem_sex']!=x_gender)].mem_no.values
        y_range = [y for y in y_range if y not in x_view and y not in x_send]
        y_info = df[df['mem_no'].isin(y_range)]

        if x_gender == 'm' :
            y_info.columns = ['ptr_' + c for c in y_info.columns]
            y_info['mem_no'] = x
        else :
            y_info.columns = ['ptr_' + c for c in x_info.columns]
            y_info['ptr_mem_no'] = x

        x_y_info = pd.merge(x_info, y_info)
        x_y_info.drop(['mem_no', 'ptr_mem_no', 'mem_sex', 'ptr_mem_sex'], axis=1, inplace=True)
        x_y_info['age_gap'] = x_y_info['age'] - x_y_info['ptr_age']

        x_y_info['loc_like'] = x_y_info.apply(lambda x : 1 if x['mem_loc'] == x['ptr_mem_loc'] else 0, axis =1)
        x_y_info['marriage_like'] = x_y_info.apply(lambda x : 1 if x['mem_loc'] == x['ptr_mem_loc'] else 0, axis =1)
        x_y_info.drop(['mem_loc', 'mate_slct', 'ptr_mem_loc', 'ptr_mate_slct'], axis=1, inplace=True)

        pred = pr_model.predict_proba(x_y_info)
        pred_result = pd.DataFrame(pred[:, 1], columns=['score'], index=y_range)
        pred_result = pred_result.sort_values('score', ascending=False).head(top_k)

        return pred_result