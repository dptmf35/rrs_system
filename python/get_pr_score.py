from utils.preprocess.dbconnector import data_path
import pandas as pd
import joblib
import numpy as np
model = joblib.load(f'{data_path}predpr.pkl')

class PR_score :
    def __init__(self):
        pass
    def get_PR_score(self, x, y): # predict positive reply between user x, y
        feature_df = pd.read_csv(f'{data_path}feature_df.csv')
        feature_df = feature_df.rename(columns={'send_list_count': 'send_count', 'view_list_count': 'view_count',
                                                    'sent_list_count': 'sent_count',
                                                    'viewed_list_count': 'viewed_count'})
        x_features = feature_df[feature_df['mem_no']==x]
        y_features = feature_df[feature_df['mem_no']==y]

        if feature_df[feature_df['mem_no']==x].mem_sex.values[0] == 'f' :
            x_features.columns = ['ptr_' + c for c in x_features.columns]

        else :
            y_features.columns = ['ptr_' + c for c in y_features.columns]

        x_y_features = pd.concat([x_features.reset_index(drop=True), y_features.reset_index(drop=True)], axis=1)
        x_y_features.drop(['mem_no','ptr_mem_no','mem_sex','ptr_mem_sex'], axis=1, inplace=True)
        x_y_features['age_gap'] = x_y_features['age'] - x_y_features['ptr_age']
        pred = model.predict_proba(x_y_features)[0]
        idx = np.argmax(pred)
        return idx, pred[idx]
