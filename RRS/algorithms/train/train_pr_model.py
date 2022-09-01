import pandas as pd
from utils.config.common import data_path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from catboost import CatBoostClassifier
from sklearn.metrics import auc, f1_score, roc_curve
import joblib

smote = SMOTE()
model = CatBoostClassifier(random_state=0, silent=True)

df = pd.read_pickle(f'{data_path}.pkl')
msg_df = pd.read_pickle(f'{data_path}msg_df.pkl')
ptr_df = df.copy()



class TrainPRModel() :
    def __init__(self):
        pass

    def train(self, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)

        fpr, tpr, threshold = roc_curve(y_test, pred_proba[:, 1])
        print(f'AUC Score : {auc(fpr, tpr):.3f}')

        # f1_score(y_test, pred)
        print(f'f1 score : {f1_score(y_test, pred):.3f}')

        joblib.dump(model, f"{data_path}pr_score_model.pkl")


    def data_splitter(self, X, y ):
        X_res, y_res = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
        print(f"Oversampled Dataset Result : {Counter(y_res)}")
        return X_train, X_test, y_train, y_test

    def generate_train_dataset(self):
        # Detect positive response message
        bi_temp = pd.DataFrame()
        cnt = 0
        temp = msg_df.copy()[['mem_no', 'ptr_mem_no']]
        temp2 = temp.rename(columns={"mem_no" : "ptr_mem_no", "ptr_mem_no" : "mem_no"})
        for i in range(len(temp)) :
            t = temp[(temp['mem_no']==temp2.loc[i, 'mem_no']) & (temp['ptr_mem_no']==temp2.loc[i, 'ptr_mem_no'])]
            if len(t) >= 1 :
                cnt += 1
                bi_temp = pd.concat([bi_temp, t])

        bi_temp['positive_response'] = 1
        temp = temp.drop(bi_temp.index)
        temp['positive_response'] = 0

        res = pd.concat([bi_temp, temp])
        print(res['positive_response'].value_counts())

        temp = pd.merge(res, df)
        data = pd.merge(temp, ptr_df)

        male_data = data[data['mem_sex']=='m']
        female_data = data[data['mem_sex']=='f']

        female_data.columns = [i.replace('ptr_', '') if 'ptr_' in i or i == 'positive_response' else 'ptr_' + i for i
                                  in female_data.columns]
        data = pd.concat([male_data, female_data])
        data = data.drop(['mem_no', 'ptr_mem_no', 'mem_sex', 'ptr_mem_sex'], axis=1)

        data['age_gap'] = (data['age'] - data['ptr_age'])

        data['loc_like'] = data.apply(lambda x: 1 if x['mem_loc'] == x['ptr_mem_loc'] else 0, axis=1)
        data['marriage_like'] = data.apply(lambda x: 1 if x['mate_slct'] == x['ptr_mate_slct'] else 0, axis=1)
        data.drop(['mem_loc', 'mate_slct', 'ptr_mem_loc', 'ptr_mate_slct'], axis=1, inplace=True)

        X = data.drop('positive_response', axis=1)
        y = data['positive_response']

        return X, y
