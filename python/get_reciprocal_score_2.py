from utils.preprocess.dbconnector import data_path
import pandas as pd
import numpy as np
import time
import joblib
feature_df = pd.read_csv(f'{data_path}feature_df.csv')
df = pd.read_csv(f'{data_path}df.csv')
model = joblib.load(f'{data_path}predpr.pkl')

x = 1697018

start = time.time()

def make_df(x, feature_df = feature_df) :
    feature_df = feature_df.rename(columns={'send_list_count': 'send_count', 'view_list_count': 'view_count',
                                                'sent_list_count': 'sent_count',
                                                'viewed_list_count': 'viewed_count'})
    x_features = feature_df[feature_df['mem_no'] == x]
    # age limit +- 5
    x_age = df[df['mem_no'] == x].age.values[0]
    x_gender = df[df['mem_no'] == x].mem_sex.values[0]
    # recommendation candidate y range(age +- 5)
    y_range = df[(abs(x_age - df['age']) <= 5) & (df['mem_sex'] != x_gender)].mem_no.values


    y_features = feature_df[feature_df['mem_no'].isin(y_range)]

    if feature_df[feature_df['mem_no'] == x].mem_sex.values[0] == 'f':
        x_features.columns = ['ptr_' + c for c in x_features.columns]

    else:
        y_features.columns = ['ptr_' + c for c in y_features.columns]

    x_y_features = pd.concat([x_features.reset_index(drop=True), y_features.reset_index(drop=True)], axis=1)

    if x_gender == 'm' :
        candidate_list = x_y_features['ptr_mem_no'].values
    else :
        candidate_list = x_y_features['mem_no'].values

    x_y_features.drop(['mem_no', 'ptr_mem_no', 'mem_sex', 'ptr_mem_sex'], axis=1, inplace=True)
    x_y_features['age_gap'] = x_y_features['age'] - x_y_features['ptr_age']
    x_y_features.fillna(method='ffill', inplace=True)



    return x_y_features, candidate_list

x_y_features, candidate_list = make_df(x)
# print(x_y_features.head())

def get_PR_score(x_y_features, candidate_list) :  # predict positive reply between user x, y

    pred = model.predict_proba(x_y_features)
    pred = pd.DataFrame(pred, columns=['0', '1'])
    candi = pd.DataFrame([candidate_list]).T
    candi.columns=['candi_no']
    result = pd.concat([candi, x_y_features, pred], axis=1)
    result = result.sort_values('1', ascending=False)[:50]
    return result[['candi_no','1']]

pr_result = get_PR_score(x_y_features, candidate_list)

print(f'elapsed time : {time.time()-start:.4f}')

pr_result['mem_no'] = x

def get_cf_score(x, y):
    sent_to_x = eval(df[df['mem_no'] == x].sent_list.values[0])
    sent_to_y = eval(df[df['mem_no'] == y].sent_list.values[0])
    score_x_y, score_y_x = 0, 0
    for u in sent_to_y:
        try:
            u_refrom = eval(df[df['mem_no'] == u].send_list.values[0])
            x_refrom = eval(df[df['mem_no'] == x].send_list.values[0])
            intersection = list(set(u_refrom) & set(x_refrom))
            union = list(set(u_refrom) | set(x_refrom))
            if len(intersection) == 0:
                continue
            score_x_y += len(intersection) / len(union)
        except:
            pass

    for v in sent_to_x:
        try:
            v_refrom = eval(df[df['mem_no'] == v].send_list.values[0])
            y_refrom = eval(df[df['mem_no'] == y].send_list.values[0])
            intersection = list(set(v_refrom) & set(y_refrom))
            union = list(set(v_refrom) | set(y_refrom))
            if len(intersection) == 0:
                continue
            score_y_x += len(intersection) / len(union)
        except Exception as e:
            pass
    #     print(score_x_y, score_y_x)

    if score_x_y > 0 and score_y_x > 0:
        harmonic_mean_score = (2 * score_x_y * score_y_x) / (score_x_y + score_y_x)
    else:
        harmonic_mean_score = 0

    return harmonic_mean_score

pr_result['cf_score'] = pr_result.apply(lambda x : get_cf_score(x['mem_no'],x['candi_no']),axis=1)



print(pr_result[['mem_no','candi_no','cf_score','1']])
print(f'elapsed time : {time.time()-start:.4f}')