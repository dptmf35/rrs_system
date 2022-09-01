from utils.preprocess.dbconnector import data_path
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(f'{data_path}df.csv')
feature_df = pd.read_csv(f'{data_path}feature_df.csv')
model = joblib.load(f'{data_path}predpr.pkl')

class WeightOptimizer() :
    def __init__(self):
        pass

    def find_optimal_weight(self, x):

        send_list = df[df['mem_no'] == x]['send_list'].values[0]
        sent_list = df[df['mem_no'] == x]['sent_list'].values[0]

        SuccInter_x = [i for i in send_list if i in sent_list]  # SuccInter_x
        V_x = send_list  # V_x

        weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        results = []
        for a in weights:
            score_list = []
            for y in V_x:
                if y not in df.mem_no.values:
                    continue
                reciprocal_score = (self.get_cf_score(x, y) * a) + (self.pred_positive_reply(x, y)[1] * (1 - a))
                score_list.append((y, reciprocal_score))

            rank = sorted(score_list, key=lambda x: -x[1])[:len(SuccInter_x)]
            rank_succ = [r[0] for r in rank if r[0] in SuccInter_x]  # hit pred succ = real succ

            hit_rate = len(rank_succ) / len(SuccInter_x)

            if hit_rate == 1:  # return weight a with highest hit rate
                print(f'found optimal weight : {a}')
                return a
            else:
                results.append(hit_rate)

        else:
            opt_index = np.argmax(results)
            print(f'found optimal weight : {weights[opt_index]}')
            return weights[opt_index]

    def pred_positive_reply(x, y):
        x_info = feature_df[feature_df['mem_no'] == x]
        y_info = feature_df[feature_df['mem_no'] == y]
        if feature_df[feature_df['mem_no'] == x].mem_sex.values[0] == 'f':
            x_info.columns = ['ptr_' + c for c in x_info.columns]
        else:
            y_info.columns = ['ptr_' + c for c in y_info.columns]
        x_y_info = pd.concat([y_info.reset_index(drop=True), x_info.reset_index(drop=True)], axis=1)
        x_y_info.drop(['mem_no', 'ptr_mem_no', 'mem_sex', 'ptr_mem_sex'], axis=1, inplace=True)
        x_y_info['age_gap'] = x_y_info['age'] - x_y_info['ptr_age']
        pred = model.predict_proba(x_y_info)[0]
        idx = np.argmax(pred)
        return idx, pred[idx]

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


if __name__ == '__main__' :
    opt = WeightOptimizer()
    x = 1697018
    opt.find_optimal_weight(x)