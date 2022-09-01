from utils.preprocess.dbconnector import data_path
import pandas as pd
df = pd.read_csv(f'{data_path}df.csv')
class CF_score :
    def __init__(self):
        pass

    def get_CF_score(self, x, k=1): # input service user x, default top_k=1
        # age limit +- 5
        x_age = df[df['mem_no']==x].age.values[0]
        x_gender = df[df['mem_no']==x].mem_sex.values[0]
        # recommendation candidate y range(age +- 5)
        y_range = df[(abs(x_age - df['age']) <= 5) & (df['mem_sex']!=x_gender)].mem_no.values
        print('y range 제거 전 :', len(y_range))
        y_range = [i for i in y_range if i not in eval(df[df['mem_no']==x].view_list.values[0])]
        print(f'number of candidates for service user {x} : {len(y_range)}')

        recs = list() # initialize Recs
        sent_to_x = eval(df[df['mem_no']==x].sent_list.values[0]) # SentTO_x

        # loop for every y in RecommendationCandidates
        for y in y_range :
            sent_to_y = eval(df[df['mem_no']==y].sent_list.values[0]) # SentTO_y
            score_x_y = 0 # initialize score_x_y
            for u in sent_to_y : # loop for every u in SentTo_y
                try :
                    u_refrom = eval(df[df['mem_no']==u].send_list.values[0]) # ReFrom_u
                    x_refrom = eval(df[df['mem_no']==x].send_list.values[0]) # ReFrom_x
                    intersection = list(set(u_refrom) & set(x_refrom)) # ReFrom_u ∩ ReFrom_x
                    union = list(set(u_refrom) | set(x_refrom)) # ReFrom_u ∪ ReFrom_x

                    if len(intersection) == 0 :
                        continue
                    # score_x_y <-- score_x_y + similarity_x_y
                    score_x_y += len(intersection) / len(union)
                except :
                    pass

            score_y_x = 0 # initialize score_y_x
            for v in sent_to_x : # loop for every v in SentTO_x
                try :
                    v_refrom = eval(df[df['mem_no']==v].send_list.values[0]) # ReFrom_v
                    y_refrom = eval(df[df['mem_no']==y].send_list.values[0]) # Refrom_y
                    intersection = list(set(v_refrom) & set(y_refrom)) # ReFrom_v ∩ ReFrom_y
                    union = list(set(v_refrom) | set(y_refrom)) # ReFrom_v ∪ ReFrom_y

                    if len(intersection) == 0 :
                        continue
                    # score_y_x <-- score_y_x + similarity_y_v
                    score_y_x += len(intersection) / len(union)
                except :
                    pass

            # calculate harmonic mean
            if score_x_y != 0 and score_y_x != 0 :
                harmonic_mean_score = (2 * score_x_y * score_y_x) / (score_x_y + score_y_x)
            else :
                harmonic_mean_score = 0

            recs.append((y, harmonic_mean_score)) # Recs <--- (y, reciprocalScore_x,y)
        if max([i[1] for i in recs]) == 0 :
            top_k = recs
        else :
            top_k = sorted(recs, key=lambda x : -x[1])[:k]

        return top_k # sort score with descending order and return top-k
