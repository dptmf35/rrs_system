from algorithms.get_PR_score import PR_score
from algorithms.get_CF_score import CF_score
from algorithms.get_reciprocal_explanation import Reciprocal_Explanation
from utils.config.common import conf
import pandas as pd
import random
import time


class ReciprocalRecommenderSystem(PR_score, CF_score, Reciprocal_Explanation) :
    def __init__(self):
        super(ReciprocalRecommenderSystem, self).__init__()

    def get_RRS(self, x, top_k=1):
        top_n = self.predict_positive_reply(x)
        result = self.get_cf_score(x, y_range=top_n, top_k=top_k)
        explanations = self.get_reciprocal_explanations_for_y_list(x, result)

        return result, explanations



if __name__ == '__main__' :
    # Test For random 1
    temp = pd.read_pickle(f"{conf.dataPath}cf_base_df.pkl")
    temp = temp.sort_values(by='send_list_count',ascending=False).head(100).reset_index(drop=True)
    mem_no = temp.loc[random.choice(range(100)), 'mem_no']
    print(mem_no)
    rrs = ReciprocalRecommenderSystem()

    start = time.time()
    res, exp = rrs.get_RRS(mem_no)
    print(res, exp)

    print(f"Time Elapsed for Computing Top 1 : {time.time() - start :.3f}s")