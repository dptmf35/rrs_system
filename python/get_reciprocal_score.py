from get_cf_score import CF_score
from get_pr_score import PR_score
from get_reciprocal_explanaton import ReciprocalExplanation
import time
import pandas as pd

temp = pd.read_csv('./multi_thread_test/temp.csv')


class ReciprocalScore(CF_score, PR_score, ReciprocalExplanation):
    def __init__(self):
        super(ReciprocalScore, self).__init__()

    def get_reciprocal_score(self, x, a=0.3978, k=1):
        reciprocal_score = []

        # 1. get CF score
        cf_score = self.get_CF_score(x, k=5)
        # print(cf_score)
        # 2. get PR score
        for y, cf_score in cf_score :
            pr_score = self.get_PR_score(x, y)[1]
            weighted_score = (cf_score * a) + (pr_score * (1 - a)) # add weight(α)
            reciprocal_score.append((int(y), float(weighted_score)))

        # get Reciprocal Score
        top_k = sorted(reciprocal_score, key=lambda x : -x[1])

        # get reciprocal explanation for x, y(candidate)
        explanations = []
        for i in range(k) :
            y = top_k[i][0]
            explanations.append(self.get_reciprocal_explanation(x, y))
        # print('=' * 40)
        # print(f'RECOMMENDATION RESULT for {x} :', end='')
        return top_k[:k], explanations

    def test(self, x, a=0.3978, k=1):
        reciprocal_score = []
        # 1. get CF score
        cf_scores = eval(temp[temp['mem_no']==x].cf_scores.values[0])
        print(cf_scores[0])

        # 2. get PR score
        for y, cf_score in cf_scores:
            pr_score = self.get_PR_score(x, y)[1]
            weighted_score = (cf_score * a) + (pr_score * (1 - a))  # add weight(α)
            reciprocal_score.append((int(y), float(weighted_score)))

        # get Reciprocal Score
        top_k = sorted(reciprocal_score, key=lambda x: -x[1])

        # get reciprocal explanation for x, y(candidate)
        explanations = []
        for i in range(k):
            y = top_k[i][0]
            explanations.append(self.get_reciprocal_explanation(x, y))
        # print('=' * 40)
        # print(f'RECOMMENDATION RESULT for {x} :', end='')
        return top_k[:k], explanations

if __name__ == '__main__' :
    rpc = ReciprocalScore()
    x = 1697018

    # start = time.time()
    # top_k, explanation = rpc.test(x, k=1)
    # print('=' * 40)
    # print(top_k, explanation)
    # print('=' * 40)
    # print(f'elapsed time : {time.time() - start}')

    start = time.time()
    top_k, explanation = rpc.get_reciprocal_score(x, k=1)
    print('=' * 40)
    print(top_k, explanation)
    print('=' * 40)
    print(f'elapsed time : {time.time() - start}')