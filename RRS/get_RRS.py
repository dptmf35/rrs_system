from algorithms.get_PR_score import PR_score
from algorithms.get_CF_score import CF_score
from algorithms.get_reciprocal_explanation import Reciprocal_Explanation

class get_RRS(PR_score, CF_score, Reciprocal_Explanation) :
    def __init__(self):
        super(get_RRS, self).__init__()

    def get_RRS(self, x):
        top_n = self.predict_positive_reply(x)
        result = self.get_cf_score(x, y_range=top_n)
        explanations = self.get_reciprocal_explanations_for_y_list(x, result)

        return result, explanations

