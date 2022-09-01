import pandas as pd
from utils.config.common import data_path
df = pd.read_pickle(f"{data_path}explanation_base_df.pkl")


class Reciprocal_Explanation() :
    def __init__(self):
        pass

    def get_reciprocal_explanations_for_y_list(self, x, y_list):
        reciprocal_exp_list = list()

        for y in y_list :
            reciprocal_exp = self.get_reciprocal_explanations(x, y)
            reciprocal_exp_list.append(reciprocal_exp)

        return reciprocal_exp_list

    def get_reciprocal_explanations(self, x, y):
        exp_x_r = self.get_explanation(x, y)
        exp_y_r = self.get_explanation(y, x)

        reciprocal_exp = f"""{x}에게 {y}가 추천된 이유 : {exp_x_r},
                        {y}에게 {x}가 추천된 이유 : {exp_y_r}"""

        return reciprocal_exp

    def get_explanation(self, x, y, k=3):
        view_list = df[df['mem_no']==x].view_list.values[0]
        send_list = df[df['mem_no']==x].send_list.values[0]

        # m_x
        m_x = [1 if i in send_list else 0 for i in view_list]
        a_v_df= pd.DataFrame([m_x], columns=view_list, index=[x]).T

        # S_x_a_v
        for col in df.columns[1:] :
            S_x_a_v = [df[df['mem_no']==y][col].values[0] for y in a_v_df.index]
            a_v_df[col] = S_x_a_v

        a_v_df = pd.get_dummies(a_v_df)
        y_a_v = pd.get_dummies(df[df['mem_no']==y])

        for y_col in y_a_v.columns :
            if y_col in a_v_df.columns and y_a_v[y_col].values[0] != 1 :
                a_v_df.drop(y_col, inplace=True, axis=1)

        a_v_df.drop([c for c in a_v_df.columns if c not in y_a_v.columns and c!=x], axis=1, inplace=True)

        result = a_v_df.corr()[x].sort_values(ascending=False)
        print(f"Highest correlation features for {x} : {[result.index[1]]}, Correlation value : {result.values[1]}")
        return result.index[1:k+1].tolist()

