from utils.preprocess.get_base_dataframe import getBaseDataFrame
import pandas as pd

bdf = getBaseDataFrame()



if __name__ == '__main__' :

    pr_base = bdf.pr_base_dataframe()
    pr_base.to_pickle("pr_base_df.pkl")
    cf_base = bdf.get_cf_base_dataframe()
    cf_base.to_pickle("cf_base_df.pkl")
    corr_base = bdf.corr_base_transform()
    corr_base.to_pickle("corr_df.pkl")


