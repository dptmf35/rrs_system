
import pandas as pd
from utils.config.common import conf


data = pd.read_pickle(f'{conf.dataPath}pr_base_df.pkl')
print(data.sort_values(by='send_list_count',ascending=False)['mem_no'].values[1:30])