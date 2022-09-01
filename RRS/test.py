from utils.db_connector import DBConnector
import pymysql
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from utils.config.db_configs import *
import warnings
warnings.filterwarnings('ignore')
dbcon = DBConnector()


dbcon.save_sql()

# conn = pymysql.connect(
#     user='my_srv',
#     passwd='wkrldi@duqhdi12',
#     host='125.141.223.156',
#     db='m_yeoboya_ai_dv',
#     charset='utf8',
#     port=33141
# )
#
# # member_msg_log_back
# cursor = conn.cursor(pymysql.cursors.DictCursor)
# sql = f"select * from m_yeoboya.member_msg_log_back where ins_date between '{yesterday}' and '{today}'"
# print(f"select * from m_yeoboya.member_msg_log_back where ins_date between '{yesterday}' and '{today}'")
# cursor.execute(sql)
# result = cursor.fetchall()
# cursor.close()
# msg_df = pd.DataFrame(result)
# print(msg_df.head())

