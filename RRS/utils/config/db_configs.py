import pymysql
from datetime import datetime, timedelta
data_path='./utils/data/'


today = str(datetime.today())[:10]
yesterday = str(datetime.today() - timedelta(1))[:10]

conn = pymysql.connect(
            user='my_srv',
            passwd='wkrldi@duqhdi12',
            host='125.141.223.156',
            db='m_yeoboya_ai_dv',
            charset='utf8',
            port=33141
        )
