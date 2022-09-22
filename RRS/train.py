from utils.preprocess.get_base_dataframe import getBaseDataFrame
from utils.config.common import conf
from algorithms.train.train_pr_model import TrainPRModel
bdf = getBaseDataFrame()

# make backup files by sql query
bdf.save_sql()

# make base dataframe
df = bdf.make_base_dataframe()

# pr base dataframe
pr_base = df[['mem_no', 'mem_sex', 'age', 'mem_loc', 'mate_religion', 'mate_car', 'mate_ann_salary',
              'mate_career', 'possess_property', 'mate_height', 'smoke_slct', 'drink_slct', 'health_slct',
              'mate_slct', 'upd_cnt', 'photo_cnt', 'conts_upd_cnt','join_cnt', 'login_cnt', 'tot_stay_time',
              'view_list', 'viewed_list', 'view_list_count','viewed_list_count', 'send_list', 'sent_list',
              'send_list_count','sent_list_count', 'char_type_1', 'char_type_2', 'char_type_3', 'char_type_4',
              'hobby_type_1', 'hobby_type_2', 'hobby_type_3', 'hobby_type_4']]
pr_base.to_pickle(f"{conf.dataPath}pr_base_df.pkl")

# cf base dataframe
cf_base = df[['mem_no', 'mem_sex', 'age', 'view_list', 'viewed_list', 'view_list_count',
              'viewed_list_count', 'send_list', 'sent_list', 'send_list_count', 'sent_list_count']]
cf_base.to_pickle(f"{conf.dataPath}cf_base_df.pkl")

# explanation base dataframe
corr_base = df[['mem_no', 'mem_sex', 'mate_religion', 'mate_car', 'mate_job', 'mate_ann_salary',
               'mate_career', 'mate_style', 'possess_property','drink_slct','health_slct',
                'mate_height', 'mate_slct', 'char_type_1', 'char_type_2', 'char_type_3', 'char_type_4', 'food_type_1',
              'food_type_2', 'food_type_3', 'food_type_4', 'food_type_5', 'food_type_6', 'food_type_7',
              'food_type_8', 'food_type_9', 'hobby_type_1', 'hobby_type_2', 'hobby_type_3', 'hobby_type_4',
                'view_list', 'send_list']]
corr_base.to_pickle(f"{conf.dataPath}corr_df.pkl")

# Train Model and save pickle file
tpr = TrainPRModel()
model = tpr.save_model()
print(model.feature_names_)