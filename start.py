import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

#数据处理
train_sales  = pd.read_csv('train_sales_data.csv')
train_search = pd.read_csv('train_search_data.csv')
train_user   = pd.read_csv('train_user_reply_data.csv')
evaluation_public = pd.read_csv('evaluation_public.csv')
submit_example    = pd.read_csv('submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True) # 将要预测的数据与训练集列拼接
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth']) # 将搜索量数据进行拼接
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth']) #将媒体评价数量与车型评价数据进行拼接
data['label'] = data['salesVolume'] # 销量做标签
data['id'] = data['id'].fillna(0).astype(int) # 将id列空白用0填充并且转化为int
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
# LabelEncoder 将文本数据转化为数值
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']

#提取特征
def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model'] #省份编码+车型编码  识别
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in tqdm(['label','popularity']): #tqdm函数可以显示进度条
        # 平移12个月
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    return df,stat_feat

#评价指标
def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

#模型选择
def get_model_type(train_x,train_y,valid_x,valid_y,m_type='x gb'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
            num_leaves=2**5-1, # 叶子节点数
            reg_alpha=0.25, # L1正则化项权重
            reg_lambda=0.25, # L2正则化项权重
            objective='mse', # 学习任务和要使用的相应学习目标或目标自定义函数
            max_depth=-1, #最大深度 <0 为无限制
            learning_rate=0.05, #学习速率
            min_child_samples=5, # 子叶中所需要的最小数据数
            random_state=2019, # 随机数种子
            n_estimators=2000, # 要适应的增强树数量
            subsample=0.9,
            colsample_bytree=0.7, # 构造每棵树时列的子采样率
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              categorical_feature=cate_feat,
              early_stopping_rounds=100, verbose=100)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=2000,
            objective='reg:gamma',
            tree_method='hist',
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_samples=5,
            eval_metric='rmse'
        )
        model.fit(train_x, train_y,
                  eval_set=[(train_x, train_y), (valid_x, valid_y)],
                  early_stopping_rounds=100, verbose=100)
    return model

#模型训练
def get_train_model(df_, m, m_type='xgb'):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx   = (df['mt'].between(st , m-1))
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m  , m  ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st ,m-5)
    print('valid_idx:',m-4,m-4)
    print('test_idx :',m  ,m  )
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type)
    # offline
    df['pred_label'] = model.predict(df[features])
    best_score = score(df[valid_idx])
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['label'])

    df['forecastVolum'] = model.predict(df[features])
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    return sub,df[valid_idx]['pred_label']

# 逐步预测
for month in [25, 26, 27, 28]:
    m_type = 'xgb'

    data_df, stat_feat = get_stat_feature(data)

    num_feat = ['regYear'] + stat_feat

    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    features = num_feat + cate_feat
    print(len(features), len(set(features)))

    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
sub[['id', 'forecastVolum']].round().astype(int).to_csv('xgb.csv', index=False)