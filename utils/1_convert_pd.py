import pickle
import pandas as pd
# 数据说明见: http://jmcauley.ucsd.edu/data/amazon/
# 或者raw_data中的data_readme.txt

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index') # 从字典中，逐行加载数据
    return df

reviews_df = to_df('../raw_data/reviews_Electronics_5_1w.json') # 本例中用1w条样本做演示
with open('../raw_data/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../raw_data/meta_Electronics_1w.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())] # 只取那些在评论里存在的meta数据
meta_df = meta_df.reset_index(drop=True)  # 将以前的行label去掉，只用行的下标作为索引
with open('../raw_data/meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
