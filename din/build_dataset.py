import random
import pickle

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f) # 读取时，多次读取pkl数据
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist() # 每个key所对应的value group成list
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))] # 为每个正样本生成负样本

  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != len(pos_list) - 1:
      train_set.append((reviewerID, hist, pos_list[i], 1)) # 用它们前面的历史信息预测后面
      train_set.append((reviewerID, hist, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
