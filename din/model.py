import tensorflow as tf

from Dice import dice

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list):

    self.user_context = tf.placeholder(tf.int32, [None, ]) # [B], user_context
    self.i = tf.placeholder(tf.int32, [None,]) # [B], candidate ad items,即待预测的商品item
    self.j = tf.placeholder(tf.int32, [None,]) # [B], negative_items,只在evaluate时才会用到
    self.label = tf.placeholder(tf.float32, [None, ]) # [B], label
    self.history_item_sequence = tf.placeholder(tf.int32, [None, None]) # [B, T], 历史商品的序列,[batch,seq_length]
    self.sequence_length = tf.placeholder(tf.int32, [None, ]) # [B], 当前batch中的最长序列的长度
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])  # [user_count, hidden]
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) # [item_count, hidden/2]
    item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) # [item_count]
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2]) # [cate_count, hidden/2]
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64) # 访问商品的类目id, 407*18

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.user_context) # [batch , hidden]

    i_cate = tf.gather(cate_list, self.i) # gather与embedding_lookup类似, [batch, X]
    i_emb = tf.concat(values = [ # batch* (hidden/2+ X) => [batch, hidden]
        tf.nn.embedding_lookup(item_emb_w, self.i), # batch* hidden/2
        tf.nn.embedding_lookup(cate_emb_w, i_cate), # batch* X，从后文中看，X = hidden/2
        ], axis=1)
    i_bias = tf.gather(item_b, self.i)  # [batch]

    # history_cate_seq: [batch, seq_len, X]
    history_cate_seq = tf.gather(cate_list, self.history_item_sequence) # cate_list: cate_num*X, history_item_seq: batch*seq_len
    # history_embed: [batch, seq_len, hidden]
    history_embed = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.history_item_sequence), # [batch, seq_len, hidden/2]
        tf.nn.embedding_lookup(cate_emb_w, history_cate_seq), # [batch, seq_len, hidden/2]
        ], axis=2)

    # history_attend: [batch,1,hidden]
    user_history_attend = attention(i_emb, history_embed, self.sequence_length) # i_emb:[batch,hidden] , history_embed:[batch,seq_len, hidden]

    #-- attention end ---
    user_history_attend = tf.layers.batch_normalization(inputs = user_history_attend)
    user_history_attend = tf.reshape(user_history_attend, [-1, hidden_units]) # [batch,hidden]
    user_history_attend = tf.layers.dense(user_history_attend, hidden_units) # 接入一个全连接层，输出为 hidden, [batch,hidden]

    u_emb = user_history_attend # [batch, hidden]
    print(u_emb.get_shape().as_list())
    print(i_emb.get_shape().as_list())
    #-- fcn begin -------
    # for i, 将user_embed, item_embed 连接起来,只用了user_id,history_item_id的信息，并未使用user的其它信息，应该可以加入进来
    din_i = tf.concat([u_emb, i_emb], axis=-1) # [batch, hidden]
    din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
    d_layer_1_i = tf.layers.dense(din_i, units=80, activation=tf.nn.sigmoid, name='f1')
    #if you want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
    #d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
    #d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
    d_layer_2_i = tf.layers.dense(d_layer_1_i, units=40, activation=tf.nn.sigmoid, name='f2')
    #d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3') # [batch, 1]
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1]) # [batch]

    self.logits = i_bias + d_layer_3_i # [batch], 最后的logits，用来计算交叉熵损失函数
    self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label))

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    # --------------------
    # for j
    j_cate = tf.gather(cate_list, self.j) #  [batch, X]
    j_emb = tf.concat([ # batch* (hidden/2+ X)
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, j_cate),
    ], axis=1)

    print(j_emb.get_shape().as_list())

    j_bias = tf.gather(item_b, self.j) # [batch]
    din_j = tf.concat([u_emb, j_emb], axis=-1) # [batch, hidden+hidden]
    din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
    d_layer_1_j = tf.layers.dense(din_j, units=80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    #d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
    d_layer_2_j = tf.layers.dense(d_layer_1_j, units=40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    #d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
    d_layer_3_j = tf.layers.dense(d_layer_2_j, units=1, activation=None, name='f3', reuse=True) # [batch, 1]
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1]) # [batch]

    x = i_bias - j_bias + d_layer_3_i - d_layer_3_j # [B]
    u_emb_all = tf.expand_dims(u_emb, 1) # [batch,1,hidden]
    u_emb_all = tf.tile(u_emb_all, [1, item_count, 1]) # [batch, item_count, hidden]

    # logits for all item:
    all_emb = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    all_emb = tf.expand_dims(all_emb, 0)
    all_emb = tf.tile(all_emb, [512, 1, 1])
    din_all = tf.concat([u_emb_all, all_emb], axis=-1)
    din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
    d_layer_1_all = tf.layers.dense(din_all, units=80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    #d_layer_1_all = dice(d_layer_1_all, name='dice_1_all')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, units=40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    #d_layer_2_all = dice(d_layer_2_all, name='dice_2_all')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, units=1, activation=None, name='f3', reuse=True)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
    self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
    #-- fcn end -------

    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_bias + d_layer_3_i)
    self.score_j = tf.sigmoid(j_bias + d_layer_3_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.positive_and_negative = tf.concat([self.score_i, self.score_j], axis=-1)
    print(self.positive_and_negative.get_shape().as_list())

  def train(self, sess, uij, l):
    # uij: user_context, pos_or_neg_item(candidate_item), label, history_item_seq, sequence_length
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.user_context: uij[0], # user_id
        self.i: uij[1], # candidate_item`
        self.label: uij[2], # label
        self.history_item_sequence: uij[3], # history_item_sequence
        self.sequence_length: uij[4], # sequence_length
        self.lr: l,
        })
    return loss

  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.positive_and_negative], feed_dict={
        self.user_context: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.history_item_sequence: uij[3],
        self.sequence_length: uij[4],
        })
    return u_auc, socre_p_and_n

  def test(self, sess, uid, hist_i, sl):
    return sess.run(self.logits_all, feed_dict={
        self.user_context: uid,
        self.history_item_sequence: hist_i,
        self.sequence_length: sl,
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def attention(queries, keys, keys_length):
  '''
    queries:     [B, H], B:batch, H:hidden, T:seq_length
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]]) # [batch, hidden*seq_length]
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units]) # [batch, seq_length, hidden]
  # queries * keys: 是内积吧，而不是论文中所说的外积
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1) # [batch,seq_length, hidden+hidden+hidden+hidden]
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att') # [batch, seq_length, 1]
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]]) # [batch, 1, seq_length]
  outputs = d_layer_3_all  # [batch, 1, seq_length]
  # Mask
  # 将batch中的每一行都填充成seq_length的长度，其中true的个数为keys_length中的值
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T], keys_length:[batch], tf.shape[keys][1]: seq_length
  key_masks = tf.expand_dims(key_masks, 1) # [batch, 1, seq_length]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # 全填成最小值
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T], 条件选择，true的话，选择outputs,false则选择paddings,感觉这里用 bool_mask 就行了吧

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5) # output/(hidden_size**0.5)

  # Activation
  weigths = tf.nn.softmax(outputs)  # [B, 1, T], 在时间维度上进行softmax选择,输出权重

  # Weighted sum
  # weigths:[batch, 1, seq_length],
  # keys:[batch, seq_length, hidden]
  # weighted outputs: [batch, 1, hidden]
  weighted_outputs = tf.matmul(weigths, keys)  # [B, 1, H], 3-d矩阵相乘，就是第0维要相同，去掉第0维后，里面都是普通二级矩阵相乘, 在time_length维度进行相乘及加权求和，非常巧妙,妙！

  # weighted outputs: [batch, 1, hidden]
  return weighted_outputs

