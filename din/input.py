import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def next(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, y, sl = [], [], [], []
    # dataset里的每一行(reviewerID, hist_item_seq, pos_or_neg_item, label)
    for t in ts:
      u.append(t[0]) # user_id
      i.append(t[2]) # pos_or_neg_item
      y.append(t[3]) # label
      sl.append(len(t[1])) # history_item_seq_len
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1
    # user_context(user_id), pos_or_neg_item, label, hist_item_seq, sequence_length
    return self.i, (u, i, y, hist_i, sl)

class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1 # 没有除尽，应该多一个epoch
    self.i = 0

  def __iter__(self):
    return self

  def next(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, j, sl = [], [], [], []
    # dataset里的每一行(reviewerID, hist_item_seq, (pos_item, neg_item))
    for t in ts:
      u.append(t[0]) # reviewerID
      i.append(t[2][0]) # pos_item
      j.append(t[2][1]) # neg_item
      sl.append(len(t[1])) # history_item_seq_length
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, j, hist_i, sl)
