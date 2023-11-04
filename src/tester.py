import time
from VLAD_finder import VLAD

vlad = VLAD()
train_start = time.time()
vlad.train()
train_end = time.time()
median= vlad.query()
print('median: ', median)
print('querytime(sec): ', time.time() - train_end)
print('traintime(min): ', (train_end - train_start)/60)
