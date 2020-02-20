from sklearn import svm
import scipy.io as sio
import numpy as np
import funcs as fc

raw_data = sio.loadmat('train.mat')
train = raw_data['res']
raw_data = sio.loadmat('20.mat')
test = raw_data['res']

train[:, 0:7] -= train[:, 0:7].mean(axis=0)  # reduce average from train
train[:, 0:7] = train[:, 0:7] / np.std(train[:, 0:7], axis=0)  # devide train by STD

test[:, 0:7] -= test[:, 0:7].mean(axis=0)  # reduce average from train
test[:, 0:7] = test[:, 0:7] / np.std(test[:, 0:7], axis=0)  # devide train by STD

clf = svm.SVC()
clf.fit(train[:, 0:7], train[:, 7])

result = clf.predict(test[:, 0:7]).reshape(-1, 1)

fc.plot_hypnogram(test[:, 7].reshape(-1, 1), result)

#  PSD: Delta, Theta, Alpha, Beta

# Wake: Beta + Alpha (0)
# S1: Low amplitude Theta + some alpha (1)
# S2: Intermediate amplitude Theta + Rare alpha + Spindles + K-complex (2)
# S3: High amplitude Delta + Spindles (3)
# REM: Theta + Some lower freq alpha (5)
