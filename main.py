import scipy.io as sio
import funcs as fc
from importlib import reload

folder = 'C:/Users/saul6/Documents/Electrooptical Eng/Brain waves signal processing/Asignment/10studies/python/data/'
# files = ["02", "05", "07", "08", "10", "15", "18", "20", "23", "25"]  # all

KC_thresh = 40
SP_thresh = 15
epoch = 30
trainORtest = "train"  ##########################-- choose run mode here --########################

if trainORtest == "train":

    files = ["05", "07", "08", "15", "18", "23", "25"]  # train
    trained_features = fc.analyze_data(folder, files, KC_thresh, SP_thresh, epoch, save=True, plot=True)

    #sio.savemat('train.mat', {'res': trained_features, 'KC_thresh': KC_thresh, 'SP_thresh': SP_thresh, 'epoch': epoch})  # save results to file to create train data

    #clf = fc.SVM_train(features)  # use calculated features to create clf

elif trainORtest == "test":

    files = ["20"]
    #files = ["02", "10", "20"]  # test
    features = fc.analyze_data(folder, files, KC_thresh, SP_thresh, epoch, save=True, plot=True)

    trained_features = sio.loadmat('train.mat')['res']  # load trained data

    clf = fc.SVM_train(trained_features)  # use calculated features to create clf
    hyp = fc.SVM_test(clf, file="20", plot=True)  # use clf to estimate hypnogram

else:

    dat = sio.loadmat('25.mat')['res']  # load data file
    fc.plotit('25', dat)  # plots all features


"""
dat = sio.loadmat(folder + '2602_py.mat')['data_10hz']

dat = sio.loadmat('20.mat')['res']

# fc = reload(fc)

# Wake: Beta + Alpha (0)
# S1: Low amplitude Theta + some alpha (1)
# S2: Intermediate amplitude Theta + Rare alpha + Spindles + K-complex (2)
# S3: High amplitude Delta + Spindles (3)
# REM: Theta + Some lower freq alpha (5)


"""