import scipy.io as sio
import numpy as np
from scipy import signal as sg
import time
import funcs as fc
from importlib import reload


start_time = time.time()
folder = 'C:/Users/saul6/Documents/Electrooptical Eng/Brain waves signal processing/Asignment/10studies/python/data/'
#file = ["02", "05", "07", "08", "10", "15", "18", "20", "23", "25"]
#file = ["02", "05", "07", "08", "10", "15", "18"]  # train
file = ["20", "23", "25"]  # test
#file = ["07"]

for file_num in file:
    path = folder + '26' + file_num + '_py.mat'
    raw_data = sio.loadmat(path)
    GT = raw_data['GT']  # Ground trouth
    sos = sg.butter(6, 99.99/(200/2), btype="low", output='sos')    # prepare butterworth filter to down-sample signal by factor of 2 (from 200Hz to 100Hz)
    sig = sg.sosfiltfilt(sos, raw_data['data_200hz'], axis=0)[::2]  # perform LPF and then down-sample signal to 100 Hz

    KC_thresh = 40  # [uV]
    spindle_thresh = 15  # [uV]
    epoch = 30  # each segment length [sec]
    fs = 100  # sampling rate [Hz]

    PSD = fc.power_calc(sig[:, 0:6], fs, epoch, low=0.5, high=3.5, avg=True)  # Delta waves PSD
    PSD = np.append(PSD, fc.power_calc(sig[:, 0:6], fs, epoch, low=4, high=7, avg=True), axis=1)  # Theta waves PSD
    PSD = np.append(PSD, fc.power_calc(sig[:, 0:6], fs, epoch, low=7.5, high=12, avg=True), axis=1)  # Alpha waves PSD
    PSD = np.append(PSD, fc.power_calc(sig[:, 0:6], fs, epoch, low=15.5, high=30, avg=True), axis=1)  # Beta waves PSD

    REM = fc.REM_finder(sig[:, 7:9], fs, epoch)  # calculate left & right eyes signal power in epochs

    KC_raw, KC = fc.k_complex(sig[:, 0:6], fs, epoch, KC_thresh, folder)  # calculates K-complex activity

    SP_raw, SP = fc.spindle(sig[:, 0:6], fs, epoch, spindle_thresh, folder)  # calculates spindle activity

    if 'res' in locals():  # checks if res exists
        res = np.append(res, fc.prepare_mat(PSD, REM, KC, SP, GT), axis=0)  # creates res at first itteration
    else:
        res = fc.prepare_mat(PSD, REM, KC, SP, GT)  # if res doesn't exist- create it

    #results = fc.s_stage(PSD, REM, KC, SP)

    #fc = reload(fc)
    fc.plot_hypnogram(GT[:, 1])

    #fc = reload(fc)
    #fc.plotit(PSD, REM, KC, SP)  # prints results graphs

    sio.savemat(file_num + '.mat', {'res':res})  # save results to file

    print('Done processing and saving file ', file_num)

print("Process took %s seconds." % np.round(time.time() - start_time))
