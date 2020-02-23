import numpy as np
from scipy import signal as sg
import scipy.io as sio
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib
import time
from sklearn import svm
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def pulse_finder(sig, epoch):

    fs = 10  # sampling rate of the data is 10 Hz
    low = 0.6 / (fs / 2)
    high = 1.5 / (fs / 2)
    sos = sg.butter(6, [low, high], btype="band", output='sos')  # prepare butterworth filter to down-sample signal by factor of 2 (from 200Hz to 100Hz)
    sig = sg.sosfiltfilt(sos, sig, axis=0)  # performs bandpass filter on the signal

    f, t, Zxx = sg.stft(sig, fs, window='hann', nperseg=epoch*fs*2, axis=0, noverlap=None)  # performs STFT on the signal

    zxx = np.abs(Zxx[:, 0, :])  # takes the abselute value of all segments
    zxx_max = np.argmax(zxx, axis=0).reshape(-1, 1)
    HR = f[zxx_max]

    """
    plt.pcolormesh(t, f, zxx)
    plt.title('Heart rate STFT time-frequency analysis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    l = np.uint16(np.floor(len(sig) / (epoch*fs)))  # calculates expected length of all epochs
    t_vec = np.linspace(1, l / 2, l)  # create time vector for x-axis
    plt.plot(t_vec, HR[0:l, 0] * 60)
    plt.title('Heart rate')
    plt.ylabel('Heart rate')
    plt.xlabel('Time [sec]')
    plt.show()
    """

    return HR


def analyze_data(folder, files, KC_thresh, SP_thresh, epoch, save=True, plot=True):

    start_time = time.time()

    for file_num in files:

        path = folder + '26' + file_num + '_py.mat'  # prepare file path and name

        GT = sio.loadmat(path)['GT']  # load Ground trouth data

        sos = sg.butter(6, 99.99 / (200 / 2), btype="low",
                        output='sos')  # prepare butterworth filter to down-sample signal by factor of 2 (from 200Hz to 100Hz)
        sig = sg.sosfiltfilt(sos, sio.loadmat(path)['data_200hz'], axis=0)[::2]  # perform LPF and then down-sample signal to 100 Hz
        fs = 100  # new sampling rate [Hz]

        HR_sig = sio.loadmat(path)['data_10hz'][:, 0].reshape(-1, 1)  # loads heart rate data

        HR = pulse_finder(HR_sig, epoch)  # calls pulse calculator function

        chin = power_finder(sig[:, 9], fs, epoch)
        #chin[chin > 1000] = 1000  # removes artifacts

        REM = power_finder(sig[:, 7:9], fs, epoch)  # calculate left & right eyes signal power in epochs
        #REM[REM > 10000] = 10000  # removes artifacts

        #  PSD: Delta, Theta, Alpha, Beta
        PSD = power_calc(sig[:, 0:6], fs, epoch, low=0.5, high=3.5, avg=True)  # Delta waves PSD
        PSD = np.append(PSD, power_calc(sig[:, 0:6], fs, epoch, low=4, high=7, avg=True), axis=1)  # Theta waves PSD
        PSD = np.append(PSD, power_calc(sig[:, 0:6], fs, epoch, low=7.5, high=12, avg=True), axis=1)  # Alpha waves PSD
        PSD = np.append(PSD, power_calc(sig[:, 0:6], fs, epoch, low=15.5, high=30, avg=True), axis=1)  # Beta waves PSD

        KC_raw, KC = k_complex(sig[:, 0:6], fs, epoch, KC_thresh, folder)  # calculates K-complex activity

        SP_raw, SP = spindle(sig[:, 0:6], fs, epoch, SP_thresh, folder)  # calculates spindle activity

        res = prepare_mat(PSD, REM, KC, SP, GT, HR, chin)  # prepare current itteration results into one matrix

        if 'res_all' in locals():  # checks if res exists
            res_all = np.append(res_all, res, axis=0)  # creates res at first itteration
        else:
            res_all = res  # if res doesn't exist- create it

        if plot:
            plotit(file_num, res)  # plots all features

        if save:
            sio.savemat(file_num + '.mat', {'res': res, 'PSD': PSD, 'REM': REM, 'KC': KC, 'SP': SP, 'HR': HR, 'chin': chin, 'epoch': epoch})  # save results to file

        print('Done processing and saving file 26' + file_num + '_py.mat, total time in file: ' + str(len(KC) / 2) + ' Min')

    print("Total process time: %s seconds." % np.round(time.time() - start_time))

    return res_all  # return the results of ALL itterations (all files)


def SVM_test(clf, file, plot=True):

    test = sio.loadmat(file + '.mat')['res']  # load test data

    test[:, 0:9] -= test[:, 0:9].mean(axis=0)  # reduce average from train
    test[:, 0:9] = test[:, 0:9] / np.std(test[:, 0:9], axis=0)  # divide train by STD

    result = clf.predict(test[:, 0:9]).reshape(-1, 1)  # save results

    if plot:
        plot_hypnogram(test[:, 9].reshape(-1, 1), result, file)

    return result


def SVM_train(train):

    train[:, 0:9] -= train[:, 0:9].mean(axis=0)  # reduce average from train
    train[:, 0:9] = train[:, 0:9] / np.std(train[:, 0:9], axis=0)  # divide train by STD

    clf = svm.SVC(C=0.5, kernel='rbf')
    clf.fit(train[:, 0:9], train[:, 9])

    return clf


def prepare_mat(PSD, REM, KC, SP, GT, HR, chin):

    l = len(GT)  # ground trouth data length

    res = np.zeros((l, 10))  # create results array

    res[:, 0:4] = PSD[0:l, :]  # power spectral density (PSD) data
    res[:, 4] = REM[0:l, 0] + REM[0:len(GT), 1]  # REM data
    res[:, 5] = KC[0:l, :].reshape(len(GT))  # K-complex data
    res[:, 6] = SP[0:l, :].reshape(len(GT))  # spindle data
    res[:, 7] = HR[0:l, 0]  # heart rate data
    res[:, 8] = chin[0:l, 0]  # chin data
    res[:, 9] = GT[0:l, 1]  # ground trouth - must be last vector in the matrix

    return res


def spindle(sig, fs, epoch, SP_thresh, folder):

    if sig.ndim > 1:
        channels = sig.shape[1]  # number of channels recieved
    else:
        channels = 1
        sig = sig.reshape(-1, 1)  # makes sig a proper column vector

    sos = sg.butter(6, np.array([9, 16]) / (100 / 2), btype='band', output='sos')  # prepare butterworth LPF parameters at 4 Hz cut-off frq.
    sig = sg.sosfiltfilt(sos, sig, axis=0)  # perform LPF
    
    l = len(sig)
    SP_raw = np.zeros((int(np.round(l / (epoch * fs))) + 1, channels))  # create results array
    spindle_ref_temp = sio.loadmat(folder + 'spindle_ref.mat')  # loading referance signal mat file
    spindle_ref = spindle_ref_temp['spindle_ref']  # loading K-complex normilized referance signal's data
    sig = sig * -1  # signal's DC voltage direction is opposite! this needs to be fixed.
    seg_idx = 0

    for j in range(channels):
        for i in range(l):

            if sig[i, j] > SP_thresh and seg_idx > fs:

                if i + 0.55 * fs < l and i - 0.45 * fs > 0:
                    segment = sig[int(i - 0.45 * fs):int(i + 0.55 * fs), j]  # take segment

                    segment -= np.min(segment)  # positive values only
                    segment = segment / np.max(segment)  # normilize to max 1
                    segment = segment.reshape(100, 1)  # adjust to spindle referance signal shape

                    E_dist = np.sqrt(np.sum((spindle_ref - segment)**2))
                    #E_dist_old = np.sum(np.sqrt(np.square(spindle_ref - segment)))  # Euclidean distance

                    this_epoch = np.floor(i / (epoch * fs))  # find which epoch the result belongs to
                    SP_raw[int(this_epoch), j] = E_dist  # save result

                    '''
                    fig = plt.figure()
                    fig.suptitle('Segment-Reference signals euclidean distance :' + str(np.round(E_dist, 2)), fontsize=14)
                    ax1 = fig.add_subplot(121)
                    ax1.plot(spindle_ref[:, 0], label='Spindle reference signal')
                    ax2 = fig.add_subplot(122)
                    ax2.plot(segment[:, 0], label='Current examined segment')
                    ax1.title.set_text('Spindle reference signal')
                    ax2.title.set_text('Current examined segment')
                    print(E_dist)
                    print(i)
                    plt.show()
                    '''

                seg_idx = 0
            seg_idx += 1

    SP_raw[SP_raw == 0] = np.inf  # change zeros to inf to prevent devision by zero later
    SP_merged = SP_raw ** -1
    SP_merged = np.sum(SP_merged, axis=1)  # change to 1D array while higher values means better spindle correlation
    SP_merged = SP_merged.reshape(len(SP_merged), 1)
    #SP_merged = SP_merged / np.max(SP_merged)  # normilize to 1

    return SP_raw, SP_merged


def k_complex(sig, fs, epoch, KC_thresh, folder):

    if sig.ndim > 1:
        channels = sig.shape[1]  # number of channels recieved
    else:
        channels = 1
        sig = sig.reshape(-1, 1)  # makes sig a proper column vector

    sos = sg.butter(6, 4 / (100 / 2), btype='low', output='sos')  # prepare butterworth LPF parameters at 4 Hz cut-off frq.
    sig = sg.sosfiltfilt(sos, sig, axis=0)  # perform LPF
    l = len(sig)
    KC_raw = np.zeros((int(np.round(l / (epoch * fs))) + 1, channels))  # create results array
    KC_ref_temp = sio.loadmat(folder + 'k_complex_ref.mat')  # loading referance signal mat file
    KC_ref = KC_ref_temp['KC_ref']  # loading K-complex normilized referance signal's data
    sig = sig*-1  # signal's DC voltage direction is opposite! this needs to be fixed.
    seg_idx = 0
    
    for j in range(channels):
        for i in range(l):

            if sig[i, j] > KC_thresh and seg_idx > 2.5*fs:

                if i + 1.5 * fs < l and i - fs > 0:
                    segment = sig[i - fs:int(i + 1.5 * fs), j]  # take segment

                    segment -= np.min(segment)  # positive values only
                    segment = segment / np.max(segment)  # normilize to max 1
                    segment = segment.reshape(250, 1)  # adjust to K-complex referance signal shape

                    E_dist = np.sqrt(np.sum((KC_ref - segment) ** 2))
                    #E_dist_old = np.sum(np.sqrt(np.square(KC_ref - segment)))  # Euclidean distance

                    this_epoch = np.floor(i / (epoch * fs))  # find which epoch the result belongs to
                    KC_raw[int(this_epoch), j] = E_dist  # save result

                    '''
                    fig = plt.figure()
                    fig.suptitle('Segment-Reference signals euclidean distance: ' + str(np.round(E_dist, 2)), fontsize=14)
                    ax1 = fig.add_subplot(121)
                    ax1.plot(KC_ref[:, 0], label='K-Complex reference signal')
                    ax2 = fig.add_subplot(122)
                    ax2.plot(segment[:, 0], label='Current examined segment')
                    ax1.title.set_text('K-Complex reference signal')
                    ax2.title.set_text('Current examined segment')
                    print(i)
                    plt.show()
                    '''

                seg_idx = 0
            seg_idx += 1
    KC_raw[KC_raw == 0] = np.inf  # change zeros to inf to prevent devision by zero later
    KC_merged = KC_raw**-1
    KC_merged = np.sum(KC_merged, axis=1)  # change to 1D array while higher values means better K-complex correlation
    KC_merged = KC_merged.reshape(len(KC_merged), 1)
    #KC_merged = KC_merged / np.max(KC_merged)  # normilize to 1

    return KC_raw, KC_merged


def power_finder(sig, fs, epoch):

    if sig.ndim > 1:
        channels = sig.shape[1]  # number of channels recieved
    else:
        channels = 1
        sig = sig.reshape(-1, 1)  # makes sig a proper column vector

    l = len(sig)
    sig_pow = np.zeros((int(np.round(l / (epoch * fs))) + 1, channels))  # creates 2d matrix to store results in
    sig_sqr = np.square(sig)  # square signal voltage to correlate data to power [uv^2]
    k = 0

    for i in range(0, l, epoch * fs):  # itterates overs time epochs and create data segments

        if i + epoch * fs < l:  # case there's enough data for full epoch
            segment = sig_sqr[i:i + epoch * fs, :]
        else:  # case there isn't enough data for full epoch - take the data that's left
            segment = sig_sqr[i:i + (l - i), :]

        sig_pow[k, :] = np.mean(segment, axis=0)  # calculate segment's average for each eye data
        k = k + 1

    return sig_pow


def power_calc(sig, fs, epoch, low, high, avg=False):

    if sig.ndim > 1:
        channels = sig.shape[1]  # number of channels recieved
    else:
        channels = 1
        sig = sig.reshape(-1, 1)  # makes sig a proper column vector

    l = len(sig)
    power = np.zeros((int(np.round(l / (epoch * fs))) + 1, channels))  # creates array to store results in

    for j in range(channels):  # itterates over channels

        k = 0
        for i in range(0, l, epoch*fs):  # itterates overs time epochs and create data segments
            
            if i + epoch*fs < l:  # case there's enough data for full epoch 
                segment = sig[i:i + epoch*fs, j]
            else:  # case there isn't enough data for full epoch - take the data that's left
                segment = sig[i:i + (l - i), j]
                
            power[k, j] = bandpower(segment, fs, low, high)  # calculate segment's PSD at specific frequency band
            k = k + 1

    if sig.ndim > 1 and avg is True:
        power = np.mean(power, axis=1)  # average all channels results into one channel
        power = power[..., np.newaxis]  # fix vector

    return power


def bandpower(data, fs, low, high, method='welch', window_sec=None, relative=True):
    # Compute the average power of the signal in a specific frequency band.

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * fs
        else:
            nperseg = (2 / low) * fs

        freqs, psd = welch(data, fs, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, fs, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def plotit(file, res):

    t_vec = np.linspace(1, len(res) / 2, len(res))  # create time vector for x-axis, units: Min

    fig = plt.figure()
    fig.suptitle('26' + file + ' - Extracted features', fontsize=14)

    ax1 = fig.add_subplot(331)
    ax1.plot(t_vec, res[:, 0]*100)
    ax1.set_ylabel('Relative power [%]')
    ax1.set_xlabel('Time [Min]')
    ax1.legend(['Delta'])
    #ax1.grid('on')

    ax2 = fig.add_subplot(332)
    ax2.plot(t_vec, res[:, 1]*100)
    ax2.set_ylabel('Relative power [%]')
    ax2.set_xlabel('Time [Min]')
    ax2.legend(['Theta'])
    #ax2.grid('on')

    ax3 = fig.add_subplot(333)
    ax3.plot(t_vec, res[:, 2]*100)
    ax3.set_ylabel('Relative power [%]')
    ax3.set_xlabel('Time [Min]')
    ax3.legend(['Alpha'])
    #ax3.grid('on')

    ax4 = fig.add_subplot(334)
    ax4.plot(t_vec, res[:, 3]*100)
    ax4.set_ylabel('Relative power [%]')
    ax4.set_xlabel('Time [Min]')
    ax4.legend(['Beta'])
    #ax4.grid('on')

    ax5 = fig.add_subplot(335)
    ax5.plot(t_vec, res[:, 4], color='r')
    ax5.set_ylabel('Eyes muscle activity [uV^2]')
    ax5.set_xlabel('Time [Min]')
    ax5.legend(['REM'])
    #ax5.grid('on')

    ax6 = fig.add_subplot(336)
    ax6.plot(t_vec, res[:, 8], color='k')
    ax6.set_ylabel('Chin muscle activity [uV^2]')
    ax6.set_xlabel('Time [Min]')
    ax6.legend(['Chin'])
    #ax6.grid('on')

    ax7 = fig.add_subplot(337)
    ax7.bar(t_vec, res[:, 6], width=1, color='m')
    ax7.set_ylabel('Relative activity measure [A.U]')
    ax7.set_xlabel('Time [Min]')
    ax7.legend(['Spindle'])
    #ax7.grid('on')

    ax8 = fig.add_subplot(338)
    ax8.bar(t_vec, res[:, 5], width=1, color='g')
    ax8.set_ylabel('Relative activity measure [A.U]')
    ax8.set_xlabel('Time [Min]')
    ax8.legend(['K-complex'])
    #ax8.grid('on')

    ax9 = fig.add_subplot(339)
    ax9.plot(t_vec, res[:, 7]*60, color='c')
    ax9.set_ylabel('Heart rate [BPM]')
    ax9.set_xlabel('Time [Min]')
    ax9.legend(['Pulse'])
    #ax9.grid('on')

    #plt.tight_layout()
    plt.show()


def plot_hypnogram(GT, res, file):

    l = len(GT)
    err = (np.abs(GT - res))  # calculate ground trouth - results array differance
    err[err > 0] = 1  # every miss counts the same (as one)
    err_ratio = str(np.round(np.uint16(np.sum(err)) * 100 / l, decimals=1))  # calculate error precentage

    fig = plt.figure()
    fig.suptitle('26' + file + ' - Hypnogram, error: ' + err_ratio + ' %', fontsize=14)

    ax1 = fig.add_subplot(121)
    ax1.bar(np.linspace(0, l / 2, l), GT[:, 0], width=1, color='b')
    #ax1.plot(np.linspace(0, len(GT), len(GT)), GT[:, 0])
    ax1.set_ylabel('Sleeping stage')
    ax1.set_xlabel('Time [Min]')
    ax1.legend(['Ground trouth'])

    ax2 = fig.add_subplot(122)
    ax2.bar(np.linspace(0, l / 2, l), res[:, 0], width=1, color='b')
    #ax2.plot(np.linspace(0, len(res), len(res)), res[:, 0])
    ax2.set_ylabel('Sleeping stage')
    ax2.set_xlabel('Time [Min]')
    ax2.legend(['Calculated results'])


