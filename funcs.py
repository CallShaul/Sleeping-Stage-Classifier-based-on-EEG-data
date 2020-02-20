import numpy as np
from scipy import signal as sg
import scipy.io as sio
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def prepare_mat(PSD, REM, KC, SP, GT):

    res = np.zeros((len(GT), 8))  # create results array
    res[:, 0:4] = PSD[0:len(GT), :]
    res[:, 4] = REM[0:len(GT), 0] + REM[0:len(GT), 1]
    res[:, 5] = KC[0:len(GT), :].reshape(len(GT))
    res[:, 6] = SP[0:len(GT), :].reshape(len(GT))
    res[:, 7] = GT[0:len(GT), 1]

    return res


def plot_hypnogram(GT, res):

    fig = plt.figure()
    fig.suptitle('Hypnogram', fontsize=14)

    ax1 = fig.add_subplot(121)
    ax1.bar(np.linspace(0, len(GT), len(GT)), GT[:, 0], width=1, color='b')
    #ax1.plot(np.linspace(0, len(GT), len(GT)), GT[:, 0])
    ax1.set_ylabel('Sleeping stage')
    ax1.set_xlabel('Epoch #')
    ax1.legend(['Ground trouth'])

    ax2 = fig.add_subplot(122)
    ax2.bar(np.linspace(0, len(res), len(res)), res[:, 0], width=1, color='b')
    #ax2.plot(np.linspace(0, len(res), len(res)), res[:, 0])
    ax2.set_ylabel('Sleeping stage')
    ax2.set_xlabel('Epoch #')
    ax2.legend(['Results'])


def spindle(sig, fs, epoch, spindle_thresh, folder):

    if sig.ndim > 1:
        channels = sig.shape[1]  # number of channels recieved
    else:
        channels = 1
        sig = sig.reshape(-1, 1)  # makes sig a proper column vector

    sos = sg.butter(6, np.array([9, 16]) / (100 / 2), btype='band',
                    output='sos')  # prepare butterworth LPF parameters at 4 Hz cut-off frq.
    sig = sg.sosfiltfilt(sos, sig, axis=0)  # perform LPF
    
    l = len(sig)
    SP_raw = np.zeros((int(np.round(l / (epoch * fs))) + 1, channels))  # create results array
    spindle_ref_temp = sio.loadmat(folder + 'spindle_ref.mat')  # loading referance signal mat file
    spindle_ref = spindle_ref_temp['spindle_ref']  # loading K-complex normilized referance signal's data
    sig = sig * -1  # signal's DC voltage direction is opposite! this needs to be fixed.
    seg_idx = 0

    for j in range(channels):
        for i in range(l):

            if sig[i, j] > spindle_thresh and seg_idx > fs:

                if i + 0.55 * fs < l and i - 0.45 * fs > 0:
                    segment = sig[int(i - 0.45 * fs):int(i + 0.55 * fs), j]  # take segment

                    segment -= np.min(segment)  # positive values only
                    segment = segment / np.max(segment)  # normilize to max 1
                    segment = segment.reshape(100, 1)  # adjust to spindle referance signal shape

                    E_dist = np.sum(np.sqrt(np.square(spindle_ref - segment)))  # Euclidean distance

                    this_epoch = np.floor(i / (epoch * fs))  # find which epoch the result belongs to
                    SP_raw[int(this_epoch), j] = E_dist  # save result

                    """
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax1.plot(spindle_ref[:, 0], label='spindle_ref')
                    ax2 = fig.add_subplot(122)
                    ax2.plot(segment[:, 0], label='segment')
                    ax1.title.set_text('spindle_ref')
                    ax2.title.set_text('segment')
                    print(E_dist)
                    print(i)
                    plt.show()
                    """

                seg_idx = 0
            seg_idx += 1

    SP_raw[SP_raw == 0] = np.inf  # change zeros to inf to prevent devision by zero later
    SP_merged = SP_raw ** -1
    SP_merged = np.sum(SP_merged, axis=1)  # change to 1D array while higher values means better spindle correlation
    SP_merged = SP_merged.reshape(len(SP_merged), 1)
    SP_merged = SP_merged / np.max(SP_merged)  # normilize to 1

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

                    E_dist = np.sum(np.sqrt(np.square(KC_ref - segment)))  # Euclidean distance

                    this_epoch = np.floor(i / (epoch * fs))  # find which epoch the result belongs to
                    KC_raw[int(this_epoch), j] = E_dist  # save result

                    '''
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax1.plot(KC_ref[:, 0], label='KC_ref')
                    ax2 = fig.add_subplot(122)
                    ax2.plot(segment[:, 0], label='segment')
                    ax1.title.set_text('KC_ref')
                    ax2.title.set_text('segment')
                    print(E_dist * RMS_diff)
                    print(i)
                    #plt.show()
                    '''

                seg_idx = 0
            seg_idx += 1
    KC_raw[KC_raw == 0] = np.inf  # change zeros to inf to prevent devision by zero later
    KC_merged = KC_raw**-1
    KC_merged = np.sum(KC_merged, axis=1)  # change to 1D array while higher values means better K-complex correlation
    KC_merged = KC_merged.reshape(len(KC_merged), 1)
    KC_merged = KC_merged / np.max(KC_merged)  # normilize to 1

    return KC_raw, KC_merged


def REM_finder(sig, fs, epoch):

    if sig.ndim != 2:  # exit if there isn't 2 channels data (that represents 2 eyes)
        return 0
    else:
        l = len(sig)
        eyes_power = np.zeros((int(np.round(l / (epoch * fs))) + 1, 2))  # creates 2d matrix to store results in
        sig_sqr = np.square(sig)  # square signal voltage to correlate data to power [uv^2]
        k = 0

        for i in range(0, l, epoch * fs):  # itterates overs time epochs and create data segments

            if i + epoch * fs < l:  # case there's enough data for full epoch
                segment = sig_sqr[i:i + epoch * fs, :]
            else:  # case there isn't enough data for full epoch - take the data that's left
                segment = sig_sqr[i:i + (l - i), :]

            eyes_power[k, :] = np.mean(segment, axis=0)  # calculate segment's average for each eye data
            k = k + 1

        return eyes_power


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
        power = power[..., np.newaxis]  # adds another dimension

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


def plotit(PSD, REM, KC, SP):

    fig = plt.figure()
    #fig, ax = plt.subplots(2, 4)
    fig.suptitle('EEG channels calculated results', fontsize=14)


    ax1 = fig.add_subplot(241)
    ax1.plot(PSD[:, 0]*100)
    ax1.set_ylabel('Relative power [%]')
    ax1.set_xlabel('Epoch #')
    ax1.legend(['Delta'])
    #ax1.grid('on')


    ax2 = fig.add_subplot(242)
    ax2.plot(PSD[:, 1]*100)
    ax2.set_ylabel('Relative power [%]')
    ax2.set_xlabel('Epoch #')
    ax2.legend(['Theta'])
    #ax2.grid('on')

    ax3 = fig.add_subplot(243)
    ax3.plot(PSD[:, 2]*100)
    ax3.set_ylabel('Relative power [%]')
    ax3.set_xlabel('Epoch #')
    ax3.legend(['Alpha'])
    #ax3.grid('on')

    ax4 = fig.add_subplot(244)
    ax4.plot(PSD[:, 3]*100)
    ax4.set_ylabel('Relative power [%]')
    ax4.set_xlabel('Epoch #')
    ax4.legend(['Beta'])
    #ax4.grid('on')

    ax5 = fig.add_subplot(245)
    ax5.plot(REM[:, 0] + REM[:, 1], color='r')
    ax5.set_ylabel('Power [uV^2]')
    ax5.set_xlabel('Epoch #')
    ax5.legend(['REM'])
    #ax5.grid('on')

    ax6 = fig.add_subplot(246)
    ax6.bar(np.linspace(1, len(KC), len(KC)), KC[:, 0], width=1, color='g')
    ax6.set_ylabel('Relative activity measure')
    ax6.set_xlabel('Epoch #')
    ax6.legend(['K-complex'])
    #ax6.grid('on')

    ax7 = fig.add_subplot(247)
    ax7.bar(np.linspace(1, len(SP), len(SP)), SP[:, 0], width=1, color='m')
    ax7.set_ylabel('Relative activity measure')
    ax7.set_xlabel('Epoch #')
    ax7.legend(['Spindle'])
    #ax7.grid('on')

    #plt.tight_layout()
    plt.show()

