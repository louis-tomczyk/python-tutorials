# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.0.0
#   Date            : 2024-06-27
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-06-28) - creation
# 
# ----- MAIN IDEA -----
# Filtering signals
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   [A1] Authors         :
#        Title           :
#        Journal/Editor  :
#        Volume - N°     :
#        Date            :
#        DOI/ISBN        :
#        Pages           :
#  ----------------------
#   CODE
#   [C1] Author          :
#        Contact         :
#        Laboratory/team :
#        Institution     :
#        Date            :
#        Program Title   : 
#        Code Version    : 
#        Web Address     :
# ---------------------------------------------
# %%


import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

#%%
def my_low_pass_filter(signal, filter_params):
# =============================================================================
# signal         : signal to filter
# filter_params  : changes according to the filter type
# rectangular_filter:
# =============================================================================

    def rectangular_filter(signal, cutoff, fs,):
        n           = len(signal)
        freq        = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
        dt          = 1/fs
        time        = np.linspace(0, (n-1)*dt, n)

        fft_signal  = np.fft.fftshift(np.fft.fft(signal))
        filter_mask = np.abs(freq) <= cutoff
        
        fft_signal[~filter_mask] = 0
        filtered_signal = np.fft.ifft(np.fft.fftshift(fft_signal))
        

        # plt.figure()
        # plt.semilogy(freq, np.abs(fft_signal), label="signal fft")
        # plt.semilogy(freq, np.abs(filter_mask), label="filter")
        # plt.legend()
        # plt.xlim([-15,15])
        # plt.show()
        # 
        # plt.figure()
        # plt.plot(time, signal)
        # plt.plot(time, np.real(filtered_signal), label="filtered signal")
        # plt.legend()
        # plt.show()

        return np.real(filtered_signal)
    # ----------------------------------------------------------------------- #
    
    def butterworth_filter(signal, cutoff, order, fs):
        nyq             = 0.5 * fs
        normal_cutoff   = cutoff / nyq
            
        b, a            = butter(order, normal_cutoff, btype='low')
        y               = filtfilt(b, a, signal)
        
        return y
    # ----------------------------------------------------------------------- #
    
    def moving_average_filter(signal, window_size, average_type='uniform'):
        
        if average_type == 'uniform':
            window = np.ones(int(window_size)) / float(window_size)
            
        elif average_type == 'gaussian':
            std_dev = window_size / 6
            window  = np.exp(-0.5 * (np.linspace(-3, 3, window_size) / std_dev) ** 2)
            window  /= np.sum(window)
        else:
            raise ValueError("Invalid average type. Use 'uniform' or 'gaussian'.")
        
        filtered_signal = np.convolve(signal, window, 'same')
        
        return filtered_signal
    # ----------------------------------------------------------------------- #
    
    filter_type = filter_params.get('type', '').lower()
    
    if 'rect' in filter_type:
        return rectangular_filter(signal, filter_params['cutoff'], filter_params['fs'])
    
    elif 'butter' in filter_type:
        return butterworth_filter(signal, filter_params['cutoff'], filter_params['order'], filter_params['fs'])

    elif 'moving_average' in filter_type:
        return moving_average_filter(signal, filter_params['window_size'])
    
    else:
        raise ValueError("Invalid filter type. Use 'rectangular', 'butterworth', or 'moving_average'.")

#%%
def generate_noisy_signal(snr_db, fs, duration, freqs):
    
    t               = np.linspace(0, duration, int(fs*duration))
    signal          = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    signal_power    = np.mean(signal ** 2)
    snr_linear      = 10 ** (snr_db / 10)
    noise_power     = signal_power / snr_linear
    noise           = np.random.normal(0, noise_power, t.shape)
    noisy_signal    = signal + noise
    
    return t, noisy_signal

#%%
# Paramètres
fs          = 5000          # sampling frequency (Hz)
duration    = 0.4           # signal duration (s)
freqs       = [5,10,70]     # sinus frequencies (Hz)
snr_db      = 5             # signal to noise raio (dB)
cutoff      = 10            # cutting frequency
order       = 4             # Butterworth filter ordre
window_size = fs/cutoff     # size of winwdow for moving average



# Génération du signal bruité
t, noisy_signal = generate_noisy_signal(snr_db, fs, duration, freqs)

# Paramètres des filtres
rect_filter_params = {
    'type'          : 'rectangular',
    'cutoff'        : cutoff,
    'fs'            : fs
}

butter_filter_params = {
    'type'          : 'butterworth',
    'order'         : order,
    'cutoff'        : cutoff,
    'fs'            : fs
}

ma_filter_params = {
    'type'          : 'moving_average',
    'window_size'   : window_size,
    'average_type'  : 'uniform'
}

# Application des filtres
filtered_signal_rect    = my_low_pass_filter(noisy_signal, rect_filter_params)
filtered_signal_butter  = my_low_pass_filter(noisy_signal, butter_filter_params)
filtered_signal_ma      = my_low_pass_filter(noisy_signal, ma_filter_params)

# Affichage des résultats
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')

plt.subplot(2, 2, 2)
plt.plot(t, filtered_signal_rect)
plt.title('Rectangular Filter')

plt.subplot(2, 2, 3)
plt.plot(t, filtered_signal_butter)
plt.title('Butterworth Filter')

plt.subplot(2, 2, 4)
plt.plot(t, filtered_signal_ma)
plt.title('Moving Average Filter')

plt.tight_layout()
plt.show()
