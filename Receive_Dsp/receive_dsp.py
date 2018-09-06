from Signal_interface import Signal_desc
from Transimitter.channel_af import Span
from scipy.constants import c
from scipy.fftpack import fft,ifft,fftfreq
import numpy as np
from typing import List
class Dsp(object):

    @staticmethod
    def CD_Compensation_without_block(span:Span,signal:Signal_desc.Signal):
        '''

        :param span: span object,
        :param signal: signal to compensate CD
        :return: None

        This method is used to compensate the CD effect of the signl in frequence domian, the signal is not divided
        into blocks:

        Frequence Domain:

            G(z,w) = exp(j *D*lambad**2*w**2/4/pi/c)
        '''

        x_sample = signal.data_sample[0, :]
        y_sample = signal.data_sample[1, :]
        fs = signal.symbol_rate * signal.sps
        freq = fftfreq(len(x_sample), 1 / fs)
        vomeg = 2 * np.pi * freq
        # vomeg = vomeg/fs
        c_km_s = c / 1000


        wave_length = span.signal_wave_length*10**(-12)
        gzw = np.exp((1j*span.D*wave_length**2*span.length/4/np.pi/c_km_s)*vomeg**2)
        x_sample = ifft(fft(x_sample)*gzw)
        y_sample = ifft(fft(y_sample)*gzw)

        signal.data_sample = np.array([x_sample,y_sample])

    @staticmethod
    def PhaseRecovery(signal:Signal_desc.Signal):
        '''

        :param signal: the Signal to recover phase
        :return:

        '''