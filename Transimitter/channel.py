import sys
sys.path.extend(['G:\\python_for_simulation'])

from Signal_interface import Signal_desc
from scipy.fftpack import fftfreq, fft, ifft
from scipy.constants import c

import progressbar
# import array as af
try:
    import cupy as np

    Enable_Gpu = True

except Exception as e:
    print(e)
    import numpy as np

    Enable_Gpu = False
    print('GPU not Supported,npU will be used')


class Span(object):

    def __init__(self, alpha, D, gamma, length, number):
        '''

        :param alpha: db/km
        :param D: [ps/nm/km]
        :param gamma: 1/w/km
        :param length: km
        '''
        self.alpha = alpha
        self.D = D
        self.gamma = gamma
        self.length = length
        self.number = number

    @property
    def alphalin(self):

        return np.log(10 ** (self.alpha / 10))

    @property
    def beta2(self):
        return -self.D * (1550 * 1e-12) ** 2 / (2 * np.pi * c * 1e-3)

    @property
    def leff(self):
        return 1 - np.exp(-self.alphalin * self.length) / self.alphalin

    def get_leff(self, length):
        return (1 - np.exp(-self.alphalin * length)) / self.alphalin

    def split_fourier(self, signal: Signal_desc.Signal, dz):
        sigx = signal.data_sample[0, :]
        sigy = signal.data_sample[1, :]
        vfreq = fftfreq(len(sigx), 1 / (signal.symbol_rate * signal.sps))
        vomeg = 2 * np.pi * vfreq

        dz_eff = self.get_leff(dz)

        sigx = np.asarray(sigx,dtype=np.complex64)
        sigy = np.asarray(sigy,dtype=np.complex64)
        vomeg = np.asarray(vomeg,dtype=np.complex64)

        ux = sigx
        uy = sigy

        Linear = self.beta2 / 2 * 1j * vomeg ** 2 - self.alphalin / 2
        Linear = Linear.astype(np.complex64)
        number_of_step = int(np.ceil(self.length / dz))
        bar = progressbar.ProgressBar()
        bar.start(number_of_step)
        for i in range(int(number_of_step)):
            ux, uy = self.linear_step(ux, uy, Linear, dz,Enable_Gpu)
            ux, uy = self.nl_step(ux, uy, self.gamma, dz_eff,Enable_Gpu)
            ux, uy = self.linear_step(ux, uy, Linear, dz,Enable_Gpu)

            bar.update(i+1)
        bar.finish()

        try:
            ux = ux.get()
            uy = uy.get()
        except Exception as e:
            ux = ux
            uy = uy

        signal.data_sample[0, :] = ux
        signal.data_sample[1, :] = uy

    def linear_step(self, ux, uy, Linear, dz, Enable_Gpu):
        if Enable_Gpu:
            sigx_fft = np.fft.fft(ux)
            ux_fft = sigx_fft * np.exp(dz / 2 * Linear)
            ux = np.fft.ifft(ux_fft)

            sigy_fft = np.fft.fft(uy)
            uy_fft = sigy_fft * np.exp(dz / 2 * Linear)
            uy = np.fft.ifft(uy_fft)
        else:
            sigx_fft = fft(ux)
            ux_fft = sigx_fft * np.exp(dz / 2 * Linear)
            ux = ifft(ux_fft)

            sigy_fft = fft(uy)
            uy_fft = sigy_fft * np.exp(dz / 2 * Linear)
            uy = ifft(uy_fft)
        return ux, uy

    def nl_step(self, ux, uy, gamma, dzeff, Enable_Gpu):
        if Enable_Gpu:
            power = np.abs(ux) ** 2 + np.abs(uy) ** 2
            gamma = gamma * 8 / 9
            Nonlinear = gamma * power * 1j * dzeff

            ux = ux * np.exp(Nonlinear)
            uy = uy * np.exp(Nonlinear)
        else:
            power = np.abs(ux) ** 2 + np.abs(uy) ** 2
            gamma = gamma * 8 / 9
            Nonlinear = gamma * power * 1j * dzeff

            ux = ux * np.exp(Nonlinear)
            uy = uy * np.exp(Nonlinear)
        return ux, uy


if __name__ == '__main__':
    from Signal_interface.Signal_desc import Signal
    from Transimitter import Trans_side, recv_side
    from Transimitter.Trans_side import Laser
    from tool.Quality import QualityMeter

    signal = Signal(0, 35e9, 4, '16qam', 0.2)
    laser = Laser(frequence=193.1, power=1)
    trac = Trans_side.Transimitter_Electrical_Signal(signal, 2 ** 16)
    trac.transimit(rc_span=1024)
    laser.transit(signal)
    QualityMeter.power_meter(signal)
    spans = [Span(0.2, 16.7, 1.3, 80, number=i) for i in range(1)]

    for span in spans:
        span.split_fourier(signal, 20e-3)

    QualityMeter.power_meter(signal)
