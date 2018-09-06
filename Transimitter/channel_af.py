
import sys

sys.path.append('G:\\python_for_simulation')

from Signal_interface import Signal_desc
from scipy.fftpack import fftfreq, fft, ifft
from scipy.constants import c

import progressbar
import arrayfire as af

import numpy as np

Enable_Gpu = True




class Span(object):

    def __init__(self, alpha, D, gamma, length, number,signal_wave_length=None):
        '''

        :param alpha: db/km
        :param D: [ps/nm/km]
        :param gamma: 1/w/km
        :param length: km
        :param signal_wave_length the wavelength of signal unit:nm
        '''
        self.alpha = alpha
        self.D = D
        self.gamma = gamma
        self.length = length
        self.number = number

        if signal_wave_length is None:
            self.signal_wave_length = 1550
        else:
            self.signal_wave_length = signal_wave_length

    @property
    def alphalin(self):

        return np.log(10 ** (self.alpha / 10))

    @property
    def beta2(self):
        # if self.signal_wave_length is not None:
        #     return -self.D * (self.signal_wave_length * 1e-12) ** 2 / (2 * np.pi * c * 1e-3)
        # else:
        assert self.signal_wave_length is not None
        return -self.D * (self.signal_wave_length * 1e-12) ** 2 / (2 * np.pi * c * 1e-3)

    @property
    def leff(self):
        return (1 - np.exp(-self.alphalin * self.length)) / self.alphalin

    def get_leff(self, length):
        return (1 - np.exp(-self.alphalin * length)) / self.alphalin

    def __str__(self):
        return f'the span length is {self.length} km\n' \
               f'D:{self.D} ps/nm/km\n' \
               f'gamma:{self.gamma} 1/w/km\n' \
               f'alpha:{self.alpha} db'

    def __repr__(self):
        return f'the span length is {self.length} km\n' \
               f'D:{self.D} ps/nm/km\n' \
               f'gamma:{self.gamma} 1/w/km\n' \
               f'alpha:{self.alpha} db'

    def split_fourier(self, signal: Signal_desc.Signal, dz):
        sigx = signal.data_sample[0, :]
        sigy = signal.data_sample[1, :]
        # sigx.astype(np.complex64)
        # sigy.astype(np.complex64)
        vfreq = fftfreq(signal.sym_length*signal.sps, 1 / (signal.symbol_rate * signal.sps))


        vomeg = 2 * np.pi * vfreq
        # normal_vomoge = vomeg/(signal.symbol_rate*signal.sps)
        normal_vomoge = vomeg
        # vomeg.astype(np.complex64)
        dz_eff = self.get_leff(dz)

        sigx = af.np_to_af_array(sigx)
        sigy = af.np_to_af_array(sigy)

        normal_vomoge = af.np_to_af_array(normal_vomoge)

        sigx = sigx.as_type(af.Dtype.c64)
        sigy = sigy.as_type(af.Dtype.c64)

        normal_vomoge = normal_vomoge.as_type(af.Dtype.c64)

        ux = sigx
        uy = sigy

        Linear = self.beta2 / 2 * 1j * normal_vomoge ** 2 - self.alphalin / 2
        Linear = Linear.as_type(af.Dtype.c64)
        number_of_step = int(np.ceil(self.length / dz))
        print(f'the {self.number}-th span', end=': \n')

        bar = progressbar.ProgressBar()
        bar.start(number_of_step)

        for i in range(int(number_of_step)):

            ux, uy = self.linear_step(ux, uy, Linear, dz, Enable_Gpu)
            ux, uy = self.nl_step(ux, uy, self.gamma, dz_eff, Enable_Gpu)
            ux, uy = self.linear_step(ux, uy, Linear, dz, Enable_Gpu)
            # print(i)
            bar.update(i + 1)
        bar.finish()

        ux = ux
        uy = uy

        signal.data_sample[0, :] = np.array(ux)
        signal.data_sample[1, :] = np.array(uy)

    def linear_step(self, ux, uy, Linear, dz, Enable_Gpu):
        if Enable_Gpu:
            # sigx_fft = np.fft.fft(ux)
            # ux_fft = sigx_fft * np.exp(dz / 2 * Linear)
            # ux = np.fft.ifft(ux_fft)
            sigx_fft = af.fft(ux)
            ux_fft = sigx_fft * af.exp(dz / 2 * Linear)
            ux = af.ifft(ux_fft)

            # sigy_fft = np.fft.fft(uy)
            # uy_fft = sigy_fft * np.exp(dz / 2 * Linear)
            # uy = np.fft.ifft(uy_fft)
            sigy_fft = af.fft(uy)
            uy_fft = sigy_fft * af.exp(dz / 2 * Linear)
            uy = af.ifft(uy_fft)
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
            power = af.abs(ux) ** 2 + af.abs(uy) ** 2
            gamma = gamma * 8 / 9
            Nonlinear = gamma * power * 1j * dzeff

            ux = ux * af.exp(Nonlinear)
            uy = uy * af.exp(Nonlinear)
        else:
            power = np.abs(ux) ** 2 + np.abs(uy) ** 2
            gamma = gamma * 8 / 9
            Nonlinear = gamma * power * 1j * dzeff

            ux = ux * np.exp(Nonlinear)
            uy = uy * np.exp(Nonlinear)
        return ux, uy


if __name__ == '__main__':
    snr_dbs = []
    from Signal_interface.Signal_desc import Signal
    from Transimitter import Trans_side, recv_side
    from Transimitter.Trans_side import Laser
    from tool.Quality import QualityMeter
    import time, up_down_sample
    import vision_dom.vision as vision
    from Receive_Dsp import receive_dsp
    # signal = Signal(0, 35e9, 4, '16qam', 0.2)



    # QualityMeter.power_meter(signal)
    spans = [Span(0.2, 16.7, 1.3, 80, number=i+1) for i in range(1)]
    # now = time.time()


    for signal_power in [-4,-3,-2,-1,0,1,2,3,4]:


        signal = Signal(signal_power, 35e9, 4, '16qam', 0.2)

        trac = Trans_side.Transimitter_Electrical_Signal(signal, 2 ** 16)
        trac.transimit(rc_span=1024)
        laser = Laser(frequence=193.1, signal=signal)
        laser.transimit(signal)
        QualityMeter.power_meter(signal)
        for span in spans:
            span.split_fourier(signal, 20e-3)
            gain = 10 ** (16 / 10)
            signal.data_sample = np.sqrt(gain) * signal.data_sample
            QualityMeter.power_meter(signal)


        # print(time.time() - now)

        power,_ = QualityMeter.power_meter(signal)


        for span in spans:
            receive_dsp.Dsp.CD_Compensation_without_block(span, signal)
        rece = recv_side.Receiver(signal)
        #
        matched_signal = rece.matched_filter(trac.pulse_shaping,trac.pulse_shaping_delay)
        #
        #
        #
        #

        matched_signal.data_sample = up_down_sample.downsample(matched_signal.data_sample, signal.sps)
        #
        # matched_signal.data_sample = matched_signal.data_sample[:, :signal.sym_length]
        matched_signal.symbol = matched_signal.data_sample
        matched_signal.symbol = matched_signal.symbol/np.sqrt(np.mean(matched_signal.symbol*np.conj(matched_signal.symbol),axis=1).reshape(2,1))

        angle = np.angle(np.sum(matched_signal.symbol / signal.symbol,axis=1).reshape(2,1))
        matched_signal.symbol = matched_signal.symbol*(np.exp(-1j*angle))

        snr,snr_db = QualityMeter.measure_snr(matched_signal.symbol[:,1024:-1024],signal.symbol[:,1024:-1024])
        snr_dbs.append(snr_db)
        print(snr_db)
        with open('snr_db_channel','w') as f:
            for snr in snr_dbs:
                f.write(str(snr)+'\n')





        rece.hard_decision_symbol(matched_signal)
        rece.decode_msg(matched_signal)


        err = rece.biterr(matched_signal.msg[0,:],signal.msg[0,:],4)

        print(err,snr_db)
