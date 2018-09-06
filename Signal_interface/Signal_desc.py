from scipy.constants import c
from scipy.constants import h
from scipy.io import loadmat
import numpy as np


class Signal(object):
    '''
        describe a Signal,include Power,symbol_rate,sps,modulation-format, when use this class,first you should construct

        object with proper arguments,and then use transimitter object's generate_symbol method to generate qpsk, 16qam symbol
    '''

    def __init__(self, power=0, symbol_rate=35e9, sps=4, mf='16qam', roll_off=0.14):
        '''

        :param power: signal power in dbm
        :param symbol_rate: symbol rate in hz
        :param sps: sample per symbol
        :param mf: modulation_format 16qam or qpsk, pol-multiplexed
        :param roll_off: the roll_off of transimitter
        '''

        self.power = power
        self.symbol_rate = symbol_rate
        self.sps = sps
        self.mf = mf
        self.roll_off = roll_off

        self.carrier_frequence = []  # will be set in transimitter

        self.symbol = None  # when call generate_symbol method ,this attribute will be set
        self.data_sample = None  # will be set in transimitter
        self.msg = None  # msg will be mapped to symbol
        self.sym_length = None
    def __str__(self):

        return f'''\t signal's power is {self.power} dbm. symbol_rate is {self.symbol_rate}\n
                    \t signal's modulation_format is {self.mf}, the roll_off factor is {self.roll_off}\n
                    \t wdm_channel is {len(self.carrier_frequence)}
                '''

    def __repr__(self):

        return f'''\t signal's power is {self.power} dbm. symbol_rate is {self.symbol_rate}\n
                    \t signal's modulation_format is {self.mf}, the roll_off factor is {self.roll_off}\n
                    \t wdm_channel is {len(self.carrier_frequence)}
                '''

    @property
    def fs(self):
        return self.sps * self.symbol_rate

    @property
    def bch(self):
        return self.symbol_rate * (1 + self.bch)

    @property
    def signal_power_dbm(self):

        return self.power

    @property
    def signal_power_in_w(self):
        return (10 ** (self.power / 10)) / 1000

    def set_data_sample_power(self, power, unit='dbm'):
        assert self.data_sample is not None
        self.data_sample = self.data_sample / np.sqrt(
            np.mean(self.data_sample * np.conj(self.data_sample), axis=1).reshape(2, 1))

        if unit == 'dbm':
            print(f'the data sample power will be set as {power} dbm')
            power = (10 ** (power / 10)) / 1000
            x_power = power / 2
            y_power = power / 2

            self.data_sample[0, :] = self.data_sample * np.sqrt(x_power)
            self.data_sample[1, :] = self.data_sample * np.sqrt(y_power)

        if unit == 'w':
            print(f'the data samples power will be set as {power} w')

            x_power = power / 2
            y_power = power / 2

            self.data_sample[0, :] = self.data_sample * np.sqrt(x_power)
            self.data_sample[1, :] = self.data_sample * np.sqrt(y_power)
