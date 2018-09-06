import numpy as np
from scipy.signal import upfirdn
from up_down_sample import upsample
from scipy.signal import fftconvolve,upfirdn
from scipy.io import loadmat
from Transimitter import pulse_shaping
from Signal_interface import Signal_desc


class Transimitter_Electrical_Signal(object):

    def __init__(self, signal_to_send: Signal_desc.Signal, symbol_length):

        self.signal = signal_to_send
        self.length = symbol_length
        self.pulse_shaping = None
        self.signal.sym_length = symbol_length
        self.pulse_shaping_delay = None

    def transimit(self, rc_span):
        print('\t------------------------- generate QAM Symbol ------------------------------------------')
        self._generate_symbol()
        print(
            '\t------------------------- generate QAM Symbol complete----------------------------------')

        h = pulse_shaping.rrc_filter(rc_span,self.signal.roll_off,self.signal.sps,'rrc')
        self.pulse_shaping = h
        print('\t------------- ----------- base band pulse_shaping---------------------------------------')
        self.signal.data_sample = fftconvolve(upsample(self.signal.symbol,self.signal.sps), h)
        # self.signal.data_sample = upfirdn(h[0],self.signal.symbol,self.signal.sps,1,axis=1)
        self.pulse_shaping_delay = int(rc_span / 2 * self.signal.sps)
        self.signal.data_sample = np.roll(self.signal.data_sample,-self.pulse_shaping_delay,axis=1)
        self.signal.data_sample = self.signal.data_sample[:,:self.signal.sym_length*self.signal.sps]


        print(
            '\t------------------------- base band pulse_shaping complete------------------------------')

    def _generate_symbol(self, plot_const=False):
        '''
            generate qam symbol,and the power of qam is unit
        :return: None
        '''

        if self.signal.mf == '16qam':

            const = loadmat('./constellation/16qam_unit_power.mat')['qam16'].reshape(4, 4)

            if plot_const:
                pass

            self.signal.msg = np.random.randint(0, 16, (2, self.length))

            self.signal.symbol = np.zeros_like(self.signal.msg, dtype=np.complex)

            for i in range(4):
                temp = const[i, 3]
                const[i, 3] = const[i, 2]
                const[i, 2] = temp

            for i in range(4):
                self.signal.symbol[self.signal.msg == i] = const[0, i]

            for i in range(4, 8):
                self.signal.symbol[self.signal.msg == i] = const[1, i - 4]

            for i in range(8, 12):
                self.signal.symbol[self.signal.msg == i] = const[3, i - 8]

            for i in range(12, 16):
                self.signal.symbol[self.signal.msg == i] = const[2, i - 12]

        if self.signal.mf == '32qam':
            const = loadmat('./constellation/qam32_unit.mat')['qam32']
            const2 = const[4:-4, :].reshape(6, 4)

            self.signal.msg = np.random.randint(0, 32, (2, self.length))
            # self.signal.msg = np.array([np.arange(0,32), np.arange(0, 32)])
            self.signal.symbol = np.zeros_like(self.signal.msg, dtype=np.complex)

            for i in range(6):
                temp = const2[i, 3]
                const2[i, 3] = const2[i, 2]
                const2[i, 2] = temp

            for i in range(4, 8):
                self.signal.symbol[self.signal.msg == i] = const2[0, i - 4]
            for i in range(12, 16):
                self.signal.symbol[self.signal.msg == i] = const2[1, i - 12]
            for i in range(8, 12):
                self.signal.symbol[self.signal.msg == i] = const2[2, i - 8]
            for i in range(24, 28):
                self.signal.symbol[self.signal.msg == i] = const2[3, i - 24]

            for i in range(28, 32):
                self.signal.symbol[self.signal.msg == i] = const2[4, i - 28]
            for i in range(20, 24):
                self.signal.symbol[self.signal.msg == i] = const2[5, i - 20]

            first_line = const[0:4, :].tolist()
            temp = first_line[3]
            first_line[3] = first_line[2]
            first_line[2] = temp

            for i in range(4):
                self.signal.symbol[self.signal.msg == i] = first_line[i]

            end_line = const[-4:, :].tolist()
            temp = end_line[3]
            end_line[3] = end_line[2]
            end_line[2] = temp

            for i in range(16, 20):
                self.signal.symbol[self.signal.msg == i] = end_line[i - 16]



class Laser(object):

    def __init__(self,frequence,signal,power=None):
        '''

        :param frequence: Thz
        :param power: dbm
        '''
        self.frequence = frequence
        self.phase_noise = None
        self.intensity_noise = None

        if power is None:
            self.power = signal.power
        else:
            self.power =power

    def LO(self,siganl:Signal_desc,frequence_offset):
        pass

    def transimit(self,signal:Signal_desc.Signal):
        signal.data_sample = signal.data_sample/np.sqrt(np.mean(signal.data_sample*np.conj(signal.data_sample),axis=1).reshape(2,1))
        power = 10**(self.power/10)/1000
        power = power/2
        signal.data_sample[0,:] = signal.data_sample[0,:]*np.sqrt(power)
        signal.data_sample[1,:] = signal.data_sample[1,:]*np.sqrt(power)

if __name__ == '__main__':
    signal = Signal_desc.Signal(mf='16qam')
    trans = Transimitter_Electrical_Signal(signal, 2 ** 16)
    trans.transimit(rc_span=1024)

    from vision_dom import vision

    vision.plot_const(signal.symbol)
