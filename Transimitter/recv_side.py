from scipy.io import loadmat
from scipy.signal import fftconvolve, upfirdn
from Signal_interface import Signal_desc
from scipy.signal import group_delay
import numpy as np


class Receiver(object):

    def __init__(self, recv_signal: Signal_desc.Signal):
        self.signal = recv_signal

    def matched_filter(self, filter_h,delay):
        '''

        :param filter_h: the impulse respones of the mathced filter, and will be used to do match filter against the
        received signal

        :return: Signal object , the object's modulation format, sps,symbol_rate and the roll_off factor will be set,
        however other attribute such as symbol are None by default, because thest attribute will only can be get after
        dsp processing

        '''

        print('\t------------------------- run matched filter operation-------------------------')
        matched_filter_data_sample = fftconvolve(self.signal.data_sample, filter_h)
        # matched_filter_data_sample = upfirdn(filter_h[0],self.signal.data_sample,1,1,axis=1)
        print('\t------------------------- matched filter complete------------------------------')
        # delay = int((len(filter_h[0])-1)/4/2 * signal.sps)
        matched_filter_data_sample = np.roll(matched_filter_data_sample, -delay, axis=1)
        # matched_filter_data_sample = np.roll(matched_filter_data_sample, -delay, axis=1)
        matched_filter_data_sample = matched_filter_data_sample[:,:self.signal.sym_length*self.signal.sps]

        matched_signal = Signal_desc.Signal(symbol_rate=self.signal.symbol_rate, sps=self.signal.sps, mf=self.signal.mf,
                                            roll_off=self.signal.roll_off)

        matched_signal.data_sample = matched_filter_data_sample
        matched_signal.sym_length = self.signal.sym_length
        return matched_signal

    def hard_decision_symbol(self, matched_signal):
        '''

        :param matched_signal: do hard decision on matched_signal.symbol, the matched_signal.data_sample should
         be 1 sps after some dsp processing

        :return: 还没想好
        '''
        # normalization,x-pol and y-pol's power will be normalized to one
        assert len(matched_signal.data_sample[0, :]) == matched_signal.sym_length

        matched_signal.data_sample = matched_signal.data_sample / np.sqrt(
            np.mean(matched_signal.data_sample * np.conj(matched_signal.data_sample), axis=1).reshape(2, 1))

        if self.signal.mf == '16qam':
            const = loadmat('./constellation/16qam_unit_power.mat')['qam16']
            const.shape = 1, 16

        elif self.signal.mf == '32qam':
            const = loadmat('./constellation/qam32_unit.mat')['qam32']
            const.shape = 1, 32
        else:
            raise Exception("const error")

        matched_signal_datasample_x = matched_signal.data_sample[0, :]
        matched_signal_datasample_y = matched_signal.data_sample[1,:]

        distance_x = []
        distance_y = []
        decision_x = []
        decision_y = []
        for x_sample in matched_signal_datasample_x:
            for symbol in (const[0, :]):
                distance_x.append(abs(x_sample - symbol))

            decision_x.append(const[0, np.argmin(distance_x)])
            distance_x = []

        for y_sample in matched_signal_datasample_y:
            for symbol in (const[0, :]):
                distance_y.append(abs(y_sample - symbol))

            decision_y.append(const[0, np.argmin(distance_y)])
            distance_y = []
        matched_signal.symbol = np.array([decision_x, decision_y])

    def decode_msg(self, matched_signal):
        '''

        :param matched_signal: turn symbol into msg
        :return:

        '''

        if matched_signal.mf == '16qam':
            const = loadmat('./constellation/16qam_unit_power.mat')['qam16'].reshape(4, 4)

            matched_signal.msg = np.zeros_like(matched_signal.symbol, dtype=np.int)

            x_pol = matched_signal.msg[0, :]
            y_pol = matched_signal.msg[1, :]

            for i in range(4):
                temp = const[i, 3]
                const[i, 3] = const[i, 2]
                const[i, 2] = temp

            for i in range(4):
                x_pol[np.abs(matched_signal.symbol[0, :] - const[0, i])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const[0, i])<np.spacing(1)] = i

            for i in range(4, 8):
                x_pol[np.abs(matched_signal.symbol[0, :] - const[1, i - 4])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const[1, i - 4])<np.spacing(1)] = i
            for i in range(8, 12):
                x_pol[np.abs(matched_signal.symbol[0, :] - const[3, i - 8])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const[3, i - 8])<np.spacing(1)] = i
            for i in range(12, 16):
                x_pol[np.abs(matched_signal.symbol[0, :] - const[2, i - 12])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const[2, i - 12])<np.spacing(1)] = i
            # matched_signal.symbol = np.array([x_pol, y_pol])

        if matched_signal.mf == '32qam':
            const = loadmat('./constellation/qam32_unit.mat')['qam32']
            const2 = const[4:-4, :].reshape(6, 4)

            matched_signal.msg = np.zeros_like(matched_signal.symbol, dtype=np.int)

            x_pol = matched_signal.msg[0, :]
            y_pol = matched_signal.msg[1, :]

            for i in range(6):
                temp = const2[i, 3]
                const2[i, 3] = const2[i, 2]
                const2[i, 2] = temp

            for i in range(4, 8):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[0, i - 4])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[0, i - 4])<np.spacing(1)] = i
            for i in range(12, 16):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[1, i - 12])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[1, i - 12])<np.spacing(1)] = i
            for i in range(8, 12):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[2, i - 8])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[2, i - 8])<np.spacing(1)] = i
            for i in range(24, 28):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[3, i - 24])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[3, i - 24])<np.spacing(1)] = i
            for i in range(28, 32):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[4, i - 28])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[4, i - 28])<np.spacing(1)] = i
            for i in range(20, 24):
                x_pol[np.abs(matched_signal.symbol[0, :] - const2[5, i - 20])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1, :] - const2[5, i - 20])<np.spacing(1)] = i

            # matched_signal.symbol = np.array([x_pol, y_pol])
            first_line = const[0:4, :].tolist()
            temp = first_line[3]
            first_line[3] = first_line[2]
            first_line[2] = temp

            for i in range(4):
                x_pol[np.abs(matched_signal.symbol[0,:] - first_line[i])<np.spacing(1)] = i
                y_pol[np.abs(matched_signal.symbol[1,:] - first_line[i])<np.spacing(1)] = i

            end_line = const[-4:, :].tolist()
            temp = end_line[3]
            end_line[3] = end_line[2]
            end_line[2] = temp

            for i in range(16, 20):
               x_pol[np.abs(matched_signal.symbol[0,:] - end_line[i-16])<np.spacing(1)] = i
               y_pol[np.abs(matched_signal.symbol[1, :] - end_line[i - 16])<np.spacing(1)] = i

    def _biterr_help(self, msg1, msg2):
        assert len(msg1) == len(msg2)
        c = 0
        for i in range(len(msg1)):
            if msg1[i] != msg2[i]:
                c += 1
        return c

    def biterr(self, msg1, msg2, modulation_order):
        '''

        :param msg1:
        :param msg2:
        :param modulation_order:
        :return: biterror of the system
        '''
        # assert msg1.dim == 2
        # assert msg2.dim == 2
        assert msg1.shape == msg2.shape
        c = 0
        for i in range(len(msg1)):
            msg1_bit = np.binary_repr(msg1[i], modulation_order)
            msg2_bit = np.binary_repr(msg2[i], modulation_order)
            c += self._biterr_help(msg1_bit, msg2_bit)

        return c / len(msg1) / modulation_order

    def transimit(self, matched_filter):
        print('\t-------------------------matched filter ^_^-------------------------')
        matched_signal = self.matched_filter(matched_filter)
        print('\t-------------------------matched filter completed^_^--------------------------')

        # assert matched_filter.data_sample




if __name__ == '__main__':
    from Transimitter.Trans_side import Transimitter_Electrical_Signal

    signal = Signal_desc.Signal(mf='16qam')
    trans = Transimitter_Electrical_Signal(signal, 10)
    # trans._generate_symbol()
    trans.transimit(rc_span=1024)
    import up_down_sample

    rece = Receiver(signal)
    matched_signal = rece.matched_filter(trans.pulse_shaping,trans.pulse_shaping_delay)
    matched_signal.data_sample = up_down_sample.downsample(matched_signal.data_sample, signal.sps)
    matched_signal.data_sample = matched_signal.data_sample[:,:signal.sym_length]
    rece.hard_decision_symbol(matched_signal)
    rece.decode_msg(matched_signal)
    print(matched_signal.msg)
    print('\n')
    print(signal.msg)