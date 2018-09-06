from Signal_interface.Signal_desc import  Signal
import numpy as np

class QualityMeter(object):

    @staticmethod
    def measure_snr(sym1:np.ndarray,sym2:np.ndarray):
        '''

        :param sym1: trans end signal 1d array
        :param sym2: receive signal and sym1 - sym2 will be noise 1d array
        :return: linear snr and snr in db
        '''
        # assert sym1.ndim == 1
        # assert sym2.ndim == 1
        sym1 = sym1/np.sqrt(np.mean(np.abs(sym1)**2,axis=1).reshape(2,1))
        sym2 = sym2 / np.sqrt(np.mean(np.abs(sym2)**2,axis=1).reshape(2,1))

        noise = sym1-sym2
        noise_power = QualityMeter.power_meter(noise)[0]

        snr = QualityMeter.power_meter(sym2)[0]/noise_power
        snr_db = 10*np.log10(snr)
        return np.real(snr),np.real(snr_db)

    @staticmethod
    def power_meter(signal):
        '''

        :param signal: Signal object
        :return: the power of signal
        '''
        if isinstance(signal,Signal):
            datasample = signal.data_sample
            power = np.mean(np.abs(datasample[0,:])**2)+np.mean(np.abs(datasample[1,:])**2)

            power_dbm = 10*np.log10(power*1000)
            print(f'power is {power}W, {power_dbm} dbm')


            return power,power_dbm
        elif isinstance(signal,np.ndarray):
            if signal.ndim == 2:
                power = np.mean(np.abs(signal[0, :])**2) + np.mean(np.abs(signal[1, :])**2)
                power_dbm = 10 * np.log10(power * 1000)
                # print(f'power is {power}W, {power_dbm} dbm')

                return power, power_dbm