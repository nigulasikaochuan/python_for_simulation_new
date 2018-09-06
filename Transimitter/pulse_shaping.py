import numpy as np


def rrc_filter(span, alpha, sps, shape='rc'):
    '''

    :param span: the number of rrc to be cut
    :param alpha: the roll_off factor
    :param sps: the sample per span
    :return: h

    '''

    delay = span * sps / 2
    t = np.linspace(-delay/sps, delay/sps, sps * span + 1)
    h = np.zeros_like(t)
    if shape == 'rc' or shape == 'normal':
        print('\t-------------------------rc filter will be design-----------------------------------')
        denom = (1 - (2 * alpha * t) ** 2)
        idx1 = np.abs(denom) > np.sqrt(np.spacing(1))
        h[idx1] = np.sinc(t[idx1]) * (np.cos(np.pi * alpha * t[idx1]) / denom[idx1]) / sps

        idx2 = np.abs(denom) < np.sqrt(np.spacing(1))
        h[idx2] = alpha * np.sin(np.pi / 2 / 2 / alpha) / 2 / sps
        print('\t------------------------- rc filter design completed-------------------------')
    if shape == 'rrc' or shape == 'sqrt':
        print('\t------------------------- rrc filter will be designed-----------------------------------')
        idx1 = t == 0
        if any(idx1):
            h[idx1] = -1 / (np.pi * sps) * (np.pi * (alpha - 1) - 4 * alpha)

        idx2 = np.abs(np.abs(4 * alpha * t) - 1) < np.sqrt(np.spacing(1))
        if any(idx2):
            h[idx2] = 1 / (2 * np.pi * sps) * (
                    np.pi * (alpha + 1) * np.sin(np.pi * (alpha + 1) / (4 * alpha)) - 4 * alpha * np.sin(
                np.pi * (alpha - 1) / 4 / alpha) + np.pi * (alpha - 1) * np.cos(np.pi * (alpha - 1) / 4 / alpha))

        idx3 = t == t
        idx3[idx1] = False
        idx3[idx2] = False
        nind = t[idx3]
        h[idx3] = -4 * alpha / sps * (
                np.cos((1 + alpha) * np.pi * nind) + np.sin((1 - alpha) * np.pi * nind) / (4 * alpha * nind) )/ (
                np.pi * ((4 * alpha * nind) ** 2 - 1))

        print('\t------------------------- rrc design complete-------------------------------------------')
    h = h / np.sqrt(np.sum(h**2))
    h.shape = 1, -1
    return h


if __name__ == '__main__':
    h = rrc_filter(1024, 0.2, 6, 'rrc')
