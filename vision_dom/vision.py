import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as pygo
plot = py.offline.plot

def plot_const(signal_sample:np.ndarray):
    '''

    :param signal_sample: np array
    :return:
    '''

    if signal_sample.ndim == 2:
        t1 = pygo.Scattergl({'x': np.real(signal_sample[0, :]), 'y': np.imag(signal_sample[0, :]), 'mode': 'markers',
                           'marker': {'color': 'blue'}})

        layout = dict(title='scatter plot x-pol',
                      yaxis=dict(title='Q',zeroline=False),
                      xaxis=dict(title='I',zeroline=False)
                      )
        plot({'data':[t1],'layout':layout})
        # plt.plot(np.real(signal_sample[0,:]),np.imag(signal_sample[0,:]),'bo')
        # plt.xlabel('I')
        # plt.ylabel('Q')
        # plt.title('x-pol')
        # plt.figure()
        # plt.plot(np.real(signal_sample[1, :]), np.imag(signal_sample[1, :]), 'bo')
        # plt.xlabel('I')
        # plt.ylabel('Q')
        # plt.title('y-pol')
        t1 = pygo.Scattergl({'x': np.real(signal_sample[1, :]), 'y': np.imag(signal_sample[1, :]), 'mode': 'markers',
                             'marker': {'color': 'blue'}})

        layout = dict(title='scatter plot y-pol',
                      yaxis=dict(title='Q',zeroline=False),
                      xaxis=dict(title='I',zeroline=False)
                      )
        plot({'data': [t1], 'layout': layout})

    else:
        plt.figure()
        plt.plot(np.real(signal_sample), np.imag(signal_sample), 'bo')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title('x-pol')


