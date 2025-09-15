import nidaqmx as dx
import numpy as np
import matplotlib.pyplot as plt
import time

class MyDAQ:
    def __init__(self, device_name='myDAQ2'):
        self.device_name = device_name
        self.voltages = None
        self.times = None
        self.last_rate = None

    def write(self, array, rate, channel='ao0'):
        """Write an array of voltages to a single output channel."""
        with dx.Task() as writeTask:
            writeTask.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{channel}')
            samps_per_chan = len(array)
            writeTask.timing.cfg_samp_clk_timing(rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=samps_per_chan)
            writeTask.write(array, auto_start=True)
            time.sleep(samps_per_chan/rate + 0.001)
            writeTask.stop()
        self.voltages = np.array(array)
        self.times = np.arange(samps_per_chan) / rate
        self.last_rate = rate

    def write_multi(self, arrays, rate, channels=['ao0','ao1']):
        """Write arrays of voltages to multiple output channels."""
        with dx.Task() as writeTask:
            for ch in channels:
                writeTask.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{ch}')
            samps_per_chan = len(arrays[0])
            writeTask.timing.cfg_samp_clk_timing(rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=samps_per_chan)
            writeTask.write(arrays, auto_start=True)
            time.sleep(samps_per_chan/rate + 0.001)
            writeTask.stop()
        self.voltages = np.array(arrays)
        self.times = np.arange(samps_per_chan) / rate
        self.last_rate = rate

    def read(self, rate, duration, channel='ai0', n_samples=None):
        """Read voltages from a single input channel for a given duration."""
        if n_samples is None:
            n_samples = int(rate * duration)
        with dx.Task() as readTask:
            readTask.ai_channels.add_ai_voltage_chan(f'{self.device_name}/{channel}')
            readTask.timing.cfg_samp_clk_timing(rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=n_samples)
            data = readTask.read(number_of_samples_per_channel=n_samples)
        self.voltages = np.array(data)
        self.times = np.arange(n_samples) / rate
        self.last_rate = rate
        return self.voltages

    def read_multi(self, rate, duration, channels=['ai0','ai1'], n_samples=None):
        """Read voltages from multiple input channels for a given duration."""
        if n_samples is None:
            n_samples = int(rate * duration)
        with dx.Task() as readTask:
            for ch in channels:
                readTask.ai_channels.add_ai_voltage_chan(f'{self.device_name}/{ch}')
            readTask.timing.cfg_samp_clk_timing(rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=n_samples)
            data = readTask.read(number_of_samples_per_channel=n_samples)
        self.voltages = np.array(data)
        self.times = np.arange(n_samples) / rate
        self.last_rate = rate
        return self.voltages

    def getVoltData(self):
        """Return last recorded/generated voltage data."""
        return self.voltages

    def getTimeData(self):
        """Return time array for last recorded/generated signal."""
        return self.times

    def plotVoltages(self):
        """Plot the last voltage data against time."""
        if self.voltages is not None and self.times is not None:
            plt.figure()
            if self.voltages.ndim == 1:
                plt.plot(self.times, self.voltages)
            else:
                for i, v in enumerate(self.voltages):
                    plt.plot(self.times, v, label=f'Channel {i+1}')
                plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.title('Voltage vs Time')
            plt.show()
        else:
            print("No voltage data to plot.")