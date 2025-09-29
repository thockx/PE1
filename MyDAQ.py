import nidaqmx as dx
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import square, sawtooth

class MyDAQ:
    def compute_ifft(self, fft_data):
        """
        Compute inverse FFT for each channel's FFT data.
        fft_data: list of arrays (as returned by np.fft.rfft)
        Returns: list of reconstructed time-domain signals (one per channel)
        """
        if isinstance(fft_data, np.ndarray) and fft_data.ndim == 1:
            fft_data = [fft_data]
        signals = []
        for data in fft_data:
            signal = np.fft.irfft(data)
            signals.append(signal)
        return signals
    def compute_fft(self, voltages, rate):
        """
        Compute FFT for each channel in voltages and return frequency, magnitude, and phase arrays.
        Returns: list of (freqs, magnitude, phase) for each channel.
        """
        results = []
        if voltages.ndim == 1:
            voltages = [voltages]
        for v in voltages:
            N = len(v)
            freqs = np.fft.rfftfreq(N, d=1/rate)
            fft_vals = np.fft.rfft(v)
            magnitude = 20 * np.log10(np.abs(fft_vals))
            phase = np.angle(fft_vals, deg=True)
            results.append((freqs, magnitude, phase))
        return results

    def plot_bode(self):
        """
        Perform FFT on the last input signal(s) and plot a Bode plot (magnitude and phase).
        Assumes self.voltages contains the input signal(s) and self.last_rate is the sample rate.
        """
        if self.voltages is None or self.last_rate is None:
            print("No input signal data available for FFT.")
            return
        results = self.compute_fft(self.voltages, self.last_rate)
        plt.figure(figsize=(12, 5))
        for i, (freqs, magnitude, phase) in enumerate(results):
            plt.subplot(2, 1, 1)
            plt.plot(freqs, magnitude, label=f'Channel {i+1}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.title('Bode Plot - Magnitude')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(freqs, phase, label=f'Channel {i+1}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (degrees)')
            plt.title('Bode Plot - Phase')
            plt.legend()
        plt.tight_layout()
        plt.show()
    def __init__(self, device_name='myDAQ1'):
        self.device_name = device_name
        self.voltages = None
        self.times = None
        self.last_rate = None

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

    def generate_waveform(self, shape, frequency, amplitude, duration, rate):
        """
        Generate a waveform array based on the given parameters.
        """
        max_voltage = 5.0
        if abs(amplitude) > max_voltage:
            amplitude = np.sign(amplitude) * max_voltage
            print(f"WARNING: Voltages above ±{max_voltage}V are not allowed. Amplitude capped at {amplitude}V.")
        t = np.arange(0, duration, 1 / rate)
        if shape == "sine" or shape == "sin":
            wave = amplitude * np.sin(2 * np.pi * frequency * t)
        elif shape == "cosine" or shape == "cos":
            wave = amplitude * np.cos(2 * np.pi * frequency * t)
        elif shape == "tangent" or shape == "tan":
            wave = amplitude * np.tan(2 * np.pi * frequency * t)
        elif shape == "square":
            wave = amplitude * square(2 * np.pi * frequency * t)
        elif shape == "sawtooth":
            wave = amplitude * sawtooth(2 * np.pi * frequency * t)
        elif shape == "triangle":
            wave = amplitude * sawtooth(2 * np.pi * frequency * t, width=0.5)
        else:
            raise ValueError(f"Unsupported waveform shape: {shape}")
        return np.clip(wave, -max_voltage, max_voltage)

    def run_task(self, output_waveform=None, input_channels=None, output_channels=None, rate=1000, duration=1.0):
        """
        Flexible function to handle output, input, or simultaneous output/input.
        - output_waveform: np.array or list of arrays (for multi-channel output)
        - input_channels: list of input channel names (e.g., ['ai0'])
        - output_channels: list of output channel names (e.g., ['ao0'])
        - rate: sample rate
        - duration: duration in seconds
        """
        n_samples = int(rate * duration)
        data_in = None
        # Treat empty lists as None
        if input_channels is not None and len(input_channels) == 0:
            input_channels = None
        if output_channels is not None and len(output_channels) == 0:
            output_channels = None
        # Output and Input (both provided and not None)
        if output_waveform is not None and output_channels is not None and input_channels is not None:
            print("[run_task] Simultaneous output and input (output task started after input task).")
            with dx.Task() as ao_task, dx.Task() as ai_task:
                for ch in output_channels:
                    ao_task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{ch}')
                ao_task.timing.cfg_samp_clk_timing(rate,
                    sample_mode=dx.constants.AcquisitionType.FINITE,
                    samps_per_chan=n_samples)
                for ch in input_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(f'{self.device_name}/{ch}')
                ai_task.timing.cfg_samp_clk_timing(rate,
                    sample_mode=dx.constants.AcquisitionType.FINITE,
                    samps_per_chan=n_samples)
                # Preload output buffer, do not start yet
                ao_task.write(output_waveform, auto_start=False)
                # Start input task first
                ai_task.start()
                # Start output task (begins output)
                ao_task.start()
                ao_task.wait_until_done(timeout=duration+2)
                data_in = ai_task.read(number_of_samples_per_channel=n_samples)
                ai_task.wait_until_done(timeout=duration+2)
        # Output only
        elif output_waveform is not None and output_channels is not None:
            with dx.Task() as ao_task:
                for ch in output_channels:
                    ao_task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{ch}')
                ao_task.timing.cfg_samp_clk_timing(rate,
                    sample_mode=dx.constants.AcquisitionType.FINITE,
                    samps_per_chan=n_samples)
                ao_task.write(output_waveform, auto_start=True)
                time.sleep(n_samples/rate + 0.001)
                ao_task.stop()
        # Input only
        elif input_channels is not None:
            with dx.Task() as ai_task:
                for ch in input_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(f'{self.device_name}/{ch}')
                ai_task.timing.cfg_samp_clk_timing(rate,
                    sample_mode=dx.constants.AcquisitionType.FINITE,
                    samps_per_chan=n_samples)
                data_in = ai_task.read(number_of_samples_per_channel=n_samples)
        # Store and plot if input
        if data_in is not None:
            self.voltages = np.array(data_in)
            self.times = np.arange(n_samples) / rate
            self.last_rate = rate
            self.plotVoltages()
        elif output_waveform is not None:
            self.voltages = np.array(output_waveform)
            self.times = np.arange(n_samples) / rate
            self.last_rate = rate

daq = MyDAQ()

# Execution
wave1 = daq.generate_waveform(shape="sin", frequency=1, amplitude=1.0, duration=1.0, rate=100000)
wave2 = daq.generate_waveform(shape="square", frequency=3, amplitude=1.0, duration=1.0, rate=100000)
output_waves = np.vstack([wave1, wave2])
daq.run_task(output_waveform=output_waves, output_channels=["ao0", "ao1"], input_channels=["ai0","ai1"], rate=100000, duration=4.0)
daq.plot_bode()