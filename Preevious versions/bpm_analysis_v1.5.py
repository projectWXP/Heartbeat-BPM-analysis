from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import warnings

# Load audio data
import os

# Read any .wav file inside the "input_data" folder
wav_files = [f for f in os.listdir("input_data") if f.endswith('.wav')]
if not wav_files:
    raise FileNotFoundError("No .wav files found in the 'input_data' folder")

# Use the first .wav file found
wav_file_path = os.path.join("input_data", wav_files[0])

#ignore warning generated by the wavfile.read function. can most likely ignore
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sample_rate, audio_data = wavfile.read(wav_file_path)

# Convert to mono if it's stereo
if len(audio_data.shape) == 2:
    audio_data_mono = np.mean(audio_data, axis=1).astype(np.int16)
else:
    audio_data_mono = audio_data

# Calculate rolling average and standard deviation
window_time = 1  # Size of the window to calculate average and std deviation, in seconds
window_size = int(window_time * sample_rate)
rolling_avg = signal.fftconvolve(np.abs(audio_data_mono), np.ones(window_size) / window_size, mode='same')
rolling_std = signal.fftconvolve(np.abs(audio_data_mono - rolling_avg), np.ones(window_size) / window_size, mode='same') ** 0.5

# Perform local normalization
alpha = .2  # can adjust, higher for weaker normalization
audio_data_normalized = (audio_data_mono - rolling_avg) / (rolling_std + alpha)

# Length of the audio in seconds
audio_length_seconds = len(audio_data_normalized) / sample_rate

# Create an array of time windows for calculating BPM
time_windows = np.arange(0, audio_length_seconds - 5, 1)

# Initialize variables
s1_s2_Divider = 1 #(1) if s1 or s2 is signifigantly louder, (2) if both beats are detected
threshold_multiplier = 2 # can adjust, higher for higher threshold
cooldown_multiplier = 0.7 # can adjust, xx% of the time between each beat
smart_peaks = []#detected peaks
current_bpm = 120  # Initialize with a reasonable default
dynamic_threshold = np.mean(np.abs(audio_data_normalized)) * threshold_multiplier

# Initialize next_peak_allowed_time array
next_peak_allowed_time = np.zeros(len(audio_data_normalized))

# Loop through the audio data to perform smart peak detection
for i in range(len(audio_data_normalized) - 1):
    current_time = i / sample_rate
    current_amplitude = audio_data_normalized[i]
  
    if current_time >= next_peak_allowed_time[i]:
        if current_amplitude > dynamic_threshold:
            # Detected a peak
            smart_peaks.append(i)
            
            # Update current_bpm based on last two peaks
            if len(smart_peaks) >= 2:
                time_between_peaks = (smart_peaks[-1] - smart_peaks[-2]) / sample_rate
                current_bpm = 60 / time_between_peaks
                
            # Update next_peak_allowed_time to enforce cooldown
            cooldown_samples = int(sample_rate * (60 / current_bpm) * cooldown_multiplier)
            next_allowed_time = current_time + (cooldown_samples / sample_rate)
            next_peak_allowed_time[i:i+cooldown_samples] = next_allowed_time

# Convert to a NumPy array for easier manipulation
smart_peaks = np.array(smart_peaks)
smart_peaks = np.unique(smart_peaks)

# Exit Script if no peaks detected
if smart_peaks.size == 0:
    print("No peaks detected")
    sys.exit()

# Calculate time instances for these smartly detected peaks
smart_peak_times = smart_peaks / sample_rate

# Initialize array to store smart BPM values
smart_bpm_values = []

# Calculate BPM for each time window using smart peaks
for start_time in time_windows:
    end_time = start_time + 5
    peaks_in_window = smart_peak_times[(smart_peak_times >= start_time) & (smart_peak_times < end_time)]
    if len(peaks_in_window) >= 2:
        avg_bpm = (60 * len(peaks_in_window) / (peaks_in_window[-1] - peaks_in_window[0])) / s1_s2_Divider
        smart_bpm_values.append(avg_bpm)
    else:
        smart_bpm_values.append(np.nan)

# Visualization
segment_start = 1  # Start time in seconds
segment_end = 100  # End time in seconds
segment_start_sample = int(segment_start * sample_rate)
segment_end_sample = int(segment_end * sample_rate)

# Create the figure
fig = go.Figure()

# Add the waveform trace
fig.add_trace(go.Scatter(
    x=np.arange(segment_start_sample, segment_end_sample) / sample_rate,
    y=audio_data_mono[segment_start_sample:segment_end_sample],
    mode='lines',
    name='Waveform'
))

# Add the peaks trace
segment_smart_peaks = smart_peaks[(smart_peaks >= segment_start_sample) & (smart_peaks < segment_end_sample)]
fig.add_trace(go.Scatter(
    x=segment_smart_peaks / sample_rate,
    y=audio_data_mono[segment_smart_peaks],
    mode='markers',
    marker=dict(size=10, color='red'),
    name='Peaks'
))

# Layout customization
fig.update_layout(
    title="Detected Peaks in Segment",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="linear"
    ),
    dragmode='pan'
)

# Show the figure
fig.show(config={'scrollZoom': True})

# Plot smart BPM over time
plt.figure(figsize=(15, 6))
plt.plot(time_windows, smart_bpm_values, marker='o', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('BPM')
plt.title('BPM Over Time')
plt.grid(True)
plt.show()
