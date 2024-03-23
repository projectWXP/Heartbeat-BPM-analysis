from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt

# Load audio data
import os

# Read any .wav file inside the "input_data" folder
wav_files = [f for f in os.listdir("input_data") if f.endswith('.wav')]
if not wav_files:
    raise FileNotFoundError("No .wav files found in the 'input_data' folder")

# Use the first .wav file found
wav_file_path = os.path.join("input_data", wav_files[0])
sample_rate, audio_data = wavfile.read(wav_file_path)

# Convert to mono if it's stereo
if len(audio_data.shape) == 2:
    audio_data_mono = np.mean(audio_data, axis=1).astype(np.int16)
else:
    audio_data_mono = audio_data

# Length of the audio in seconds
audio_length_seconds = len(audio_data_mono) / sample_rate

# Create an array of time windows for calculating BPM
time_windows = np.arange(0, audio_length_seconds - 5, 1)

# Initialize an array to store "smart" detected peaks
smart_peaks = []

# Initialize a variable to store the current BPM
current_bpm = None

# Initialize a variable to store the time of the last detected peak
last_peak_time = None

# Initialize the tracking and high_points list variables
tracking = False
high_points = []

# Initialize cooldown
cooldown = 0 

# Initialize variables
window_time = 0.2  # Calculate the average amplitude for a set window, in seconds (you can adjust this)
window_size = int(window_time * sample_rate)
sensitivity_factor = 1.2  # Sensitivity factor for dynamic threshold (you can adjust this)
dynamic_threshold = 5000  # Initial threshold, Lower this number to make more sensitive (you can adjust this)
print("debug1")
# Compute the rolling average of the audio data
rolling_avg = fftconvolve(np.abs(audio_data_mono), np.ones(window_size) / window_size, mode='same')

# Loop through the audio data to perform smart peak detection
for i in range(len(audio_data_mono) - 1):
    print("debug starting loop...")
    if cooldown > 0: 
        cooldown -= 1
    current_amplitude = audio_data_mono[i]
    
    # Calculate average amplitude for the current window
    start = max(0, i - window_size)
    end = min(len(audio_data_mono), i + window_size)
    average_amplitude = rolling_avg[i]

    # Update the dynamic threshold
    dynamic_threshold = average_amplitude * sensitivity_factor
    
    # Start tracking when amplitude crosses dynamic threshold
    if cooldown == 0 and current_amplitude > dynamic_threshold:
        tracking = True

    # While tracking, add the high points to the list
    if tracking:
        high_points.append(current_amplitude)

    # Stop tracking when amplitude goes below dynamic threshold, mark the highest peak
    if tracking and current_amplitude < 5000:  # Use the same threshold as above
        highest_point_index = i  # Assuming i is the index of the highest point
        highest_point = max(high_points)
        smart_peaks.append(highest_point_index)
        high_points = []
        tracking = False

        # Update last_peak_time and current_bpm based on the detected peak
        last_peak_time = i / sample_rate
        if len(smart_peaks) >= 2:
            time_between_peaks = (smart_peaks[-1] - smart_peaks[-2]) / sample_rate
            current_bpm = 60 / time_between_peaks
            cooldown = int(sample_rate * (60 / current_bpm) * 0.7)  # 70% of the time between each beat

# Convert to a NumPy array for easier manipulation
smart_peaks = np.array(smart_peaks)
smart_peaks = np.unique(smart_peaks)

# Calculate time instances for these smartly detected peaks
smart_peak_times = smart_peaks / sample_rate

# Initialize array to store smart BPM values
smart_bpm_values = []

# Calculate BPM for each time window using smart peaks
for start_time in time_windows:
    end_time = start_time + 5
    peaks_in_window = smart_peak_times[(smart_peak_times >= start_time) & (smart_peak_times < end_time)]
    if len(peaks_in_window) >= 2:
        avg_bpm = (60 * len(peaks_in_window) / (peaks_in_window[-1] - peaks_in_window[0])) / 2
        smart_bpm_values.append(avg_bpm)
    else:
        smart_bpm_values.append(np.nan)
print("debug2")
# Plot smart BPM over time
plt.figure(figsize=(15, 6))
plt.plot(time_windows, smart_bpm_values, marker='o', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('BPM')
plt.title('BPM Over Time')
plt.grid(True)
plt.show()

# Visualization
segment_start = 10  # Start time in seconds
segment_end = 18  # End time in seconds
segment_start_sample = int(segment_start * sample_rate)
segment_end_sample = int(segment_end * sample_rate)

# Plot the audio data and detected peaks for the given segment
plt.figure(figsize=(15, 6))
plt.plot(np.arange(segment_start_sample, segment_end_sample) / sample_rate, audio_data_mono[segment_start_sample:segment_end_sample], label="Waveform")
plt.scatter(smart_peaks[(smart_peaks >= segment_start_sample) & (smart_peaks < segment_end_sample)] / sample_rate, audio_data_mono[smart_peaks[(smart_peaks >= segment_start_sample) & (smart_peaks < segment_end_sample)]], color="red", label="Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Detected Peaks in Segment")
plt.legend()
plt.grid(True)
plt.show()
