from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Function to get the closest expected systole duration for a given BPM
def get_closest_systole_duration(bpm, systole_duration_reference):
    closest_bpm = min(systole_duration_reference.keys(), key=lambda x: abs(x - bpm))
    return systole_duration_reference[closest_bpm]

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

# Read Systole duration reference from a CSV file
import csv

systole_duration_reference = {}

with open("ExpectedSystoleLength.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            bpm, duration = int(row[0]), float(row[1])
            systole_duration_reference[bpm] = duration

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

# Loop through the audio data to perform smart peak detection
for i in range(len(audio_data_mono) - 1):
    current_amplitude = audio_data_mono[i]
    if last_peak_time is None or ((i / sample_rate) - last_peak_time > 0.5):
        last_peak_time = None
    if current_amplitude > 5000: # Lower this number to make more sensitive
        if last_peak_time is None or ((i / sample_rate) - last_peak_time >= 0.1):
            smart_peaks.append(i)
            last_peak_time = i / sample_rate
            if len(smart_peaks) >= 2:
                time_between_peaks = (smart_peaks[-1] - smart_peaks[-2]) / sample_rate
                current_bpm = 60 / time_between_peaks
            if current_bpm is not None:
                expected_systole_duration = get_closest_systole_duration(current_bpm / 2, systole_duration_reference)
                if (i / sample_rate) - last_peak_time > expected_systole_duration:
                    dummy_peak_position = int((last_peak_time + expected_systole_duration) * sample_rate)
                    smart_peaks.append(dummy_peak_position)
                    last_peak_time = dummy_peak_position / sample_rate

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

# Plot smart BPM over time
plt.figure(figsize=(15, 6))
plt.plot(time_windows, smart_bpm_values, marker='o', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('BPM')
plt.title('BPM Over Time')
plt.grid(True)
plt.show()
