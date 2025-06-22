import os
import warnings
import csv
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocess_audio(file_path, downsample_factor=10, bandpass_freqs=(20, 150)):
    """
    Loads, filters, and preprocesses the audio file.

    Args:
        file_path (str): Path to the .wav file.
        downsample_factor (int): The factor by which to downsample the audio.
        bandpass_freqs (tuple): The frequency range (min_freq, max_freq) for the band-pass filter.

    Returns:
        tuple: A tuple containing:
            - np.array: The processed audio data (envelope).
            - int: The new, downsampled sample rate.
    """
    # Ignore harmless warnings from reading WAV files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # --- 1. Band-pass filter to isolate heart sounds ---
    lowcut, highcut = bandpass_freqs
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_data)

    # --- 2. Downsample for performance ---
    new_sample_rate = sample_rate // downsample_factor
    audio_downsampled = audio_filtered[::downsample_factor]

    # --- 3. Calculate the envelope of the signal ---
    # Taking the absolute value and then smoothing gives a better representation of intensity
    audio_abs = np.abs(audio_downsampled)
    # Use a simple moving average to create the envelope
    window_size = new_sample_rate // 10 # 100ms smoothing window
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values

    return audio_envelope, new_sample_rate

def find_heartbeat_peaks(audio_envelope, sample_rate, min_bpm=40, max_bpm=220, s1_s2_max_interval_sec=0.35):
    """
    Finds the peaks (heartbeats) in the audio envelope by identifying S1-S2 patterns.

    This method first detects a broad set of peaks, then analyzes the time intervals
    between them. It assumes a heartbeat consists of an S1-S2 pair (a short interval)
    followed by a longer diastolic interval. This helps avoid double-counting beats
    at lower heart rates.

    Args:
        audio_envelope (np.array): The envelope of the audio signal.
        sample_rate (int): The sample rate of the audio.
        min_bpm (int): The minimum expected heart rate (used for final filtering).
        max_bpm (int): The maximum expected heart rate (used for final filtering).
        s1_s2_max_interval_sec (float): The maximum time in seconds for an interval to be
                                       considered an S1-S2 systolic gap.

    Returns:
        np.array: An array of indices corresponding to the detected true heartbeat events.
    """
    # --- Step 1: Find ALL potential peaks ---
    # We need a small distance to ensure we can detect both S1 and S2.
    # This distance corresponds to the highest possible physiological heart rate (~240bpm).
    min_peak_distance_samples = int((60.0 / 240.0) * sample_rate)

    # Use a more sensitive prominence and height to detect both S1 and S2,
    # which can have varying amplitudes.
    prominence_threshold = np.quantile(audio_envelope, 0.6)
    height_threshold = np.mean(audio_envelope) * 0.6

    all_peaks, _ = find_peaks(
        audio_envelope,
        distance=min_peak_distance_samples,
        prominence=prominence_threshold,
        height=height_threshold
    )

    if len(all_peaks) < 2:
        print("Warning: Not enough initial peaks found to perform pattern analysis.")
        return all_peaks

    # --- Step 2: Analyze inter-peak intervals to identify true heartbeats ---
    peak_times_sec = all_peaks / sample_rate
    intervals_sec = np.diff(peak_times_sec)

    true_beat_indices = []
    i = 0
    while i < len(intervals_sec):
        # Check if the interval is a short S1-S2 interval
        if intervals_sec[i] <= s1_s2_max_interval_sec:
            # We found an S1-S2 pair. A good heuristic is to choose the one with the higher amplitude.
            peak1_idx = all_peaks[i]
            peak2_idx = all_peaks[i + 1]

            if audio_envelope[peak1_idx] >= audio_envelope[peak2_idx]:
                true_beat_indices.append(peak1_idx)
            else:
                true_beat_indices.append(peak2_idx)

            # Skip the next peak (S2) and its interval
            i += 2
        else:
            # This interval is longer than the S1-S2 threshold, so it's a diastolic gap.
            # The peak at the start of this gap is a valid beat.
            true_beat_indices.append(all_peaks[i])
            i += 1

    # Handle the very last peak if the loop didn't cover it
    if i == len(intervals_sec):
        true_beat_indices.append(all_peaks[-1])

    true_beat_indices = np.array(true_beat_indices)

    # --- Step 3: Final physiological filtering ---
    # As a final sanity check, filter out beats that are too close or too far apart
    # based on the overall min/max BPM.
    if len(true_beat_indices) > 1:
        final_peak_times = true_beat_indices / sample_rate
        final_intervals = np.diff(final_peak_times)

        min_interval = 60.0 / max_bpm
        max_interval = 60.0 / min_bpm

        filtered_peaks = [true_beat_indices[0]]
        for j in range(len(final_intervals)):
            if min_interval <= final_intervals[j] <= max_interval:
                filtered_peaks.append(true_beat_indices[j + 1])
        return np.array(filtered_peaks)
    else:
        return true_beat_indices


def calculate_bpm_series(peaks, sample_rate, smoothing_window_sec=5):
    """
    Calculates the BPM over time from the detected peaks and smooths it.

    Args:
        peaks (np.array): Array of peak indices.
        sample_rate (int): The sample rate of the audio.
        smoothing_window_sec (int): The size of the rolling window for smoothing BPM in seconds.

    Returns:
        tuple: A tuple containing:
            - pd.Series: The smoothed BPM values.
            - np.array: The time points (in seconds) corresponding to each BPM value.
    """
    if len(peaks) < 2:
        return pd.Series(dtype=np.float64), np.array([])

    # Calculate time differences between consecutive peaks
    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)

    # Calculate instantaneous BPM
    instant_bpm = 60.0 / time_diffs

    # Create a pandas Series for easy manipulation
    bpm_series = pd.Series(instant_bpm, index=peak_times[1:])

    # Apply a rolling mean to smooth the BPM data
    # The window size is based on the number of beats in the specified time window
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        beats_in_window = int(np.ceil((smoothing_window_sec / 60) * avg_heart_rate))
        if beats_in_window < 2:
            beats_in_window = 2
        smoothed_bpm = bpm_series.rolling(window=beats_in_window, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)


    # Align times to the center of the peak intervals
    time_points = (peak_times[:-1] + peak_times[1:]) / 2

    return smoothed_bpm, time_points

def plot_results(audio_envelope, peaks, smoothed_bpm, bpm_times, sample_rate, file_name):
    """
    Creates an interactive plot of the results using Plotly, combining features.

    Args:
        audio_envelope (np.array): The audio signal envelope.
        peaks (np.array): Array of detected peak indices.
        smoothed_bpm (pd.Series): Smoothed BPM data.
        bpm_times (np.array): Time points for the BPM data.
        sample_rate (int): The audio sample rate.
        file_name (str): The name of the audio file for the plot title.
    """
    # Create time axis for the waveform
    time_axis = np.arange(len(audio_envelope)) / sample_rate

    # Create figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add audio waveform trace (using the envelope)
    fig.add_trace(
        go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")),
        secondary_y=False,
    )

    # Add detected peaks trace
    fig.add_trace(
        go.Scatter(
            x=peaks / sample_rate,
            y=audio_envelope[peaks],
            mode='markers',
            name='Detected Heartbeats',
            marker=dict(color='#e36f6f', size=8)
        ),
        secondary_y=False,
    )

    # Add smoothed BPM trace
    if not smoothed_bpm.empty:
        fig.add_trace(
            go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM", line=dict(color="#4a4a4a", width=3)),
            secondary_y=True,
        )

    # Calculate and add annotations for Min, Max, and Average BPM
    if not smoothed_bpm.empty:
        max_bpm_val = smoothed_bpm.max()
        min_bpm_val = smoothed_bpm.min()
        avg_bpm_val = smoothed_bpm.mean()

        # CORRECTED LINES: Get time directly from the index of the smoothed_bpm Series
        max_bpm_time = smoothed_bpm.idxmax()
        min_bpm_time = smoothed_bpm.idxmin()

        # Add annotations
        fig.add_annotation(
            x=max_bpm_time, y=max_bpm_val,
            text=f"Max BPM: {max_bpm_val:.1f}",
            showarrow=True, arrowhead=1, ax=20, ay=-40,
            font=dict(color="#e36f6f"),
            yref="y2"
        )
        fig.add_annotation(
            x=min_bpm_time, y=min_bpm_val,
            text=f"Min BPM: {min_bpm_val:.1f}",
            showarrow=True, arrowhead=1, ax=20, ay=40,
            font=dict(color="#a3d194"),
            yref="y2"
        )
        # Position average BPM annotation
        fig.add_annotation(
            x=bpm_times[-1] if bpm_times.size > 0 else 0, y=avg_bpm_val,
            text=f"Avg BPM: {avg_bpm_val:.1f}",
            showarrow=False,
            xanchor="right", yanchor="top",
            font=dict(color="#4a4a4a"),
            yref="y2"
        )

    # Update layout and axis titles
    fig.update_layout(
        title_text=f"Heartbeat Analysis - {file_name}",
        xaxis_title="Time (seconds)",
        dragmode='pan',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[-2000, 20000]) # Adjust 1000 based on your signal's typical max
    fig.update_yaxes(title_text="Beats Per Minute (BPM)", secondary_y=True, range=[max(0, smoothed_bpm.min(skipna=True)-10) if not smoothed_bpm.empty else 0, smoothed_bpm.max(skipna=True)+10 if not smoothed_bpm.empty else 260])

    # Enable scroll to zoom and save to an HTML file
    output_html_path = f"{os.path.splitext(file_name)[0]}_bpm_plot.html"
    fig.write_html(output_html_path, config={'scrollZoom': True})
    print(f"Interactive plot saved to {output_html_path}")


def save_bpm_to_csv(bpm_series, time_points, output_path):
    """Saves the calculated BPM data to a CSV file."""
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time (s)", "BPM"])
        if not bpm_series.empty:
            for t, bpm in zip(time_points, bpm_series):
                if not np.isnan(bpm):
                    writer.writerow([f"{t:.2f}", f"{bpm:.1f}"])

def main():
    """Main function to run the BPM analysis."""
    # --- Tunable Parameters ---
    DOWNSAMPLE_FACTOR = 100       # Higher factor means faster processing but less detail
    BANDPASS_FREQS = (20, 150)   # Frequency range for heart sounds in Hz
    MIN_BPM = 40                 # Minimum expected heart rate
    MAX_BPM = 220                # Maximum expected heart rate
    SMOOTHING_WINDOW_SEC = 5     # Seconds for BPM smoothing window
    S1_S2_MAX_INTERVAL_SEC = 0.35 # *** NEW PARAMETER ***: Max time between S1 and S2 sounds

    # --- Script Execution ---
    try:
        # Find the first .wav file in the current directory
        wav_files = [f for f in os.listdir(".") if f.lower().endswith('.wav')]
        if not wav_files:
            raise FileNotFoundError("No .wav files found in the current directory.")
        wav_file_path = wav_files[0]
        file_name_no_ext = os.path.splitext(wav_file_path)[0]

        print(f"Processing file: {wav_file_path}...")

        # 1. Preprocess the audio file
        audio_envelope, sample_rate = preprocess_audio(wav_file_path, DOWNSAMPLE_FACTOR, BANDPASS_FREQS)

        # 2. Find heartbeat peaks using the new pattern-based method
        peaks = find_heartbeat_peaks(
            audio_envelope,
            sample_rate,
            MIN_BPM,
            MAX_BPM,
            S1_S2_MAX_INTERVAL_SEC
        )
        print(f"Detected {len(peaks)} true heartbeats.")

        if len(peaks) < 2:
            print("Not enough peaks detected to calculate BPM.")
            return

        # 3. Calculate and smooth BPM
        smoothed_bpm, bpm_times = calculate_bpm_series(peaks, sample_rate, SMOOTHING_WINDOW_SEC)

        # 4. Save results to CSV
        output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
        save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)
        print(f"BPM data saved to {output_csv_path}")

        # 5. Plot the results
        plot_results(audio_envelope, peaks, smoothed_bpm, bpm_times, sample_rate, wav_file_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
