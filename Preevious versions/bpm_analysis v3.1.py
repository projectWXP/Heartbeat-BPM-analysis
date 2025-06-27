import os
import warnings
import csv
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
try:
    from pydub import AudioSegment
except ImportError:
    print("Pydub library not found. Please install it with 'pip install pydub'")
    print("You also need FFmpeg installed and in your system's PATH for audio conversion.")
    AudioSegment = None

def convert_to_wav(file_path, target_path):
    """
    Converts any audio file supported by FFmpeg to a mono WAV file.
    """
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")

    print(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        sound = AudioSegment.from_file(file_path)
        sound = sound.set_channels(1) # Convert to mono
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        print(f"Could not convert file {file_path}. Error: {e}")
        return False


def preprocess_audio(file_path, downsample_factor=10, bandpass_freqs=(20, 150), save_debug_file=False):
    """
    Loads, filters, and preprocesses the audio file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # --- 1. Band-pass filter to isolate heart sounds ---
    lowcut, highcut = bandpass_freqs
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_data)

    if save_debug_file:
        debug_path = f"{os.path.splitext(file_path)[0]}_filtered_debug.wav"
        normalized_audio = np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767)
        wavfile.write(debug_path, sample_rate, normalized_audio)
        print(f"Saved filtered debug audio to {debug_path}")

    # --- 2. Downsample for performance ---
    new_sample_rate = sample_rate // downsample_factor
    audio_downsampled = audio_filtered[::downsample_factor]

    # --- 3. Calculate the envelope of the signal ---
    audio_abs = np.abs(audio_downsampled)
    window_size = new_sample_rate // 10
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values

    return audio_envelope, new_sample_rate

def find_heartbeat_peaks(audio_envelope, sample_rate, min_bpm=40, max_bpm=220):
    """
    Finds heartbeats by dynamically identifying S1-S2 patterns.
    Returns the final filtered peaks and the initial raw peaks for debugging.
    """
    min_peak_distance_samples = int((60.0 / 240.0) * sample_rate)
    prominence_threshold = np.quantile(audio_envelope, 0.6)
    height_threshold = np.mean(audio_envelope) * 0.6

    all_peaks, _ = find_peaks(
        audio_envelope,
        distance=min_peak_distance_samples,
        prominence=prominence_threshold,
        height=height_threshold
    )

    if len(all_peaks) < 3:
        return all_peaks, all_peaks

    peak_times_sec = all_peaks / sample_rate
    intervals_sec = np.diff(peak_times_sec)

    # --- NEW: Dynamic S1-S2 interval thresholding ---
    if len(intervals_sec) > 5:
        median_interval = np.median(intervals_sec)
        # Assume systolic interval is less than 45% of the total beat cycle.
        dynamic_threshold = median_interval * 0.45
        # Clamp the threshold to a physiologically plausible range.
        s1_s2_max_interval_sec = np.clip(dynamic_threshold, 0.15, 0.4)
    else:
        # Fallback for short signals
        s1_s2_max_interval_sec = 0.35

    true_beat_indices = []
    i = 0
    while i < len(intervals_sec):
        if intervals_sec[i] <= s1_s2_max_interval_sec:
            peak1_idx, peak2_idx = all_peaks[i], all_peaks[i + 1]
            true_beat_indices.append(peak1_idx if audio_envelope[peak1_idx] >= audio_envelope[peak2_idx] else peak2_idx)
            i += 2
        else:
            true_beat_indices.append(all_peaks[i])
            i += 1

    if i == len(intervals_sec):
        true_beat_indices.append(all_peaks[-1])
    true_beat_indices = np.array(list(dict.fromkeys(true_beat_indices)))

    if len(true_beat_indices) > 1:
        final_peak_times = true_beat_indices / sample_rate
        final_intervals = np.diff(final_peak_times)
        min_interval, max_interval = 60.0 / max_bpm, 60.0 / min_bpm
        filtered_peaks = [true_beat_indices[0]]
        for j in range(len(final_intervals)):
            if min_interval <= final_intervals[j] <= max_interval:
                filtered_peaks.append(true_beat_indices[j + 1])
        return np.array(filtered_peaks), all_peaks
    return true_beat_indices, all_peaks


def calculate_bpm_series(peaks, sample_rate, smoothing_window_sec=5):
    """
    Calculates the BPM over time from the detected peaks and smooths it.
    """
    if len(peaks) < 2:
        return pd.Series(dtype=np.float64), np.array([])

    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    instant_bpm = 60.0 / time_diffs
    bpm_series = pd.Series(instant_bpm, index=peak_times[1:])

    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        beats_in_window = int(np.ceil((smoothing_window_sec / 60) * avg_heart_rate))
        beats_in_window = max(2, beats_in_window)
        smoothed_bpm = bpm_series.rolling(window=beats_in_window, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)

    time_points = (peak_times[:-1] + peak_times[1:]) / 2
    return smoothed_bpm, time_points

def plot_results(audio_envelope, peaks, all_raw_peaks, smoothed_bpm, bpm_times, sample_rate, file_name):
    """
    Creates an interactive plot of the results, including a hidden raw peaks trace for debugging.
    """
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")), secondary_y=False)

    # --- NEW: Add hidden trace for all raw peaks for debugging ---
    if len(all_raw_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=all_raw_peaks / sample_rate,
                y=audio_envelope[all_raw_peaks],
                mode='markers',
                name='All Detected Peaks (Raw)',
                marker=dict(color='grey', symbol='x', size=6),
                visible='legendonly' # This hides the trace by default
            ),
            secondary_y=False
        )

    fig.add_trace(go.Scatter(x=peaks / sample_rate, y=audio_envelope[peaks], mode='markers', name='Detected Heartbeats', marker=dict(color='#e36f6f', size=8)), secondary_y=False)

    if not smoothed_bpm.empty:
        fig.add_trace(go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM", line=dict(color="#4a4a4a", width=3)), secondary_y=True)

    if not smoothed_bpm.empty:
        max_bpm_val, min_bpm_val, avg_bpm_val = smoothed_bpm.max(), smoothed_bpm.min(), smoothed_bpm.mean()
        max_bpm_time, min_bpm_time = smoothed_bpm.idxmax(), smoothed_bpm.idxmin()
        fig.add_annotation(x=max_bpm_time, y=max_bpm_val, text=f"Max BPM: {max_bpm_val:.1f}", showarrow=True, arrowhead=1, ax=20, ay=-40, font=dict(color="#e36f6f"), yref="y2")
        fig.add_annotation(x=min_bpm_time, y=min_bpm_val, text=f"Min BPM: {min_bpm_val:.1f}", showarrow=True, arrowhead=1, ax=20, ay=40, font=dict(color="#a3d194"), yref="y2")
        fig.add_annotation(x=bpm_times[-1] if bpm_times.size > 0 else 0, y=avg_bpm_val, text=f"Avg BPM: {avg_bpm_val:.1f}", showarrow=False, xanchor="right", yanchor="top", font=dict(color="#4a4a4a"), yref="y2")

    max_time_sec = time_axis[-1] if len(time_axis) > 0 else 1
    tick_interval_sec = max(30, np.ceil(max_time_sec / 10))
    tick_vals_sec = np.arange(0, max_time_sec, tick_interval_sec)
    tick_text_min = [f"{s / 60:.1f}" for s in tick_vals_sec]

    fig.update_layout(
        title_text=f"Heartbeat Analysis - {os.path.basename(file_name)}",
        xaxis_title="Time (seconds)",
        dragmode='pan',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(title="Time (minutes)", overlaying='x', side='top', tickvals=tick_vals_sec, ticktext=tick_text_min)
    )

    max_amplitude = np.max(audio_envelope) if len(audio_envelope) > 0 else 1000
    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[-0.1 * max_amplitude, 6.0 * max_amplitude])
    fig.update_yaxes(title_text="Beats Per Minute (BPM)", secondary_y=True, range=[max(0, smoothed_bpm.min(skipna=True)-10) if not smoothed_bpm.empty else 0, smoothed_bpm.max(skipna=True)+10 if not smoothed_bpm.empty else 260])

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

def analyze_wav_file(wav_file_path, params):
    """Runs the full analysis pipeline on a single WAV file."""
    file_name_no_ext = os.path.splitext(wav_file_path)[0]
    print(f"Processing file: {os.path.basename(wav_file_path)}...")

    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params['downsample_factor'], params['bandpass_freqs'], params['save_debug_wav'])

    # Updated to receive two values from find_heartbeat_peaks
    peaks, all_raw_peaks = find_heartbeat_peaks(audio_envelope, sample_rate, params['min_bpm'], params['max_bpm'])
    print(f"Detected {len(peaks)} true heartbeats from {len(all_raw_peaks)} raw peaks.")

    if len(peaks) < 2:
        print("Not enough peaks detected to calculate BPM.")
        # Still plot the raw data even if BPM calculation fails
        plot_results(audio_envelope, peaks, all_raw_peaks, pd.Series(dtype=np.float64), np.array([]), sample_rate, wav_file_path)
        return

    smoothed_bpm, bpm_times = calculate_bpm_series(peaks, sample_rate, params['smoothing_window_sec'])

    output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
    save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)
    print(f"BPM data saved to {output_csv_path}")

    # Pass all_raw_peaks to the plotting function
    plot_results(audio_envelope, peaks, all_raw_peaks, smoothed_bpm, bpm_times, sample_rate, wav_file_path)

def main():
    """Main function to find and process all audio files in the directory."""
    params = {
        "downsample_factor": 100,
        "bandpass_freqs": (20, 150),
        "min_bpm": 40,
        "max_bpm": 220,
        "smoothing_window_sec": 5,
        "save_debug_wav": True
    }

    supported_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4']
    audio_files = [f for f in os.listdir(".") if os.path.splitext(f)[1].lower() in supported_extensions]

    if not audio_files:
        print("No supported audio files found in the current directory.")
        return

    converted_dir = "converted_wavs"
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)

    for audio_file in audio_files:
        try:
            base_name, ext = os.path.splitext(audio_file)
            wav_path = os.path.join(converted_dir, f"{base_name}.wav")

            if ext.lower() != '.wav':
                if not convert_to_wav(audio_file, wav_path):
                    continue
            else:
                import shutil
                shutil.copy(audio_file, wav_path)

            analyze_wav_file(wav_path, params)
            print("-" * 50)

        except Exception as e:
            print(f"\nAn unexpected error occurred while processing {audio_file}: {e}")
            traceback.print_exc()
            print("-" * 50)

if __name__ == "__main__":
    main()
