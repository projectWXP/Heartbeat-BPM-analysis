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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from itertools import groupby
from operator import itemgetter
import datetime

# --- Reminder to future maintainers, do not remove the debug code from this WIP version of the project ---
# --- Audio Conversion ---
try:
    from pydub import AudioSegment
except ImportError:
    print("Pydub library not found. Please install it with 'pip install pydub'")
    print("You also need FFmpeg installed and in your system's PATH for audio conversion.")
    AudioSegment = None

def convert_to_wav(file_path, target_path):
    if not AudioSegment: raise ImportError("Pydub/FFmpeg is required for audio conversion.")
    print(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        sound = AudioSegment.from_file(file_path)
        sound = sound.set_channels(1)
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        print(f"Could not convert file {file_path}. Error: {e}")
        return False

# --- Preprocessing ---
def preprocess_audio(file_path, downsample_factor=50, bandpass_freqs=(20, 150), save_debug_file=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)
    if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
    lowcut, highcut = bandpass_freqs
    nyquist = 0.5 * sample_rate
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_data)
    if save_debug_file:
        debug_path = f"{os.path.splitext(file_path)[0]}_filtered_debug.wav"
        normalized_audio = np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767)
        wavfile.write(debug_path, sample_rate, normalized_audio)
    new_sample_rate = sample_rate // downsample_factor
    audio_downsampled = audio_filtered[::downsample_factor]
    audio_abs = np.abs(audio_downsampled)
    window_size = new_sample_rate // 10
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values
    return audio_envelope, new_sample_rate

# --- Core Beat Detection Logic v6.5 ---

def calculate_blended_confidence(deviation, bpm):
    """v6.0: Replaces the piecewise function with a continuous, blended model."""
    conf_at_rest = np.interp(deviation, [0.0, 0.3, 0.6], [0.9, 0.8, 0.1])
    conf_at_exercise = np.interp(deviation, [0.1, 0.4, 0.7], [0.2, 0.9, 0.8])
    conf_at_exertion = np.interp(deviation, [0.0, 0.2, 0.5], [0.05, 0.1, 0.7])
    final_confidence = np.interp(bpm, [80, 130, 170], [conf_at_rest, conf_at_exercise, conf_at_exertion])
    return final_confidence

def find_heartbeat_peaks(audio_envelope, sample_rate, min_bpm=40, max_bpm=240, start_bpm_hint=None, verbose_debug=False):
    """
    v6.5: Adds a "Lookahead Veto" to prevent strong S1 peaks from being misclassified as S2.
    """
    min_peak_distance_samples = int(0.05 * sample_rate)

    # --- UPDATED: Robust Trough Detection ---
    trough_prominence_threshold = np.quantile(audio_envelope, 0.05)
    trough_indices, _ = find_peaks(
        -audio_envelope,
        distance=min_peak_distance_samples,
        prominence=trough_prominence_threshold
        # REMOVED the 'height' parameter to make detection less strict
    )
    noise_floor = np.nan
    if len(trough_indices) > 0:
        trough_amplitudes = audio_envelope[trough_indices]
        noise_floor = np.quantile(trough_amplitudes, 0.25)
        print(f"DEBUG: Identified {len(trough_indices)} robust troughs. Calculated Noise Floor: {noise_floor:.2f}")
    else:
        print("DEBUG: No significant troughs found to calculate noise floor.")
    # --- END UPDATED ---

    # --- Peak Detection ---
    prominence_threshold = np.quantile(audio_envelope, 0.1)
    height_threshold = np.mean(audio_envelope) * 0.1
    all_peaks, _ = find_peaks(
        audio_envelope,
        distance=min_peak_distance_samples,
        prominence=prominence_threshold,
        height=height_threshold
    )

    if verbose_debug: print(f"\n--- VERBOSE DEBUG: find_heartbeat_peaks (v6.5) ---")
    print(f"DEBUG: Found {len(all_peaks)} raw peaks.")
    if len(all_peaks) < 2:
        analysis_data = {
            "beat_debug_info": {}, "deviation_times": np.array([]), "deviation_series": np.array([]),
            "long_term_bpm_series": pd.Series(dtype=np.float64), 'trough_indices': trough_indices, 'noise_floor': noise_floor
        }
        return all_peaks, all_peaks, analysis_data

    peak_amplitudes = audio_envelope[all_peaks]
    normalized_deviations = np.abs(np.diff(peak_amplitudes)) / (np.maximum(peak_amplitudes[:-1], peak_amplitudes[1:]) + 1e-9)
    deviation_times = (all_peaks[:-1] + all_peaks[1:]) / 2 / sample_rate
    smoothing_window_peaks = max(5, int(len(normalized_deviations) * 0.05))
    smoothed_dev_series = pd.Series(normalized_deviations).rolling(window=smoothing_window_peaks, min_periods=1, center=True).mean().values

    long_term_bpm = float(start_bpm_hint) if start_bpm_hint else 80.0
    print(f"DEBUG: Initializing Long-Term BPM to: {long_term_bpm:.1f} BPM")

    candidate_beats = []
    beat_debug_info = {}
    long_term_bpm_history = []

    print("DEBUG: Step 3 (Grouping) - Applying pattern matching and confidence-weighted updates with Lookahead Veto.")
    i = 0
    while i < len(all_peaks) - 1:
        expected_rr_interval = 60.0 / long_term_bpm
        s1_s2_max_interval_sec = min(0.4, expected_rr_interval * 0.6)

        current_peak_idx = all_peaks[i]
        next_peak_idx = all_peaks[i + 1]

        interval_sec = (next_peak_idx - current_peak_idx) / sample_rate
        smoothed_deviation = smoothed_dev_series[i]
        confidence = calculate_blended_confidence(smoothed_deviation, long_term_bpm)

        reason = f"Base Conf: {confidence:.2f} (Smoothed Dev: {smoothed_deviation:.2f}, LT-BPM: {long_term_bpm:.0f})"

        bpm_if_not_paired = 60.0 / interval_sec if interval_sec > 0 else 999
        if bpm_if_not_paired > long_term_bpm * 1.7 and long_term_bpm < 150:
            confidence = min(0.95, confidence + 0.3)
            reason += f" | BOOSTED to {confidence:.2f} (BPM spike: {bpm_if_not_paired:.0f}>>{long_term_bpm:.0f})"

        pattern_match_override = False
        local_raw_deviation = normalized_deviations[i]
        if local_raw_deviation > 0.25 and i > 0 and i < len(all_peaks) - 2:
            is_current_peak_H = peak_amplitudes[i] > peak_amplitudes[i-1] and peak_amplitudes[i] > peak_amplitudes[i+1]
            is_next_peak_L = peak_amplitudes[i+1] < peak_amplitudes[i] and peak_amplitudes[i+1] < peak_amplitudes[i+2]
            if is_current_peak_H and is_next_peak_L:
                pattern_match_override = True
                reason += f" | OVERRIDE (H-L Pattern, Local Dev: {local_raw_deviation:.2f})"

        lookahead_veto = False
        if i < len(all_peaks) - 2:
            is_next_peak_a_strong_H = peak_amplitudes[i+1] > peak_amplitudes[i] and peak_amplitudes[i+1] > peak_amplitudes[i+2]
            is_next_peak_much_larger = peak_amplitudes[i+1] > peak_amplitudes[i] * 1.5
            if not pattern_match_override and is_next_peak_a_strong_H and is_next_peak_much_larger:
                lookahead_veto = True

        is_paired = False
        if not lookahead_veto and ((interval_sec <= s1_s2_max_interval_sec and confidence > 0.6) or pattern_match_override):
            is_paired = True

        if is_paired:
            s1_idx = current_peak_idx
            candidate_beats.append(s1_idx)
            beat_debug_info[s1_idx] = f"S1 (Paired). {reason}"
            beat_debug_info[next_peak_idx] = f"S2 of {s1_idx/sample_rate:.2f}s"

            if len(candidate_beats) > 1:
                prev_s1_idx = candidate_beats[-2]
                new_rr_interval = (s1_idx - prev_s1_idx) / sample_rate
                if new_rr_interval > 0:
                    instant_bpm = 60.0 / new_rr_interval
                    if 0.5 * long_term_bpm < instant_bpm < 2.0 * long_term_bpm:
                        target_bpm = (0.95 * long_term_bpm) + (0.05 * instant_bpm)
                        max_bpm_change_per_second = 3.0
                        max_change_for_this_interval = max_bpm_change_per_second * new_rr_interval
                        proposed_change = target_bpm - long_term_bpm
                        if abs(proposed_change) > max_change_for_this_interval:
                            limited_change = np.sign(proposed_change) * max_change_for_this_interval
                            new_long_term_bpm = long_term_bpm + limited_change
                        else: new_long_term_bpm = target_bpm
                        long_term_bpm = max(min_bpm, min(new_long_term_bpm, max_bpm))
            i += 2
        else:
            s1_idx = current_peak_idx
            if lookahead_veto:
                reason += " | SKIPPED PAIRING (Vetoed by lookahead)"

            candidate_beats.append(s1_idx)
            beat_debug_info[s1_idx] = f"Lone S1. {reason}"
            i += 1

        if candidate_beats:
            time_of_beat = candidate_beats[-1] / sample_rate
            long_term_bpm_history.append((time_of_beat, long_term_bpm))

    if i == len(all_peaks) - 1:
        last_peak_idx = all_peaks[-1]
        candidate_beats.append(last_peak_idx)
        beat_debug_info[last_peak_idx] = "Lone S1 (Last Peak)"

    final_peaks = np.array(sorted(list(dict.fromkeys(candidate_beats))))
    print(f"DEBUG: Final peak count after stateful grouping: {len(final_peaks)}.")
    analysis_data = {"beat_debug_info": beat_debug_info, "deviation_times": deviation_times, "deviation_series": smoothed_dev_series}

    analysis_data['trough_indices'] = trough_indices
    analysis_data['noise_floor'] = noise_floor

    if long_term_bpm_history:
        lt_bpm_times, lt_bpm_values = zip(*long_term_bpm_history)
        analysis_data["long_term_bpm_series"] = pd.Series(lt_bpm_values, index=lt_bpm_times)
    else: analysis_data["long_term_bpm_series"] = pd.Series(dtype=np.float64)
    return final_peaks, all_peaks, analysis_data

def calculate_bpm_series(peaks, sample_rate, smoothing_window_sec=5):
    if len(peaks) < 2: return pd.Series(dtype=np.float64), np.array([])
    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    valid_diffs = time_diffs > 1e-6
    if not np.any(valid_diffs): return pd.Series(dtype=np.float64), np.array([])
    instant_bpm = 60.0 / time_diffs[valid_diffs]
    valid_peak_times = peak_times[1:][valid_diffs]
    bpm_series = pd.Series(instant_bpm, index=valid_peak_times)
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        beats_in_window = max(2, int(np.ceil((smoothing_window_sec / 60) * avg_heart_rate)))
        smoothed_bpm = bpm_series.rolling(window=beats_in_window, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)
    return smoothed_bpm, bpm_series.index.values

def plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, file_name, min_bpm, max_bpm):
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")), secondary_y=False)

    if 'trough_indices' in analysis_data and analysis_data['trough_indices'].size > 0:
        trough_indices = analysis_data['trough_indices']
        fig.add_trace(go.Scatter(
            x=trough_indices / sample_rate,
            y=audio_envelope[trough_indices],
            mode='markers',
            name='Troughs (Local Minima)',
            marker=dict(color='green', symbol='circle-open', size=6),
            visible='legendonly'
        ), secondary_y=False)

    if 'noise_floor' in analysis_data and not np.isnan(analysis_data['noise_floor']):
        noise_floor = analysis_data['noise_floor']
        fig.add_hline(
            y=noise_floor,
            line_dash="dot",
            annotation_text=f"Noise Floor ({noise_floor:.2f})",
            annotation_position="bottom right",
            line_color="green",
            secondary_y=False
        )

    if len(all_raw_peaks) > 0:
        raw_peak_reasons = [analysis_data.get('beat_debug_info', {}).get(p, 'N/A').replace(' | ', '<br>- ') for p in all_raw_peaks]
        raw_peak_customdata = np.stack((all_raw_peaks / sample_rate, audio_envelope[all_raw_peaks], raw_peak_reasons), axis=-1)
        fig.add_trace(go.Scatter(x=all_raw_peaks / sample_rate, y=audio_envelope[all_raw_peaks], mode='markers', name='All Peaks (Hover for Status)', marker=dict(color='grey', symbol='x', size=6), customdata=raw_peak_customdata, hovertemplate="<b>Raw Peak</b><br>Time: %{customdata[0]:.2f}s<br>Amp: %{customdata[1]:.0f}<br><b>Status:</b><br>- %{customdata[2]}<extra></extra>", visible='legendonly'), secondary_y=False)

    if len(peaks) > 0:
        final_peak_reasons = [analysis_data.get('beat_debug_info', {}).get(p, 'N/A').replace(' | ', '<br>- ') for p in peaks]
        final_peak_customdata = np.stack((peaks / sample_rate, audio_envelope[peaks], final_peak_reasons), axis=-1)
        fig.add_trace(go.Scatter(x=peaks / sample_rate, y=audio_envelope[peaks], mode='markers', name='Final Heartbeats (S1)', marker=dict(color='#e36f6f', size=8, symbol='diamond'), customdata=final_peak_customdata, hovertemplate="<b>Final Beat</b><br>Time: %{customdata[0]:.2f}s<br><b>Reason:</b><br>- %{customdata[2]}<extra></extra>"), secondary_y=False)

    if not smoothed_bpm.empty:
        fig.add_trace(go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM", line=dict(color="#4a4a4a", width=3, dash='dash')), secondary_y=True)

    if "long_term_bpm_series" in analysis_data and not analysis_data["long_term_bpm_series"].empty:
        lt_series = analysis_data["long_term_bpm_series"]
        fig.add_trace(go.Scatter(x=lt_series.index, y=lt_series.values, name="Long-Term BPM (Belief)", line=dict(color='orange', width=2, dash='dot')), secondary_y=True)

    if 'deviation_series' in analysis_data and analysis_data['deviation_series'] is not None:
        deviation_percent = analysis_data['deviation_series'] * 100
        fig.add_trace(go.Scatter(x=analysis_data['deviation_times'], y=deviation_percent, name='Norm. Deviation % (Smoothed)', line=dict(color='purple', width=2), visible='legendonly', hovertemplate='Norm. Deviation: %{y:.2f}%<extra></extra>'), secondary_y=True)

    if not smoothed_bpm.empty:
        max_bpm_val, min_bpm_val = smoothed_bpm.max(), smoothed_bpm.min()
        max_bpm_time, min_bpm_time = smoothed_bpm.idxmax(), smoothed_bpm.idxmin()
        fig.add_annotation(x=max_bpm_time, y=max_bpm_val, text=f"Max: {max_bpm_val:.1f} BPM", showarrow=True, arrowhead=1, ax=20, ay=-40, font=dict(color="#e36f6f"), yref="y2")
        fig.add_annotation(x=min_bpm_time, y=min_bpm_val, text=f"Min: {min_bpm_val:.1f} BPM", showarrow=True, arrowhead=1, ax=20, ay=40, font=dict(color="#a3d194"), yref="y2")

    max_time_sec = time_axis[-1] if len(time_axis) > 0 else 1
    tick_vals_sec = np.arange(0, max_time_sec + 30, 30)
    tick_text_combined = [f"{int(s)}s ({int(s // 60):02d}:{int(s % 60):02d})" for s in tick_vals_sec]

    fig.update_layout(title_text=f"Heartbeat Analysis - {os.path.basename(file_name)} (v6.5)", dragmode='pan', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=100, b=100), xaxis=dict(title_text="Time", tickvals=tick_vals_sec, ticktext=tick_text_combined), hovermode='x unified')

    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[0, audio_envelope.max() * 15]) # reminder not to change this value
    fig.update_yaxes(title_text="BPM / Norm. Dev %", secondary_y=True, range=[min_bpm-10, max_bpm+10])

    output_html_path = f"{os.path.splitext(file_name)[0]}_bpm_plot.html"
    fig.write_html(output_html_path, config={'scrollZoom': True})
    print(f"Interactive plot saved to {output_html_path}")

def save_bpm_to_csv(bpm_series, time_points, output_path):
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time (s)", "BPM"])
        if not bpm_series.empty:
            for t, bpm in zip(time_points, bpm_series):
                if not np.isnan(bpm): writer.writerow([f"{t:.2f}", f"{bpm:.1f}"])

def print_and_log_chronological_data(audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm, output_log_path, file_name):
    events = []
    time_axis = np.arange(len(audio_envelope)) / sample_rate

    for p in all_raw_peaks:
        t = p / sample_rate
        reason = analysis_data.get('beat_debug_info', {}).get(p, 'N/A')
        reason_formatted = reason.replace(' | ', '\n    ')
        text = f"Raw Peak (Amp: {audio_envelope[p]:.2f})\n  Status: {reason_formatted}"
        events.append({'time': t, 'text': text})

    if not smoothed_bpm.empty:
        for t, bpm in smoothed_bpm.items():
            events.append({'time': t, 'text': f"Smoothed BPM: {bpm:.2f}"})

    lt_bpm_series = analysis_data.get("long_term_bpm_series")
    if lt_bpm_series is not None and not lt_bpm_series.empty:
        for t, bpm in lt_bpm_series.items():
            events.append({'time': t, 'text': f"Long-Term BPM (Belief): {bpm:.2f}"})

    dev_series = analysis_data.get('deviation_series')
    dev_times = analysis_data.get('deviation_times')
    if dev_series is not None and dev_times is not None:
        for i, t in enumerate(dev_times):
            events.append({'time': t, 'text': f"Norm. Deviation (Smoothed): {dev_series[i] * 100:.2f}%"})

    if 'trough_indices' in analysis_data:
        for p in analysis_data['trough_indices']:
            t = p / sample_rate
            text = f"Trough Detected (Amp: {audio_envelope[p]:.2f})"
            events.append({'time': t, 'text': text})

    events.sort(key=itemgetter('time'))

    print("\n\n" + "="*80)
    print(" Chronological Event Log")
    print("="*80 + "\n")

    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# Chronological Debug Log for {os.path.basename(file_name)}\n")
        log_file.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if 'noise_floor' in analysis_data and not np.isnan(analysis_data['noise_floor']):
            log_file.write(f"\n**Calculated Noise Floor (25th percentile of troughs): `{analysis_data['noise_floor']:.2f}`**\n\n")

        for time_val, group in groupby(events, key=itemgetter('time')):
            envelope_idx = np.abs(time_axis - time_val).argmin()
            envelope_val = audio_envelope[envelope_idx]

            print(f"Time: {time_val:.4f}s")
            print(f"  Audio Envelope: {envelope_val:.2f}")
            log_file.write(f"## Time: `{time_val:.4f}s`\n\n")
            log_file.write(f"* **Audio Envelope**: `{envelope_val:.2f}`\n")

            for event in group:
                print(f"  - {event['text']}")
                lines = event['text'].split('\n')
                first_line = lines[0]
                rest_of_lines = lines[1:]
                log_file.write(f"* **{first_line}**\n")
                for sub_line in rest_of_lines:
                    log_file.write(f"    * {sub_line.strip()}\n")

            print("-" * 20)
            log_file.write("\n---\n")

    print("\n" + "="*80)
    print(f"End of Log. Detailed output also saved to '{output_log_path}'")
    print("="*80 + "\n")


def analyze_wav_file(wav_file_path, params, start_bpm_hint):
    file_name_no_ext = os.path.splitext(wav_file_path)[0]
    print(f"\n--- Processing file: {os.path.basename(wav_file_path)} (Engine v6.5) ---")
    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params['downsample_factor'], params['bandpass_freqs'], params['save_debug_wav'])

    peaks, all_raw_peaks, analysis_data = find_heartbeat_peaks(
        audio_envelope, sample_rate, params['min_bpm'], params['max_bpm'],
        start_bpm_hint=start_bpm_hint, verbose_debug=True
    )

    if len(peaks) < 2:
        print("Not enough peaks detected to calculate BPM.")
        plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, pd.Series(dtype=np.float64), np.array([]), sample_rate, wav_file_path, params['min_bpm'], params['max_bpm'])
        return

    smoothed_bpm, bpm_times = calculate_bpm_series(peaks, sample_rate, params['smoothing_window_sec'])

    output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
    save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)
    print(f"BPM data saved to {output_csv_path}")

    plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, wav_file_path, params['min_bpm'], params['max_bpm'])

    output_log_path = f"{file_name_no_ext}_Debug_Log.md"
    print_and_log_chronological_data(audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm, output_log_path, wav_file_path)


# --- GUI Class ---
class BPMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer v6.5")
        self.root.geometry("500x300")
        self.style = ttkb.Style(theme='minty')
        self.current_file = None
        self.params = { "downsample_factor": 100, "bandpass_freqs": (20, 150), "min_bpm": 40, "max_bpm": 240, "smoothing_window_sec": 5, "save_debug_wav": True }
        self.create_widgets()
        self._find_initial_audio_file()
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20"); main_frame.pack(fill=tk.BOTH, expand=True)
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding=10); file_frame.pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=400); self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.select_file, bootstyle=INFO); browse_btn.pack(side=tk.RIGHT, padx=5)
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10); param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="Starting BPM (optional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bpm_entry = ttk.Entry(param_frame); self.bpm_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        btn_frame = ttk.Frame(main_frame); btn_frame.pack(fill=tk.X, pady=10)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.analyze, bootstyle=SUCCESS, state=tk.DISABLED); self.analyze_btn.pack(side=tk.RIGHT, padx=5)
        self.status_var = tk.StringVar(); self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN); self.status_bar.pack(fill=tk.X, pady=(10, 0))
        param_frame.columnconfigure(1, weight=1)
    def select_file(self):
        filetypes = [('Audio files', '*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(title="Select audio file", filetypes=filetypes)
        if filename:
            self.current_file = filename; self.file_label.config(text=os.path.basename(filename)); self.analyze_btn.config(state=tk.NORMAL); self.status_var.set("Ready to analyze")
    def _find_initial_audio_file(self):
        supported_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4')
        try:
            for filename in os.listdir(os.getcwd()):
                if filename.lower().endswith(supported_extensions):
                    self.current_file = os.path.join(os.getcwd(), filename)
                    self.file_label.config(text=os.path.basename(self.current_file)); self.analyze_btn.config(state=tk.NORMAL); self.status_var.set(f"Auto-loaded: {os.path.basename(self.current_file)}")
                    return
        except Exception as e: print(f"Could not auto-find file: {e}")
        self.file_label.config(text="No file selected"); self.analyze_btn.config(state=tk.DISABLED); self.status_var.set("No audio file found. Please browse.")
    def analyze(self):
        if not self.current_file: messagebox.showerror("Error", "No file selected"); return
        try:
            self.status_var.set("Analyzing..."); self.root.update()
            bpm_input = self.bpm_entry.get().strip()
            start_bpm_hint = float(bpm_input) if bpm_input else None
            converted_dir = "converted_wavs"; os.makedirs(converted_dir, exist_ok=True)
            base_name, ext = os.path.splitext(self.current_file)
            wav_path = os.path.join(converted_dir, f"{os.path.basename(base_name)}.wav")
            if ext.lower() != '.wav':
                if not AudioSegment: messagebox.showerror("Error", "Pydub/FFmpeg required for non-WAV files."); self.status_var.set("Conversion failed"); return
                if not convert_to_wav(self.current_file, wav_path): self.status_var.set("Conversion failed"); return
            else:
                import shutil; shutil.copy(self.current_file, wav_path)
            analyze_wav_file(wav_path, self.params, start_bpm_hint)
            self.status_var.set("Analysis complete!")
        except Exception as e:
            self.status_var.set("Error during analysis"); messagebox.showerror("Error", f"An error occurred:\n{str(e)}"); traceback.print_exc()

def main():
    root = ttkb.Window(themename="minty")
    app = BPMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()