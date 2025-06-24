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

def find_heartbeat_peaks(audio_envelope, sample_rate, min_bpm=40, max_bpm=220, start_bpm_hint=None, verbose_debug=False):
    """
    Finds heartbeats using only the dynamic pairing logic (Steps 4 & 5 removed).
    """
    # --- Step 1: Broad Peak Detection ---
    min_peak_distance_samples = int(0.05 * sample_rate)
    prominence_threshold = np.quantile(audio_envelope, 0.1)
    height_threshold = np.mean(audio_envelope) * 0.1
    all_peaks, _ = find_peaks(audio_envelope, distance=min_peak_distance_samples, prominence=prominence_threshold, height=height_threshold)

    if verbose_debug: print(f"\n--- VERBOSE DEBUG: find_heartbeat_peaks (v4.5) ---")
    print(f"DEBUG: Found {len(all_peaks)} raw peaks in the broad detection phase.")
    if len(all_peaks) < 2: return all_peaks, all_peaks, {}

    # --- Step 2: Preliminary Analysis ---
    print("DEBUG: Step 2 (Analysis) - Calculating NORMALIZED Deviation and Confidence.")
    peak_amplitudes = audio_envelope[all_peaks]
    peak_diffs = np.abs(np.diff(peak_amplitudes))
    max_adjacent_amps = np.maximum(peak_amplitudes[:-1], peak_amplitudes[1:])
    normalized_deviations = peak_diffs / (max_adjacent_amps + 1e-9)
    deviation_times = (all_peaks[:-1] + all_peaks[1:]) / 2 / sample_rate
    smoothing_window_peaks = max(5, int(len(normalized_deviations) * 0.05))
    smoothed_dev_series = pd.Series(normalized_deviations).rolling(window=smoothing_window_peaks, min_periods=1, center=True).mean().values

    def calculate_confidence(dev):
        if dev < 0.2: return 0.9
        if dev > 0.5: return 0.8
        return np.interp(dev, [0.2, 0.35, 0.5], [0.9, 0.1, 0.8])

    confidence_scores = np.array([calculate_confidence(d) for d in smoothed_dev_series])

    # --- Step 3: Grouping with DYNAMIC S1-S2 Interval ---
    print("DEBUG: Step 3 (Grouping) - Applying pairing logic with a DYNAMIC interval threshold.")
    candidate_beats = []
    beat_debug_info = {}

    last_s1_interval = (60.0 / start_bpm_hint) if start_bpm_hint else 60.0/80.0

    i = 0
    while i < len(all_peaks) - 1:
        current_peak_idx = all_peaks[i]
        next_peak_idx = all_peaks[i + 1]

        s1_s2_max_interval_sec = min(0.35, last_s1_interval * 0.5)
        interval_sec = (next_peak_idx - current_peak_idx) / sample_rate
        confidence = confidence_scores[i]

        if current_peak_idx not in beat_debug_info: beat_debug_info[current_peak_idx] = "Unprocessed"
        if next_peak_idx not in beat_debug_info: beat_debug_info[next_peak_idx] = "Unprocessed"

        decision_made = False
        if interval_sec <= s1_s2_max_interval_sec and confidence > 0.6:
            amp_current, amp_next = audio_envelope[current_peak_idx], audio_envelope[next_peak_idx]
            s1_idx, s2_idx = (current_peak_idx, next_peak_idx) if amp_current >= amp_next else (next_peak_idx, current_peak_idx)

            if candidate_beats:
                last_s1_interval = (s1_idx - candidate_beats[-1]) / sample_rate
            candidate_beats.append(s1_idx)
            beat_debug_info[s1_idx] = f"S1 (Kept). Paired with peak at {s2_idx/sample_rate:.2f}s"
            beat_debug_info[s2_idx] = f"S2 (Discarded). Paired with S1 at {s1_idx/sample_rate:.2f}s"
            if verbose_debug:
                print(f"  - PAIR FOUND @ {current_peak_idx/sample_rate:.2f}s: Labeled S1: {s1_idx}, S2: {s2_idx}. (Conf: {confidence:.2f})")
            i += 2
            decision_made = True

        if not decision_made:
            reason = f"Lone S1 (Interval {interval_sec:.3f}s > Dynamic Max {s1_s2_max_interval_sec:.3f}s)" if interval_sec > s1_s2_max_interval_sec else f"Lone S1 (Confidence {confidence:.2f} <= 0.6)"
            if candidate_beats:
                last_s1_interval = (current_peak_idx - candidate_beats[-1]) / sample_rate
            candidate_beats.append(current_peak_idx)
            beat_debug_info[current_peak_idx] = reason
            if verbose_debug: print(f"  - LONE PEAK @ {current_peak_idx/sample_rate:.2f}s: {reason}")
            i += 1

    if i == len(all_peaks) - 1:
        last_peak_idx = all_peaks[-1]
        candidate_beats.append(last_peak_idx)
        beat_debug_info[last_peak_idx] = "Lone S1 (Last Peak)"

    # --- STEPS 4 AND 5 HAVE BEEN REMOVED ---
    print("DEBUG: Steps 4 (Rescue) and 5 (Final Filter) were REMOVED for debugging.")

    final_peaks = np.array(sorted(list(dict.fromkeys(candidate_beats))))
    print(f"DEBUG: Final peak count after Step 3: {len(final_peaks)}.")
    analysis_data = {"deviation_times": deviation_times, "deviation_series": smoothed_dev_series, "confidence_scores": confidence_scores, "beat_debug_info": beat_debug_info}
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

def plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, file_name):
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")), secondary_y=False)

    if len(all_raw_peaks) > 0:
        raw_peak_reasons = [analysis_data.get('beat_debug_info', {}).get(p, 'N/A') for p in all_raw_peaks]
        raw_peak_customdata = np.stack((all_raw_peaks / sample_rate, audio_envelope[all_raw_peaks], raw_peak_reasons), axis=-1)
        fig.add_trace(go.Scatter(x=all_raw_peaks / sample_rate, y=audio_envelope[all_raw_peaks], mode='markers', name='All Peaks (Labeled)', marker=dict(color='grey', symbol='x', size=6), customdata=raw_peak_customdata, hovertemplate="<b>Raw Peak</b><br>Time: %{customdata[0]:.2f}s<br>Amp: %{customdata[1]:.0f}<br><b>Status:</b> %{customdata[2]}<extra></extra>", visible='legendonly'), secondary_y=False)

    if len(peaks) > 0:
        final_peak_reasons = [analysis_data.get('beat_debug_info', {}).get(p, 'N/A') for p in peaks]
        final_peak_customdata = np.stack((peaks / sample_rate, audio_envelope[peaks], final_peak_reasons), axis=-1)
        fig.add_trace(go.Scatter(x=peaks / sample_rate, y=audio_envelope[peaks], mode='markers', name='Final Heartbeats', marker=dict(color='#e36f6f', size=8, symbol='diamond'), customdata=final_peak_customdata, hovertemplate="<b>Final Beat</b><br>Time: %{customdata[0]:.2f}s<br>Amp: %{customdata[1]:.0f}<br><b>Reason:</b> %{customdata[2]}<extra></extra>"), secondary_y=False)

    if not smoothed_bpm.empty:
        fig.add_trace(go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM", line=dict(color="#4a4a4a", width=3, dash='dash')), secondary_y=True)

    if 'deviation_series' in analysis_data and analysis_data['deviation_series'] is not None:
        fig.add_trace(go.Scatter(x=analysis_data['deviation_times'], y=analysis_data['deviation_series'], name='Norm. Deviation', line=dict(color='purple', width=2), visible='legendonly'), secondary_y=True)

    if 'confidence_scores' in analysis_data and analysis_data['confidence_scores'] is not None:
        fig.add_trace(go.Scatter(x=analysis_data['deviation_times'], y=analysis_data['confidence_scores'], name='Pairing Confidence', line=dict(color='orange', width=2, dash='dot'), visible='legendonly'), secondary_y=True)

    if not smoothed_bpm.empty:
        max_bpm_val, min_bpm_val = smoothed_bpm.max(), smoothed_bpm.min()
        max_bpm_time, min_bpm_time = smoothed_bpm.idxmax(), smoothed_bpm.idxmin()
        fig.add_annotation(x=max_bpm_time, y=max_bpm_val, text=f"Max: {max_bpm_val:.1f} BPM", showarrow=True, arrowhead=1, ax=20, ay=-40, font=dict(color="#e36f6f"), yref="y2")
        fig.add_annotation(x=min_bpm_time, y=min_bpm_val, text=f"Min: {min_bpm_val:.1f} BPM", showarrow=True, arrowhead=1, ax=20, ay=40, font=dict(color="#a3d194"), yref="y2")

    max_time_sec = time_axis[-1] if len(time_axis) > 0 else 1
    tick_vals_sec = np.arange(0, max_time_sec + 30, 30)
    tick_text_combined = [f"{int(s)}s ({int(s // 60):02d}:{int(s % 60):02d})" for s in tick_vals_sec]

    fig.update_layout(title_text=f"Heartbeat Analysis - {os.path.basename(file_name)} (v4.5)", dragmode='pan', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=100, b=100), xaxis=dict(title_text="Time", tickvals=tick_vals_sec, ticktext=tick_text_combined), hovermode='x unified')
    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False)
    fig.update_yaxes(title_text="BPM / Normalized Score", secondary_y=True, range=[0, smoothed_bpm.max(skipna=True)+10 if not smoothed_bpm.empty else 260])
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

def analyze_wav_file(wav_file_path, params, start_bpm_hint):
    file_name_no_ext = os.path.splitext(wav_file_path)[0]
    print(f"\n--- Processing file: {os.path.basename(wav_file_path)} ---")
    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params['downsample_factor'], params['bandpass_freqs'], params['save_debug_wav'])

    peaks, all_raw_peaks, analysis_data = find_heartbeat_peaks(
        audio_envelope, sample_rate, params['min_bpm'], params['max_bpm'],
        start_bpm_hint=start_bpm_hint, verbose_debug=True
    )

    if len(peaks) < 2:
        print("Not enough peaks detected to calculate BPM.")
        plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, pd.Series(dtype=np.float64), np.array([]), sample_rate, wav_file_path)
        return

    smoothed_bpm, bpm_times = calculate_bpm_series(peaks, sample_rate, params['smoothing_window_sec'])
    output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
    save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)
    print(f"BPM data saved to {output_csv_path}")
    plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, wav_file_path)

# --- GUI Class ---
class BPMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer v4.5")
        self.root.geometry("500x300")
        self.style = ttkb.Style(theme='minty')
        self.current_file = None
        self.params = { "downsample_factor": 100, "bandpass_freqs": (20, 150), "min_bpm": 40, "max_bpm": 220, "smoothing_window_sec": 5, "save_debug_wav": True }
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
