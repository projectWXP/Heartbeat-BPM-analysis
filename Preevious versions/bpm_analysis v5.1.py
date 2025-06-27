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
from operator import itemgetter
import datetime
import logging
import threading
import sys
import queue

# --- Do not remove the debug code from this project! ---
# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    stream=sys.stdout
)

# --- Audio Conversion (requires pydub/ffmpeg) ---
try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Please install with 'pip install pydub'")
    logging.warning("FFmpeg is also required for audio conversion (add to system PATH).")
    AudioSegment = None

# --- Centralized Configuration for Easy Tuning (don't remove the comments)---
DEFAULT_PARAMS = {
    # Preprocessing Parameters
    "downsample_factor": 100,     # The factor by which to reduce the audio's sample rate. higher = less detail, faster processing, lower file size
    "bandpass_freqs": (20, 150),  # (low_hz, high_hz)
    # =================================================================================
    # Core Beat Detection Parameters
    # These settings govern the main algorithm for finding and classifying peaks.
    # =================================================================================
    "min_bpm": 40,   # The absolute minimum BPM the algorithm will consider valid for the long-term belief.
    "max_bpm": 240,  # The absolute maximum BPM the algorithm will consider valid for the long-term belief.
    "min_peak_distance_sec": 0.05,
    # The minimum time allowed between any two raw peaks found in the signal.
    # Increase: Prevents a single, wide heartbeat sound from being detected as multiple peaks.
    # Decrease: Allows detection of very close peaks, useful for high BPMs, but risks detecting noise as peaks.

    "noise_floor_quantile": 0.20,
    # The quantile used to calculate the dynamic noise floor from audio troughs. (0.2 = 20th percentile).
    # Increase: Raises the noise floor, making the algorithm less sensitive. Requires peaks to be taller to be considered.
    # Decrease: Lowers the noise floor, making the algorithm more sensitive. More likely to detect smaller peaks.
    "noise_window_sec": 10,
    # The size of the rolling window (in seconds) for calculating the dynamic noise floor.
    # Increase: The noise floor adapts more slowly to changes in background noise, resulting in a smoother, more stable floor.
    # Decrease: The noise floor adapts very quickly to local changes in noise level.
    "trough_prominence_quantile": 0.1,
    # The prominence required for a dip in the signal to be considered a 'trough' for noise floor calculation.
    # Increase: Requires troughs to be much deeper to be identified. Finds fewer, more significant troughs.
    # Decrease: Allows shallower dips to be considered troughs. Finds more troughs, but can make false positives.
    "peak_prominence_quantile": 0.1,
    # The prominence required for a signal spike to be considered a 'peak' in the first pass.
    # Increase: Requires peaks to stand out more from their surroundings. Filters out minor bumps more aggressively.
    # Decrease: Allows less prominent peaks to be included in the analysis, increasing sensitivity.

    "trough_veto_multiplier": 2.1,
    # Lookahead Veto Logic: If `N * (current_peak - trough) < (next_peak - trough)`, the current peak is vetoed as noise.
    # This rule helps reject a small noise peak that occurs just before a large, real peak.
    # Increase: Makes the veto harder to trigger. Fewer peaks will be vetoed.
    # Decrease: Makes the veto easier to trigger. More aggressive at rejecting small peaks that precede large ones.

    "deviation_smoothing_factor": 0.05,
    # Controls the amount of smoothing applied to the peak-to-peak amplitude deviation series.
    # Increase: More smoothing. The S1/S2 pairing confidence will be based on the general trend of amplitude changes.
    # Decrease: Less smoothing. Pairing confidence will be highly influenced by the amplitude change between two adjacent peaks.

    # =================================================================================
    # S1/S2 Pairing Logic Parameters
    # These settings control how the algorithm decides if two peaks are an S1/S2 pair or a lone S1.
    # =================================================================================
    "s1_s2_interval_cap_sec": 0.4,
    # The absolute maximum time (in seconds) allowed between a detected S1 and a subsequent S2.
    # Increase: Allows pairing of S1-S2 events that are further apart, which may be needed for very slow heart rates.
    # Decrease: Enforces a stricter time limit for pairing, preventing a distant noise peak from being classified as an S2.
    "s1_s2_interval_rr_fraction": 0.7,
    # The S1-S2 interval cannot be longer than this fraction of the current beat-to-beat (RR) interval.
    # Increase: Allows the S1-S2 interval to be a larger portion of the cardiac cycle.
    # Decrease: Restricts the S1-S2 interval to be a smaller fraction of the cycle, useful for higher heart rates.

    "pairing_confidence_threshold": 0.55,
    # The confidence score required to classify two adjacent peaks as an S1-S2 pair.
    # Increase: Requires stronger evidence for pairing. Results in fewer S1-S2 pairs and more "Lone S1" beats.
    # Decrease: Easier to form S1-S2 pairs. Risks incorrectly pairing a true S1 with a nearby noise peak.

    "trough_noise_multiplier": 3.0,
    # A peak is considered "noisy" if the trough preceding it has an amplitude N-times higher than the dynamic noise floor.
    # Increase: Requires the trough to be much louder to be considered noisy. Less likely to classify peaks as noise.
    # Decrease: Even a slightly elevated trough can mark an area as noisy, making noise classification more likely.

    "noise_confidence_threshold": 0.6,
    # If the calculated "noise confidence" for a peak exceeds this value, it will be classified as noise.
    # Increase: Harder to reject a peak. The algorithm needs to be more certain it's noise.
    # Decrease: Easier to reject a peak as noise, making the filter more aggressive.

    "strong_peak_override_ratio": 6.0,
    # A peak whose amplitude is N-times the noise floor will bypass the noise-rejection rules and be kept.
    # Increase: Requires a peak to be exceptionally tall to bypass noise rules.
    # Decrease: Allows moderately tall peaks to be kept even if they are in a section considered noisy.

    # Dynamic HRV-Based Outlier Rejection Parameters
    # =================================================================================
    "rr_interval_max_decrease_pct": 0.45, # A new RR interval can't be more than 45% shorter than the previous one.
    "rr_interval_max_increase_pct": 0.70, # A new RR interval can't be more than 70% longer than the previous one.

    "enable_bpm_boost": True, # Allows disabling the BPM spike prevention boost for the high-confidence pass
    "trough_rejection_multiplier": 4.0, # For trough sanitization, a trough N-times higher than the draft noise floor is rejected.
    "rr_correction_threshold_pct": 0.60, # For post-correction pass, An RR interval shorter than N% of the median is flagged for correction.

    # =================================================================================
    # Long-Term BPM Belief Parameters
    # These settings control the "memory" or "belief" state of the algorithm's BPM calculation.
    # =================================================================================
    "long_term_bpm_learning_rate": 0.05,
    # Controls how much a new beat influences the algorithm's long-term BPM belief (like an exponential moving average).
    # Increase: The BPM belief adapts very quickly to changes. The 'Long-Term BPM' line will closely follow instantaneous BPM.
    # Decrease: The BPM belief is more stable and changes slowly, making it robust against single outlier beats.

    "max_bpm_change_per_beat": 3.0,
    # A "speed limit" on how much the long-term BPM belief can change from one beat to the next.
    # Increase: Allows the BPM belief to make larger jumps, tracking rapid heart rate changes more closely.
    # Decrease: Forces the BPM belief to be smoother, preventing single incorrect detections from skewing the result.

    "output_smoothing_window_sec": 5,
    "save_filtered_wav": True
}


def convert_to_wav(file_path, target_path):
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")
    logging.info(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        sound = AudioSegment.from_file(file_path)
        sound = sound.set_channels(1) # Convert to mono
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"Could not convert file {file_path}. Error: {e}")
        return False


def preprocess_audio(file_path, params):
    """Reads, filters, and prepares the audio envelope for analysis."""
    downsample_factor = params['downsample_factor']
    bandpass_freqs = params['bandpass_freqs']
    save_debug_file = params['save_filtered_wav']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

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
    window_size = new_sample_rate // 10  # 100ms moving average for envelope
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values
    return audio_envelope, new_sample_rate


def _calculate_dynamic_noise_floor(audio_envelope, sample_rate, params):
    """
    Calculates a dynamic noise floor based on a sanitized set of audio troughs
    to avoid corruption from 'divots' in main waveforms.
    """
    min_peak_dist_samples = int(params['min_peak_distance_sec'] * sample_rate)
    trough_prom_thresh = np.quantile(audio_envelope, params['trough_prominence_quantile'])

    # --- STEP 1: Find all potential troughs initially ---
    all_trough_indices, _ = find_peaks(-audio_envelope, distance=min_peak_dist_samples, prominence=trough_prom_thresh)

    # If we don't have enough troughs to begin with, fall back to a simple static floor.
    if len(all_trough_indices) < 5:
        logging.warning("Not enough troughs found for sanitization. Using a static noise floor.")
        fallback_value = np.quantile(audio_envelope, params['noise_floor_quantile'])
        dynamic_noise_floor = pd.Series(fallback_value, index=np.arange(len(audio_envelope)))
        return dynamic_noise_floor, all_trough_indices

    # --- STEP 2: Create a preliminary 'draft' noise floor from ALL troughs ---
    # This draft version is used only to evaluate the troughs themselves.
    trough_series_draft = pd.Series(index=all_trough_indices, data=audio_envelope[all_trough_indices])
    dense_troughs_draft = trough_series_draft.reindex(np.arange(len(audio_envelope))).interpolate()
    noise_window_samples = int(params['noise_window_sec'] * sample_rate)
    quantile_val = params['noise_floor_quantile']
    draft_noise_floor = dense_troughs_draft.rolling(window=noise_window_samples, min_periods=3, center=True).quantile(quantile_val)
    draft_noise_floor = draft_noise_floor.bfill().ffill() # Fill any gaps

    # --- STEP 3: Sanitize the trough list ---
    # A true trough should be close to the actual noise floor. If a trough's amplitude
    # is significantly higher than the draft floor, it's likely a divot in a
    # larger sound wave and should be discarded.
    sanitized_trough_indices = []
    rejection_multiplier = params.get('trough_rejection_multiplier', 4.0)
    for trough_idx in all_trough_indices:
        trough_amp = audio_envelope[trough_idx]
        floor_at_trough = draft_noise_floor.iloc[trough_idx]

        # Keep the trough only if it's not too high above the draft floor
        if not pd.isna(floor_at_trough) and trough_amp <= (rejection_multiplier * floor_at_trough):
            sanitized_trough_indices.append(trough_idx)

    logging.info(f"Trough Sanitization: Kept {len(sanitized_trough_indices)} of {len(all_trough_indices)} initial troughs.")

    # --- STEP 4: Calculate the FINAL, more accurate noise floor using only sanitized troughs ---
    if len(sanitized_trough_indices) > 2:
        trough_series_final = pd.Series(index=sanitized_trough_indices, data=audio_envelope[sanitized_trough_indices])
        dense_troughs_final = trough_series_final.reindex(np.arange(len(audio_envelope))).interpolate()
        dynamic_noise_floor = dense_troughs_final.rolling(window=noise_window_samples, min_periods=3, center=True).quantile(quantile_val)
        dynamic_noise_floor = dynamic_noise_floor.bfill().ffill()
    else:
        # If sanitization removed too many troughs, it's safer to use the original draft floor.
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = draft_noise_floor

    # Final check for any remaining null values
    if dynamic_noise_floor.isnull().all():
         fallback_val = np.quantile(audio_envelope, 0.1)
         dynamic_noise_floor = pd.Series(fallback_val, index=np.arange(len(audio_envelope)))

    # Return the sanitized troughs for potential plotting/debugging in the future
    return dynamic_noise_floor, np.array(sanitized_trough_indices)


def _find_raw_peaks(audio_envelope, height_threshold, params, sample_rate):
    """Finds all potential peaks above the given height threshold."""
    prominence_thresh = np.quantile(audio_envelope, params['peak_prominence_quantile'])
    min_peak_dist_samples = int(params['min_peak_distance_sec'] * sample_rate)

    peaks, _ = find_peaks(
        audio_envelope,
        height=height_threshold,
        prominence=prominence_thresh,
        distance=min_peak_dist_samples
    )
    logging.info(f"Found {len(peaks)} raw peaks using dynamic height threshold.")
    return peaks


def calculate_blended_confidence(deviation, bpm):
    """Calculates a confidence score for pairing two peaks based on amplitude deviation and current BPM."""
    # These curves can be tuned in DEFAULT_PARAMS if abstracted, but are kept here for now.
    conf_at_rest = np.interp(deviation, [0.0, 0.3, 0.6], [0.9, 0.8, 0.1])
    conf_at_exercise = np.interp(deviation, [0.1, 0.4, 0.7], [0.2, 0.9, 0.8])
    conf_at_exertion = np.interp(deviation, [0.0, 0.2, 0.5], [0.05, 0.1, 0.7])
    final_confidence = np.interp(bpm, [80, 130, 170], [conf_at_rest, conf_at_exercise, conf_at_exertion])
    return final_confidence


def find_heartbeat_peaks(audio_envelope, sample_rate, params, start_bpm_hint=None, precomputed_noise_floor=None, precomputed_troughs=None):
    """
    Main logic to classify raw peaks into S1, S2, and Noise.
    This function has been refactored to use helper functions for clarity.
    """
    analysis_data = {}

    # Step 1: Calculate or use pre-calculated dynamic noise floor
    if precomputed_noise_floor is not None and precomputed_troughs is not None:
        logging.info("Using pre-computed noise floor and troughs for this pass.")
        dynamic_noise_floor = precomputed_noise_floor
        trough_indices = precomputed_troughs
    else:
        logging.info("No pre-computed floor found. Calculating noise floor for this pass.")
        dynamic_noise_floor, trough_indices = _calculate_dynamic_noise_floor(audio_envelope, sample_rate, params)

    analysis_data['dynamic_noise_floor_series'] = dynamic_noise_floor
    analysis_data['trough_indices'] = trough_indices

    # Step 2: Find all raw peaks above the noise floor
    all_peaks = _find_raw_peaks(audio_envelope, dynamic_noise_floor.values, params, sample_rate)

    if len(all_peaks) < 2:
        analysis_data.update({"beat_debug_info": {}, "deviation_times": np.array([]), "deviation_series": np.array([]), "long_term_bpm_series": pd.Series(dtype=np.float64)})
        return all_peaks, all_peaks, analysis_data

    # Step 3: Calculate amplitude deviations between peaks to help classification
    peak_amplitudes = audio_envelope[all_peaks]
    normalized_deviations = np.abs(np.diff(peak_amplitudes)) / (np.maximum(peak_amplitudes[:-1], peak_amplitudes[1:]) + 1e-9)
    deviation_times = (all_peaks[:-1] + all_peaks[1:]) / 2 / sample_rate
    smoothing_window_peaks = max(5, int(len(normalized_deviations) * params['deviation_smoothing_factor']))
    smoothed_dev_series = pd.Series(normalized_deviations).rolling(window=smoothing_window_peaks, min_periods=1, center=True).mean().values

    analysis_data["deviation_times"] = deviation_times
    analysis_data["deviation_series"] = smoothed_dev_series

    # Step 4: Stateful classification loop
    long_term_bpm = float(start_bpm_hint) if start_bpm_hint else 80.0
    logging.info(f"Initializing Long-Term BPM to: {long_term_bpm:.1f} BPM")

    candidate_beats, beat_debug_info, long_term_bpm_history = [], {}, []
    sorted_troughs = sorted(trough_indices)

    i = 0
    while i < len(all_peaks):
        current_peak_idx = all_peaks[i]
        reason = ""

        # --- Trough-based Lookahead Veto ---
        if i < len(all_peaks) - 1:
            next_peak_idx = all_peaks[i+1]
            trough_search_start_idx = np.searchsorted(sorted_troughs, current_peak_idx, side='right')
            if trough_search_start_idx < len(sorted_troughs):
                trough_between_idx = sorted_troughs[trough_search_start_idx]
                if trough_between_idx < next_peak_idx:
                    current_peak_amp, next_peak_amp, trough_amp = audio_envelope[current_peak_idx], audio_envelope[next_peak_idx], audio_envelope[trough_between_idx]

                    veto_multiplier = params['trough_veto_multiplier']
                    lhs = veto_multiplier * (current_peak_amp - trough_amp)
                    rhs = next_peak_amp - trough_amp

                    if lhs < rhs:
                        reason = (f"Noise (Vetoed by Lookahead). Condition: {lhs:.1f} < {rhs:.1f}")
                        beat_debug_info[current_peak_idx] = reason
                        i += 1
                        continue

        # --- S2 Logic ---
        expected_rr = 60.0 / long_term_bpm
        s1_s2_max_interval = min(params['s1_s2_interval_cap_sec'], expected_rr * params['s1_s2_interval_rr_fraction'])
        is_potential_s2 = False
        if candidate_beats and (current_peak_idx - candidate_beats[-1]) / sample_rate <= s1_s2_max_interval:
            is_potential_s2 = True

        # --- Noise Confidence & Strong Peak Exception ---
        noise_confidence = 0.0
        preceding_trough_search = np.searchsorted(sorted_troughs, current_peak_idx, side='left')
        if preceding_trough_search > 0:
            preceding_trough_idx = sorted_troughs[preceding_trough_search - 1]
            preceding_trough_amp = audio_envelope[preceding_trough_idx]
            noise_floor_at_trough = dynamic_noise_floor.iloc[preceding_trough_idx]
            if preceding_trough_amp > noise_floor_at_trough * params['trough_noise_multiplier']:
                noise_confidence = 0.8
                reason += f"| Noise Conf: High (Trough is {preceding_trough_amp/noise_floor_at_trough:.1f}x floor) "

        peak_to_floor_ratio = audio_envelope[current_peak_idx] / (dynamic_noise_floor.iloc[current_peak_idx] + 1e-9)
        strong_peak_override = peak_to_floor_ratio >= params['strong_peak_override_ratio']

        if noise_confidence > params['noise_confidence_threshold'] and not is_potential_s2 and not strong_peak_override:
            beat_debug_info[current_peak_idx] = f"Noise (High local noise confidence). {reason.lstrip(' |')}"
            i += 1
            continue
        elif noise_confidence > params['noise_confidence_threshold']:
            reason += "| Bypassed Noise Rule "
            if is_potential_s2: reason += "(Potential S2) "
            if strong_peak_override: reason += f"(Strong Peak: {peak_to_floor_ratio:.1f}x floor)"

        # --- Main Pairing Logic ---
        # This section iterates through peaks, decides if they form S1-S2 pairs or are
        # lone S1 beats, and rejects outliers.
        if i >= len(all_peaks) - 1: # Last peak is always a lone S1
            candidate_beats.append(current_peak_idx)
            beat_debug_info[current_peak_idx] = "Lone S1 (Last Peak)"
            i += 1
            # The loop will terminate, but the BPM update below will run one last time.
        else:
            next_peak_idx = all_peaks[i + 1]
            interval_sec = (next_peak_idx - current_peak_idx) / sample_rate
            smoothed_deviation = smoothed_dev_series[i]

            pairing_confidence = calculate_blended_confidence(smoothed_deviation, long_term_bpm)
            reason += f"| Base Pairing Conf: {pairing_confidence:.2f} "

            # Heuristic: Boost pairing confidence if NOT pairing would cause a massive, unlikely BPM spike
            if params.get("enable_bpm_boost", True):
                bpm_if_not_paired = 60.0 / interval_sec if interval_sec > 0 else 999
                if bpm_if_not_paired > long_term_bpm * 1.7 and long_term_bpm < 150:
                    pairing_confidence = min(0.95, pairing_confidence + 0.3)
                    reason += f"| BOOSTED (Prevented {bpm_if_not_paired:.0f} BPM spike) "

            is_paired = interval_sec <= s1_s2_max_interval and pairing_confidence > params['pairing_confidence_threshold']

            if is_paired:
                s1_idx = current_peak_idx
                candidate_beats.append(s1_idx)
                beat_debug_info[s1_idx] = f"S1 (Paired). {reason.lstrip(' |')}"
                beat_debug_info[next_peak_idx] = f"S2 (Paired). Justification: {reason.lstrip(' |')}"
                i += 2  # Skip S2 in the next iteration
            else:  # Not paired, treat as a potential lone S1.
                s1_idx = current_peak_idx

                # --- BPM Outlier Rejection Logic ---
                if candidate_beats:
                    previous_beat_idx = candidate_beats[-1]
                    rr_interval_sec = (s1_idx - previous_beat_idx) / sample_rate

                    if rr_interval_sec > 0.0:
                        # Get the expected RR interval based on the algorithm's current belief
                        expected_rr_sec = 60.0 / long_term_bpm

                        # Define the plausible range for the new RR interval
                        min_plausible_rr = expected_rr_sec * (1 - params['rr_interval_max_decrease_pct'])
                        max_plausible_rr = expected_rr_sec * (1 + params['rr_interval_max_increase_pct'])

                        # Check if the new interval is outside the plausible window
                        if not (min_plausible_rr <= rr_interval_sec <= max_plausible_rr):
                            instant_bpm = 60.0 / rr_interval_sec
                            rejection_reason = (f"Noise (Rejected: RR interval {rr_interval_sec:.3f}s "
                                                f"is outside plausible range [{min_plausible_rr:.3f}s - {max_plausible_rr:.3f}s] "
                                                f"based on current BPM of {long_term_bpm:.1f}). "
                                                f"Implies instant BPM of {instant_bpm:.0f}.")
                            beat_debug_info[s1_idx] = rejection_reason
                            i += 1
                            continue # Reject this peak and move to the next

                # If the check above passed, proceed to add the peak as a Lone S1.
                candidate_beats.append(s1_idx)
                beat_debug_info[s1_idx] = f"Lone S1. {reason.lstrip(' |')}"
                i += 1

        # --- Long-Term BPM Belief Update ---
        if len(candidate_beats) > 1:
            # The new S1 beat is the last one added.
            new_s1_idx = candidate_beats[-1]
            # The previous S1 beat is the second to last.
            previous_s1_idx = candidate_beats[-2]

            new_rr = (new_s1_idx - previous_s1_idx) / sample_rate
            if new_rr > 0:
                instant_bpm = 60.0 / new_rr
                # Apply learning rate
                lr = params['long_term_bpm_learning_rate']
                target_bpm = ((1 - lr) * long_term_bpm) + (lr * instant_bpm)
                # Limit rate of change
                max_change = params['max_bpm_change_per_beat'] * new_rr
                proposed_change = target_bpm - long_term_bpm
                limited_change = np.clip(proposed_change, -max_change, max_change)
                long_term_bpm = max(params['min_bpm'], min(long_term_bpm + limited_change, params['max_bpm']))

        # Add the current state of the long-term BPM to the history for plotting.
        if candidate_beats:
            long_term_bpm_history.append((candidate_beats[-1] / sample_rate, long_term_bpm))

    final_peaks = np.array(sorted(list(dict.fromkeys(candidate_beats))))
    logging.info(f"Final peak count after stateful grouping: {len(final_peaks)}.")

    analysis_data["beat_debug_info"] = beat_debug_info
    if long_term_bpm_history:
        lt_bpm_times, lt_bpm_values = zip(*long_term_bpm_history)
        analysis_data["long_term_bpm_series"] = pd.Series(lt_bpm_values, index=lt_bpm_times)
    else:
        analysis_data["long_term_bpm_series"] = pd.Series(dtype=np.float64)

    return final_peaks, all_peaks, analysis_data

def correct_peaks_by_rhythm(peaks, audio_envelope, sample_rate, params):
    """
    Refines a list of S1 peaks by removing rhythmically implausible beats.
    If two beats are too close together, the one with the lower amplitude is discarded.
    """
    # If we have too few peaks, correction is unreliable and unnecessary.
    if len(peaks) < 5:
        return peaks

    logging.info(f"--- STAGE 4: Correcting peaks based on rhythm. Initial count: {len(peaks)} ---")

    # Calculate the median R-R interval to establish a stable rhythmic expectation.
    rr_intervals_sec = np.diff(peaks) / sample_rate
    median_rr_sec = np.median(rr_intervals_sec)

    # Any interval shorter than this threshold is considered a conflict.
    correction_threshold_sec = median_rr_sec * params.get("rr_correction_threshold_pct", 0.6)
    logging.info(f"Median R-R: {median_rr_sec:.3f}s. Correction threshold: {correction_threshold_sec:.3f}s.")

    # We build a new list of corrected peaks. Start with the first peak as a given.
    corrected_peaks = [peaks[0]]

    # Iterate through the original peaks, starting from the second one.
    for i in range(1, len(peaks)):
        current_peak = peaks[i]
        last_accepted_peak = corrected_peaks[-1]

        interval_sec = (current_peak - last_accepted_peak) / sample_rate

        if interval_sec < correction_threshold_sec:
            # CONFLICT: The current peak is too close to the last accepted one.
            # We must decide which one to keep. The one with the higher amplitude wins.
            last_peak_amp = audio_envelope[last_accepted_peak]
            current_peak_amp = audio_envelope[current_peak]

            if current_peak_amp > last_peak_amp:
                # The current peak is stronger, so it REPLACES the last accepted peak.
                logging.info(f"Conflict at {current_peak/sample_rate:.2f}s. Replaced previous peak at {last_accepted_peak/sample_rate:.2f}s due to higher amplitude.")
                corrected_peaks[-1] = current_peak
            else:
                # The last accepted peak was stronger, so we DISCARD the current peak.
                logging.info(f"Conflict at {current_peak/sample_rate:.2f}s. Discarding current peak due to lower amplitude.")
                pass  # Do nothing, effectively dropping the current_peak
        else:
            # NO CONFLICT: The interval is plausible. Add the peak to our corrected list.
            corrected_peaks.append(current_peak)

    final_peak_count = len(corrected_peaks)
    if final_peak_count < len(peaks):
        logging.info(f"Correction complete. Removed {len(peaks) - final_peak_count} peak(s). Final count: {final_peak_count}")
    else:
        logging.info("Correction pass complete. No rhythmic conflicts found.")

    return np.array(corrected_peaks)

def calculate_hrv_metrics(s1_peaks, sample_rate):
    """Calculates SDNN and RMSSD from a series of S1 heartbeats."""
    if len(s1_peaks) < 10: # Need a reasonable number of beats for meaningful HRV
        return None, None, None

    # Calculate R-R intervals in milliseconds
    rr_intervals_sec = np.diff(s1_peaks) / sample_rate
    rr_intervals_ms = rr_intervals_sec * 1000

    if len(rr_intervals_ms) < 2:
        return None, None, None

    avg_rr = np.mean(rr_intervals_ms)
    sdnn = np.std(rr_intervals_ms)

    # Calculate successive differences
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs**2))

    return avg_rr, sdnn, rmssd

def calculate_bpm_series(peaks, sample_rate, params):
    """Calculates and smooths the final BPM series from S1 peaks."""
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
        smoothing_window_sec = params['output_smoothing_window_sec']
        beats_in_window = max(2, int(np.ceil((smoothing_window_sec / 60) * avg_heart_rate)))
        smoothed_bpm = bpm_series.rolling(window=beats_in_window, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)
    return smoothed_bpm, bpm_series.index.values


def plot_results(audio_envelope, peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, file_name, params, hrv_metrics=None):
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")), secondary_y=False)

    if 'trough_indices' in analysis_data and analysis_data['trough_indices'].size > 0:
        trough_indices = analysis_data['trough_indices']
        fig.add_trace(go.Scatter(
            x=trough_indices / sample_rate,
            y=audio_envelope[trough_indices],
            mode='markers',
            name='Troughs',
            marker=dict(color='green', symbol='circle-open', size=6),
            visible='legendonly'
        ), secondary_y=False)

    if 'dynamic_noise_floor_series' in analysis_data and not analysis_data['dynamic_noise_floor_series'].empty:
        noise_floor_series = analysis_data['dynamic_noise_floor_series']
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=noise_floor_series.values,
            name="Dynamic Noise Floor",
            line=dict(color="green", dash="dot", width=1.5),
            hovertemplate="Noise Floor: %{y:.2f}<extra></extra>"
        ), secondary_y=False)

    # --- Peak Plotting Logic ---
    all_peaks_data = {
        'S1': {'indices': [], 'customdata': []},
        'S2': {'indices': [], 'customdata': []},
        'Noise': {'indices': [], 'customdata': []}
    }
    debug_info = analysis_data.get('beat_debug_info', {})

    for p_idx in all_raw_peaks:
        reason_str = debug_info.get(p_idx, 'Unknown')
        peak_time = p_idx / sample_rate
        peak_amp = audio_envelope[p_idx]

        category = 'Noise'
        if reason_str.startswith('S1') or reason_str.startswith('Lone S1'):
            category = 'S1'
        elif reason_str.startswith('S2'):
            category = 'S2'

        peak_type = reason_str
        reason_details = ""

        # Parse S1 and Noise reasons
        if category in ['S1', 'Noise']:
            if '. ' in reason_str:
                parts = reason_str.split('. ', 1)
                peak_type = parts[0].strip()
                reason_details = parts[1].strip().replace(' | ', '<br>- ')
            elif '(' in reason_str and ')' in reason_str:
                peak_type_part = reason_str[:reason_str.find('(')].strip()
                details_part = reason_str[reason_str.find('(') + 1:reason_str.rfind(')')].strip()
                peak_type = peak_type_part
                reason_details = details_part
        # Parse new detailed S2 reasons
        elif category == 'S2':
            if '. Justification: ' in reason_str:
                parts = reason_str.split('. Justification: ', 1)
                peak_type = "S2"
                reason_details = f"{parts[0]}.<br><b>Justification:</b><br>- {parts[1].strip().replace(' | ', '<br>- ')}"
            else: # Fallback for simple S2 message
                peak_type = "S2"
                reason_details = reason_str

        custom_data_tuple = (peak_type, peak_time, peak_amp, reason_details)
        all_peaks_data[category]['indices'].append(p_idx)
        all_peaks_data[category]['customdata'].append(custom_data_tuple)

    # Generic hover template for all peak types
    peak_hovertemplate = ("<b>Calculated Peak Type:</b> %{customdata[0]}<br>" +
                          "<b>Time:</b> %{customdata[1]:.2f}s<br>" +
                          "<b>Amp:</b> %{customdata[2]:.0f}<br>" +
                          "<b>Reason:</b><br>- %{customdata[3]}<extra></extra>")

    # Add S1 trace
    if all_peaks_data['S1']['indices']:
        s1_indices = np.array(all_peaks_data['S1']['indices'])
        s1_customdata = np.stack(all_peaks_data['S1']['customdata'], axis=0)
        fig.add_trace(go.Scatter(
            x=s1_indices / sample_rate, y=audio_envelope[s1_indices],
            mode='markers', name='S1 Beats',
            marker=dict(color='#e36f6f', size=8, symbol='diamond'),
            customdata=s1_customdata,
            hovertemplate=peak_hovertemplate
        ), secondary_y=False)

    # Add S2 trace with the same detailed hover template
    if all_peaks_data['S2']['indices']:
        s2_indices = np.array(all_peaks_data['S2']['indices'])
        s2_customdata = np.stack(all_peaks_data['S2']['customdata'], axis=0)
        fig.add_trace(go.Scatter(
            x=s2_indices / sample_rate, y=audio_envelope[s2_indices],
            mode='markers', name='S2 Beats',
            marker=dict(color='orange', symbol='circle', size=6),
            customdata=s2_customdata,
            hovertemplate=peak_hovertemplate
        ), secondary_y=False)

    # Add Noise trace
    if all_peaks_data['Noise']['indices']:
        noise_indices = np.array(all_peaks_data['Noise']['indices'])
        noise_customdata = np.stack(all_peaks_data['Noise']['customdata'], axis=0)
        fig.add_trace(go.Scatter(
            x=noise_indices / sample_rate, y=audio_envelope[noise_indices],
            mode='markers', name='Noise/Rejected Peaks',
            marker=dict(color='grey', symbol='x', size=6),
            customdata=noise_customdata,
            hovertemplate=peak_hovertemplate
        ), secondary_y=False)

    if not smoothed_bpm.empty:
        fig.add_trace(go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM",
                                 line=dict(color="#4a4a4a", width=3, dash='solid')), secondary_y=True)

    if "long_term_bpm_series" in analysis_data and not analysis_data["long_term_bpm_series"].empty:
        lt_series = analysis_data["long_term_bpm_series"]
        fig.add_trace(go.Scatter(x=lt_series.index, y=lt_series.values, name="Long-Term BPM",
                                 line=dict(color='orange', width=2, dash='dot')), secondary_y=True)

    if 'deviation_series' in analysis_data and analysis_data['deviation_series'] is not None:
        deviation_percent = analysis_data['deviation_series'] * 100
        fig.add_trace(go.Scatter(x=analysis_data['deviation_times'], y=deviation_percent, name='Norm. Deviation %',
                                 line=dict(color='purple', width=2), visible='legendonly',
                                 hovertemplate='Norm. Deviation: %{y:.2f}%<extra></extra>'), secondary_y=True)

    if not smoothed_bpm.empty:
        max_bpm_val, min_bpm_val = smoothed_bpm.max(), smoothed_bpm.min()
        max_bpm_time, min_bpm_time = smoothed_bpm.idxmax(), smoothed_bpm.idxmin()
        fig.add_annotation(x=max_bpm_time, y=max_bpm_val, text=f"Max: {max_bpm_val:.1f} BPM", showarrow=True,
                           arrowhead=1, ax=20, ay=-40, font=dict(color="#e36f6f"), yref="y2")
        fig.add_annotation(x=min_bpm_time, y=min_bpm_val, text=f"Min: {min_bpm_val:.1f} BPM", showarrow=True,
                           arrowhead=1, ax=20, ay=40, font=dict(color="#a3d194"), yref="y2")

    max_time_sec = time_axis[-1] if len(time_axis) > 0 else 1
    tick_vals_sec = np.arange(0, max_time_sec + 30, 30)
    tick_text_combined = [f"{int(s)}s ({int(s // 60):02d}:{int(s % 60):02d})" for s in tick_vals_sec]

    # --- HRV & Summary Annotation Box ---
    if hrv_metrics:
        # Format the text with line breaks for the plot
        # Using .get(key, 'N/A') is a safe way to access dictionary keys that might be missing
        avg_bpm_val = hrv_metrics.get('avg_bpm')
        min_bpm_val = hrv_metrics.get('min_bpm')
        max_bpm_val = hrv_metrics.get('max_bpm')
        sdnn_val = hrv_metrics.get('sdnn')
        rmssd_val = hrv_metrics.get('rmssd')

        annotation_text = "<b>Analysis Summary</b><br>"
        annotation_text += f"Avg BPM: {avg_bpm_val:.1f}<br>" if avg_bpm_val is not None else ""
        annotation_text += f"Min/Max BPM: {min_bpm_val:.1f} / {max_bpm_val:.1f}<br>" if min_bpm_val is not None else ""
        annotation_text += f"SDNN (short-term HRV): {sdnn_val:.1f} ms<br>" if sdnn_val is not None else ""
        annotation_text += f"RMSSD (overall HRV): {rmssd_val:.1f} ms" if rmssd_val is not None else ""

        fig.add_annotation(
            text=annotation_text,
            align='left',
            showarrow=False,
            xref='paper', # Positions the box relative to the plotting area, not the data
            yref='paper',
            x=0.02,       # 2% from the left edge
            y=0.98,       # 98% from the bottom edge (i.e., in the top-left corner)
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(255, 253, 231, 0.4)' # A semi-transparent light yellow
        )

    fig.update_layout(title_text=f"Heartbeat Analysis - {os.path.basename(file_name)} (v7.0)", dragmode='pan',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(t=100, b=100),
                      xaxis=dict(title_text="Time", tickvals=tick_vals_sec, ticktext=tick_text_combined),
                      hovermode='x unified')

    robust_upper_limit = np.quantile(audio_envelope, 0.95) # 95th percentile to ignore outliers
    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[0, robust_upper_limit * 60])
    fig.update_yaxes(title_text="BPM / Norm. Dev %", secondary_y=True, range=[params['min_bpm'] - 10, params['max_bpm'] + 10])

    output_html_path = f"{os.path.splitext(file_name)[0]}_bpm_plot.html"
    fig.write_html(output_html_path, config={'scrollZoom': True})
    logging.info(f"Interactive plot saved to {output_html_path}")


def save_bpm_to_csv(bpm_series, time_points, output_path):
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time (s)", "BPM"])
        if not bpm_series.empty:
            for t, bpm in zip(time_points, bpm_series):
                if not np.isnan(bpm): writer.writerow([f"{t:.2f}", f"{bpm:.1f}"])
    logging.info(f"BPM data saved to {output_path}")


def create_chronological_log_file(audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm,
                                     output_log_path, file_name):
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    debug_info = analysis_data.get('beat_debug_info', {})
    dynamic_noise_floor_series = analysis_data.get('dynamic_noise_floor_series')
    lt_bpm_series = analysis_data.get("long_term_bpm_series", pd.Series(dtype=np.float64))
    dev_times = analysis_data.get('deviation_times', np.array([]))
    dev_values = analysis_data.get('deviation_series', np.array([]))
    dev_series = pd.Series(dev_values, index=dev_times)
    loggable_events = []

    for p in all_raw_peaks:
        reason = debug_info.get(p)
        if reason and not reason.startswith('Unknown'):
            loggable_events.append({
                'time': p / sample_rate,
                'type': 'Peak',
                'amp': audio_envelope[p],
                'reason': reason
            })

    if 'trough_indices' in analysis_data:
        for p in analysis_data['trough_indices']:
            loggable_events.append({
                'time': p / sample_rate,
                'type': 'Trough',
                'amp': audio_envelope[p]
            })

    loggable_events.sort(key=itemgetter('time'))
    logging.info(f"Generating readable debug log at '{output_log_path}'...")

    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# Chronological Debug Log for {os.path.basename(file_name)}\n")
        log_file.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with Engine v7.0\n\n")

        if not loggable_events:
            log_file.write("No significant events (peaks or troughs) were detected to log.\n")
            return

        for event in loggable_events:
            t = event['time']
            envelope_idx = np.abs(time_axis - t).argmin()
            envelope_val = audio_envelope[envelope_idx]
            noise_floor_val = dynamic_noise_floor_series.iloc[
                envelope_idx] if dynamic_noise_floor_series is not None else 'N/A'
            time_as_series = pd.Series([None], index=[t])
            smoothed_val = smoothed_bpm.reindex(time_as_series.index, method='nearest', tolerance=0.5).iloc[
                0] if not smoothed_bpm.empty else np.nan
            lt_bpm_val = lt_bpm_series.reindex(time_as_series.index, method='nearest', tolerance=0.5).iloc[
                0] if not lt_bpm_series.empty else np.nan
            dev_val = dev_series.reindex(time_as_series.index, method='nearest', tolerance=0.5).iloc[
                0] if not dev_series.empty else np.nan

            log_file.write(f"## Time: `{t:.4f}s`\n")

            if event['type'] == 'Peak':
                reason = event['reason']
                status_line = ""
                # This logic now handles S1, S2, and Noise peaks uniformly.
                if '. ' in reason:
                    # Check for the detailed S2 format first
                    if '. Justification: ' in reason:
                        parts = reason.split('. Justification: ', 1)
                        status = parts[0].strip()
                        details = parts[1].strip().replace(' | ', '\n- ')
                        status_line = f"**{status}.**\n- **Justification:** {details}"
                    else:
                        parts = reason.split('. ', 1)
                        status = parts[0].strip()
                        details = parts[1].strip().replace(' | ', '\n- ')
                        status_line = f"**{status}.**\n- {details}"
                else:
                    status_line = f"**{reason}**"

                log_file.write(f"{status_line}\n")
                log_file.write(f"**Audio Envelope**: `{envelope_val:.2f}`\n")
                log_file.write(f"**Noise Floor**: `{noise_floor_val:.2f}`\n")
                log_file.write(f"**Raw Peak** (Amp: {event['amp']:.2f})\n")

                if 'S1' in reason:
                    if not pd.isna(smoothed_val): log_file.write(f"**Smoothed BPM**: {smoothed_val:.2f}\n")
                    if not pd.isna(lt_bpm_val): log_file.write(f"**Long-Term BPM (Belief)**: {lt_bpm_val:.2f}\n")
                    if not pd.isna(dev_val): log_file.write(f"**Norm. Deviation**: {dev_val * 100:.2f}%\n")

            elif event['type'] == 'Trough':
                log_file.write(f"**Trough Detected** (Amp: {event['amp']:.2f})\n")
                log_file.write(f"**Audio Envelope**: `{envelope_val:.2f}`\n")
                log_file.write(f"**Noise Floor**: `{noise_floor_val:.2f}`\n")

            log_file.write("\n")

    logging.info("Debug log generation complete.")


def analyze_wav_file(wav_file_path, params, start_bpm_hint): # We keep the signature for GUI compatibility
    """
    Main analysis pipeline that orchestrates multiple analysis passes.
    """
    file_name_no_ext = os.path.splitext(wav_file_path)[0]
    logging.info(f"--- Processing file: {os.path.basename(wav_file_path)} ---")

    # --- Initial Preprocessing ---
    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params)

    # =================================================================================
    # STAGE 1: AUTOMATED GLOBAL BPM ESTIMATION
    # =================================================================================
    logging.info("--- STAGE 1: Running High-Confidence pass to find anchor beats ---")

    params_pass_1 = params.copy()
    params_pass_1["pairing_confidence_threshold"] = 0.75  # Stricter pairing
    params_pass_1["enable_bpm_boost"] = False            # Disable boosting heuristic

    # Run the first pass to get a reliable "rhythm skeleton"
    anchor_beats, _, _ = find_heartbeat_peaks(audio_envelope, sample_rate, params_pass_1)

    global_bpm_estimate = None
    if len(anchor_beats) >= 10: # Need enough beats for a reliable estimate
        rr_intervals_sec = np.diff(anchor_beats) / sample_rate
        # Use the median to get a robust estimate, ignoring outlier gaps
        median_rr_sec = np.median(rr_intervals_sec)

        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            logging.info(f"Automatically determined Global BPM Estimate: {global_bpm_estimate:.1f} BPM")
        else:
            logging.warning("Median R-R interval was zero. Cannot estimate BPM.")
    else:
        logging.warning(f"Found only {len(anchor_beats)} anchor beats. Falling back to user hint or default.")

    # Decision logic for the starting BPM
    # If a user provides a hint, it's probably for a good reason, so we can prioritize it.
    if start_bpm_hint:
        final_start_bpm = start_bpm_hint
        logging.info(f"Using user-provided starting BPM of {start_bpm_hint:.1f} BPM for main analysis.")
    elif global_bpm_estimate:
        final_start_bpm = global_bpm_estimate
        logging.info(f"Using automatically determined BPM of {global_bpm_estimate:.1f} BPM for main analysis.")
    else:
        final_start_bpm = 80.0 # Fallback default
        logging.warning("Could not determine starting BPM. Using fallback default of 80.0 BPM.")

    # =================================================================================
    # STAGE 2: TROUGH SANITIZATION & NOISE FLOOR REFINEMENT
    # =================================================================================
    logging.info("--- STAGE 2: Performing trough sanitization for a refined noise floor ---")
    # We use the main 'params' dictionary for the final, most accurate calculation
    sanitized_noise_floor, sanitized_troughs = _calculate_dynamic_noise_floor(audio_envelope, sample_rate, params)

    # =================================================================================
    # STAGE 3: PRIMARY ANALYSIS PASS
    # =================================================================================
    logging.info("--- STAGE 3: Running Main Analysis Pass with refined inputs ---")

    # Run the main analysis pass using the determined start BPM and the sanitized noise floor
    peaks, all_raw_peaks, analysis_data = find_heartbeat_peaks(
        audio_envelope,
        sample_rate,
        params,
        start_bpm_hint=final_start_bpm,
        precomputed_noise_floor=sanitized_noise_floor,
        precomputed_troughs=sanitized_troughs
    )

    # =================================================================================
    # STAGE 4: POST-CORRECTION PASS (PEAK VALIDATION)
    # =================================================================================
    # This final pass corrects the S1 peaks based on rhythmic plausibility.
    final_peaks = correct_peaks_by_rhythm(peaks, audio_envelope, sample_rate, params)


    # --- FINAL CALCULATIONS AND OUTPUT ---
    # From this point on, all calculations must use 'final_peaks' instead of 'peaks'.

    if len(final_peaks) < 2:
        logging.warning("Not enough S1 peaks detected after correction to calculate BPM.")
        plot_results(audio_envelope, final_peaks, all_raw_peaks, analysis_data, pd.Series(dtype=np.float64), np.array([]), sample_rate, wav_file_path, params)
        return

    # Pass the corrected peaks to the final calculation functions.
    smoothed_bpm, bpm_times = calculate_bpm_series(final_peaks, sample_rate, params)
    avg_rr, sdnn, rmssd = calculate_hrv_metrics(final_peaks, sample_rate)
    hrv_stats_for_plot = {}
    if not smoothed_bpm.empty:
        hrv_stats_for_plot['avg_bpm'] = smoothed_bpm.mean()
        hrv_stats_for_plot['min_bpm'] = smoothed_bpm.min()
        hrv_stats_for_plot['max_bpm'] = smoothed_bpm.max()
    if avg_rr is not None:
        hrv_stats_for_plot['sdnn'] = sdnn
        hrv_stats_for_plot['rmssd'] = rmssd

    output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
    save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)

    plot_results(audio_envelope, final_peaks, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times, sample_rate, wav_file_path, params, hrv_metrics=hrv_stats_for_plot)
    # The chronological log can still use all_raw_peaks to show everything,
    # but the final classifications and BPM are now based on the corrected peaks.
    output_log_path = f"{file_name_no_ext}_Debug_Log.md"
    create_chronological_log_file(audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm, output_log_path, wav_file_path)


# --- GUI Class ---
class BPMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer v7.1 (Thread-Safe)")
        self.root.geometry("550x350")
        self.style = ttkb.Style(theme='minty')
        self.current_file = None
        self.params = DEFAULT_PARAMS.copy()
        self.log_queue = queue.Queue()
        self.create_widgets()
        self.root.after(100, self.process_log_queue)
        self._find_initial_audio_file()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=450)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.select_file, bootstyle=INFO)
        browse_btn.pack(side=tk.RIGHT, padx=5)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="Starting BPM (optional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bpm_entry = ttk.Entry(param_frame)
        self.bpm_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.start_analysis_thread, bootstyle=SUCCESS, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Select an audio file to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        param_frame.columnconfigure(1, weight=1)

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message_type, data = self.log_queue.get(0)

                if message_type == "status":
                    self.status_var.set(data)
                elif message_type == "analysis_complete":
                    self.status_var.set("Analysis complete!")
                    self.analyze_btn.config(state=tk.NORMAL)
                elif message_type == "error":
                     self.status_var.set("An error occurred. Check logs.")
                     self.analyze_btn.config(state=tk.NORMAL)
                     messagebox.showerror("Analysis Error", data)

        finally:
            # Reschedule itself to run again after 100ms
            self.root.after(100, self.process_log_queue)

    def select_file(self):
        filetypes = [('Audio files', '*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(title="Select audio file", filetypes=filetypes)
        if filename:
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self.analyze_btn.config(state=tk.NORMAL)
            self._update_status(f"Ready to analyze: {os.path.basename(filename)}")

    def _find_initial_audio_file(self):
        supported = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4')
        try:
            for filename in os.listdir(os.getcwd()):
                if filename.lower().endswith(supported):
                    self.current_file = os.path.join(os.getcwd(), filename)
                    self.file_label.config(text=os.path.basename(self.current_file))
                    self.analyze_btn.config(state=tk.NORMAL)
                    self._update_status(f"Auto-loaded: {os.path.basename(self.current_file)}")
                    return
        except Exception as e:
            logging.error(f"Could not auto-find file: {e}")

    def _update_status(self, message):
        """Safely update the status bar from any thread."""
        self.root.after(0, lambda: self.status_var.set(message))

    def start_analysis_thread(self):
        """Starts the analysis in a new thread."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return

        self.analyze_btn.config(state=tk.DISABLED)
        self._update_status("Starting analysis...")

        analysis_thread = threading.Thread(target=self._run_analysis_in_background)
        analysis_thread.daemon = True # Allows app to exit even if thread is running
        analysis_thread.start()

    def _run_analysis_in_background(self):
        try:
            bpm_input = self.bpm_entry.get().strip()
            start_bpm_hint = float(bpm_input) if bpm_input else None

            converted_dir = os.path.join(os.getcwd(), "converted_wavs")
            os.makedirs(converted_dir, exist_ok=True)
            base_name, ext = os.path.splitext(self.current_file)
            wav_path = os.path.join(converted_dir, f"{os.path.basename(base_name)}.wav")

            if ext.lower() != '.wav':
                # Put status messages on the queue
                self.log_queue.put(("status", "Converting file to WAV..."))
                if not convert_to_wav(self.current_file, wav_path):
                    self.log_queue.put(("error", "File conversion failed."))
                    return
            else:
                import shutil
                shutil.copy(self.current_file, wav_path)

            self.log_queue.put(("status", "Processing and analyzing heartbeat..."))
            analyze_wav_file(wav_path, self.params, start_bpm_hint)

            # Signal completion
            self.log_queue.put(("analysis_complete", None))

        except Exception as e:
            # Put the error message on the queue
            error_info = f"An error occurred:\n{str(e)}"
            self.log_queue.put(("error", error_info))
            logging.error(f"Full analysis error: {traceback.format_exc()}")


def main():
    root = ttkb.Window(themename="minty")
    app = BPMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()