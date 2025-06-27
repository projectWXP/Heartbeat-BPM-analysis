# Advanced Heartbeat Analyzer
## Overview
This Python-based application provides an in-depth analysis of heart sounds from audio recordings. It is designed for researchers, bio-acoustics enthusiasts, and developers working with physiological signals. The script processes audio files (like `.wav`, `.mp3`, `.m4a`), identifies individual heartbeats (S1 and S2 sounds), and calculates a wide range of cardiovascular metrics. The results are presented in an interactive HTML plot, a detailed Markdown summary, and a debug log for granular inspection.

The analysis is powered by a sophisticated multi-pass algorithm that uses a dynamic noise floor, stateful peak classification, and rhythm-based correction to ensure high accuracy, even with noisy recordings.
## Key Features
- **Multi-Format Audio Support**: Automatically converts common audio/video formats (`.mp3`, `.m4a`, `.flac`, `.mp4`, etc.) to WAV for analysis using FFmpeg.
- **Robust** Beat **Detection**: Employs a multi-stage process to accurately identify S1 (lub) and S2 (dub) heart sounds while rejecting noise.
- **Dynamic Noise Profiling**: Calculates a dynamic noise floor that adapts to changing background noise levels, improving peak detection in non-ideal conditions.
- **Stateful Analysis**: Uses a "belief" system about the current heart rate to intelligently accept or reject beats that are rhythmically implausible.
- **Comprehensive Metrics**: Calculates and reports on:
    - **BPM (Beats Per Minute)**: Smoothed, time-varying heart rate.
    - **HRV (Heart Rate Variability)**: Including RMSSD (Root Mean Square of Successive Differences) and SDNN (Standard Deviation of NN intervals).
    - **HRR (Heart Rate Recovery)**: Measures the BPM drop after peak exertion (e.g., 1-minute HRR).
    - **Exertion** & **Recovery Slopes**: Identifies periods of the fastest heart rate increase and decrease.
- **Rich, Interactive Outputs**:
    1. **Interactive HTML Plot**: A `plotly`-based chart showing the audio envelope, detected beats (S1, S2, Noise), BPM trends, and HRV metrics. Fully zoomable and pannable.
    2. **Markdown Analysis Report**: A clean, readable summary of all key findings, perfect for documentation.
    3. **Chronological Debug Log**: A detailed, time-stamped log of every decision the algorithm makes, for deep dives and parameter tuning.
- **User-Friendly GUI**: A simple Tkinter interface for file selection and initiating the analysis.
- **Highly Configurable**: All key algorithm parameters are centralized in a single dictionary, allowing for easy tuning and experimentation.
## How It Works
The analysis pipeline is composed of several key stages:
1. **Preprocessing**: The input audio is converted to a mono WAV file, downsampled for efficiency, and band-pass filtered to isolate typical heart sound frequencies (20-150 Hz). A smoothed signal envelope is then generated.
2. **High-Confidence Pass (BPM Estimation)**: The algorithm first runs with very strict settings to find only the most obvious, "anchor" beats. The median interval between these beats provides a robust global estimate of the starting BPM.
3. **Noise Floor Calculation**: The script identifies audio troughs (dips in the signal) and sanitizes them to calculate a refined, dynamic noise floor. This floor acts as the threshold for peak detection.
4. **Primary Analysis Pass**: Using the estimated BPM and refined noise floor, the main `find_heartbeat_peaks` function performs a stateful analysis. It iterates through all potential peaks, classifying them as S1, S2, or Noise based on timing, amplitude, and rhythmic plausibility.
5. **Rhythm Correction**: A post-processing step reviews the list of detected S1 beats. If two beats are too close together, it discards the one with the lower amplitude, correcting for potential double-detections.
6. **Metric Calculation & Output Generation**: With a final, clean list of S1 peaks, the script calculates all BPM, HRV, and recovery metrics. It then generates the interactive plot and summary files.

## Requirements
### Python Libraries
You can install all necessary libraries using pip:

```
pip install numpy pandas scipy plotly pydub ttkbootstrap
```

### External Dependencies
- **FFmpeg**: This is **required** for converting audio files that are not in `.wav` format.
    - **Windows**: Download the binaries from the [official FFmpeg website](https://ffmpeg.org/download.html "null"), extract them, and add the `bin` folder to your system's `PATH`.
    - **macOS (using Homebrew)**: `brew install ffmpeg`
    - **Linux** (using **apt)**: `sudo apt-get install ffmpeg`
## How to Use
1. **Clone the Repository**:
    ```
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2. **Install Dependencies**: Make sure you have installed both the Python libraries and FFmpeg as described above.
3. **Run the Script**:
    ```
    python bpm_analysis_v5.7.py
    ```
4. **Use the GUI**:
    - The application window will appear.
    - The script will automatically try to find a supported audio file in the same directory.
    - If a file isn't loaded automatically, click the **"Browse"** button to select your audio file.
    - _(Optional)_ Enter a "Starting BPM" if you have a rough idea of the heart rate. This can help guide the algorithm, but it's not required.
    - Click the **"Analyze"** button.
5. **Check the Outputs**:
    - The analysis will run in the background. The status bar will show the progress.
    - Once complete, check the directory where your original audio file is located. You will find the generated output files:
        - `your_audio_file_bpm_plot.html`
        - `your_audio_file_Analysis_Summary.md`
        - `your_audio_file_Debug_Log.md`
## Configuration & Tuning
All core algorithm parameters are located in the `DEFAULT_PARAMS` dictionary at the top of the script. This allows you to easily experiment and tune the analyzer's behavior without digging through the code. The parameters are commented to explain their function.
**Example:** If you are analyzing a very noisy recording, you might want to make the algorithm less sensitive. You could try:
- Increasing `noise_floor_quantile` (e.g., from `0.20` to `0.30`) to raise the noise threshold.
- Increasing `pairing_confidence_threshold` (e.g., from `0.55` to `0.65`) to require stronger evidence for S1-S2 pairing.
Conversely, for a very clean recording of a high heart rate, you might:
- Decrease `min_peak_distance_sec` to allow for closer peaks.
- Decrease `s1_s2_interval_cap_sec` to enforce a tighter pairing window.
## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
