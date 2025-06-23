# Heartbeat BPM Analyzer

This Python script provides a graphical user interface (GUI) application for analyzing audio files to detect heartbeats and calculate Beats Per Minute (BPM) over time. It can process various audio formats by converting them to WAV using FFmpeg (if available) and visualizes the results with an interactive Plotly graph.

## Features
- **GUI Interface:** Easy-to-use Tkinter-based GUI for file selection and analysis initiation.
- **Audio Conversion:** Automatically converts various audio formats (MP3, M4A, FLAC, OGG, MP4, MKV) to WAV for processing, leveraging `pydub` and `FFmpeg`.
- **Heartbeat Detection:** Employs signal processing techniques (band-pass filtering, envelope detection, peak finding) to identify heartbeats.
- **BPM Calculation:** Calculates instantaneous and smoothed BPM over the duration of the audio.
- **Interactive Visualization:** Generates an interactive HTML plot using `Plotly` showing the audio envelope, detected heartbeats, and smoothed BPM, complete with zoom and pan capabilities.
- **CSV Export:** Saves the calculated BPM data (time and BPM values) to a CSV file.
- **BPM Hint:** Allows users to provide an optional starting BPM hint to improve detection accuracy.

## Prerequisites
Before running the script, ensure you have the following installed:
- **Python 3.x**
- **FFmpeg:** This is crucial for converting non-WAV audio files. Download it from the [FFmpeg website](https://ffmpeg.org/download.html "null") and make sure it's added to your system's PATH.

## Installation
1. **Clone the repository (or download the script):**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    (Replace `your-username` and `your-repo-name` with your actual GitHub details.)
2. **Install the required Python libraries:**
    ```
    pip install numpy pandas scipy plotly pydub scikit-learn tkinter ttkbootstrap
    ```
    - **Note:** `tkinter` is usually included with Python, but `ttkbootstrap` needs to be installed separately.

## How to Use
1. **Run the script:**
    ```
    python bpm_analysis v3.1.5.py
    ```
2. **Select an Audio File:**
    - The application will try to auto-load the first supported audio file found in the directory where the script is run.
    - If no file is auto-loaded, or if you want to analyze a different file, click the "Browse" button to select your audio file (`.wav`, `.mp3`, `.m4a`, etc.).
3. **(Optional) Enter Starting BPM Hint:**
    - If you have an idea of the approximate BPM, enter it in the "Starting BPM (optional)" field. This can help the algorithm refine its heartbeat detection.
4. **Analyze:**
    - Click the "Analyze" button.
    - The script will first convert the audio to a mono WAV file (if it's not already WAV) and save it in a `converted_wavs` directory.
    - It will then perform the BPM analysis.
    - A status message will update in the GUI.
5. **View Results:**
    - An interactive HTML plot will be generated and saved in the same directory as your input file (e.g., `your_audio_file_bpm_plot.html`). Open this HTML file in your web browser to view the analysis.
    - A CSV file containing time and BPM data will also be saved (e.g., `your_audio_file_bpm_analysis.csv`).

## Output Files
Upon successful analysis, the script will generate the following files in the same directory as your original audio file (or in the `converted_wavs` directory for the WAV conversion):
- `[audio_filename]_bpm_plot.html`: An interactive Plotly graph visualizing the audio envelope, detected heartbeats, and smoothed BPM.
- `[audio_filename]_bpm_analysis.csv`: A CSV file containing two columns: "Time (s)" and "BPM", listing the calculated BPM over time.
- `[audio_filename]_filtered_debug.wav` (optional): If `save_debug_wav` parameter is set to `True` in the script, a filtered version of the WAV file will be saved for debugging purposes.

## Troubleshooting
- **"Pydub/FFmpeg is required..." Error:** Ensure FFmpeg is correctly installed and its executable path is added to your system's PATH environment variable. Re-install `pydub` if necessary (`pip install pydub`).
- **"Error during analysis"**: Check the console for detailed traceback messages. Common issues include corrupted audio files or unsupported formats that FFmpeg cannot process.
- **No peaks detected / Incorrect BPM**:
    - Try adjusting the `Starting BPM (optional)` hint.
    - The audio quality might be poor, or the heart sounds are not distinct enough.
    - For very quiet or noisy recordings, the default parameters might not be optimal.

## License

This project is open-source and available under the [MIT License](LICENSE.md "null").
