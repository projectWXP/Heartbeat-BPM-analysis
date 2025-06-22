# Heartbeat (BPM) Analysis from Audio
This Python script analyzes audio recordings (`.wav` files) of heart sounds to detect individual heartbeats, calculate the Beats Per Minute (BPM) over time, and visualize the results. It uses signal processing techniques to filter the audio, identify heartbeat patterns (S1-S2 sounds), and generate a smoothed BPM curve.

## Features
- **Audio Preprocessing:** Loads a `.wav` file, converts it to mono, and applies a band-pass filter to isolate heart sound frequencies.
- **Efficient Processing:** Downsamples the audio to reduce computational load, making analysis faster.
- **Advanced Peak Detection:** Implements a sophisticated peak-finding algorithm that identifies S1-S2 systolic-diastolic patterns, leading to more accurate beat detection than simple amplitude-based methods.
- **BPM Calculation:** Computes instantaneous BPM and provides a smoothed BPM curve over time for a more stable reading.
- **Interactive Visualization:** Generates an interactive HTML plot using Plotly, showing the audio envelope, detected heartbeats, and the calculated BPM trend. The plot includes annotations for maximum, minimum, and average BPM.
- **Data Export:** Saves the calculated time-stamped BPM data to a `.csv` file for further analysis.

## How It Works
The script follows a signal processing pipeline to extract the heart rate:
1. **Load & Filter:** The `.wav` file is loaded, and a **Butterworth band-pass filter** is applied to keep only the frequencies typically associated with heart sounds (S1 and S2).
2. **Envelope Detection:** The script calculates the signal's envelope by taking its absolute value and applying a rolling average. This creates a smooth curve representing the intensity of the heart sounds.
3. **Peak Finding (S1-S2 Analysis):** Instead of just finding any peak, the algorithm specifically looks for the characteristic pattern of a heartbeat:
    - It first identifies all potential peaks in the audio envelope.
    - It then analyzes the time intervals between these peaks. A short interval is classified as the systolic gap between the S1 and S2 sounds. A longer interval is the diastolic gap between beats.
    - By identifying these S1-S2 pairs, the script can more reliably count a single "true" heartbeat event per cycle, avoiding the common issue of double-counting.
4. **BPM Calculation & Smoothing:** The time between each "true" heartbeat is used to calculate the instantaneous BPM. This data is then smoothed using a rolling average to provide a stable heart rate trend.
5. **Output Generation:** The final data is plotted and saved as an interactive HTML file and a CSV data file.

## Requirements
The script requires the following Python libraries:
- `numpy`
- `pandas`
- `scipy`
- `plotly`
You can install them using pip:
```
pip install numpy pandas scipy plotly
```


## Usage
1. **Place your audio file:** Put a single `.wav` file containing the heart sound recording in the same directory as the Python script.
2. **Run the script:** Execute the script from your terminal.
    ```
    python "bpm_analysis v2.3.py"
    ```
3. **Check the output:** The script will automatically find and process the `.wav` file. Once it's finished, you will find two new files in the directory:
    - `[your_file_name]_bpm_plot.html`: An interactive plot.
    - `[your_file_name]_bpm_analysis.csv`: A CSV file with the BPM data.

## Customizing the Analysis
You can fine-tune the analysis by adjusting the parameters in the `main()` function of the script:
- `DOWNSAMPLE_FACTOR`: Increase for faster processing on very large files, decrease for higher time precision.
- `BANDPASS_FREQS`: The frequency range (in Hz) for the band-pass filter. The default `(20, 150)` is standard for heart sounds.
- `MIN_BPM` / `MAX_BPM`: The expected physiological range for the heart rate. This helps filter out impossible peak detections.
- `SMOOTHING_WINDOW_SEC`: The time window (in seconds) for smoothing the final BPM curve. A larger value creates a smoother, more stable line.
- `S1_S2_MAX_INTERVAL_SEC`: The maximum duration (in seconds) between two peaks to be considered an S1-S2 pair. This is a crucial parameter for the peak detection logic.

## Example Output
After running the script, you can open the generated `.html` file in a web browser. The interactive plot will look similar to this:
- **Blue Line:** The audio envelope of the heart sounds.
- **Red Dots:** The precise location of each detected heartbeat.
- **Black Line:** The smoothed BPM trend, plotted on the secondary y-axis.
- **Annotations:** Callouts for the maximum, minimum, and average BPM detected during the recording.
You can hover over the plot to see specific values, and use your mouse to zoom and pan to inspect the data more closely.
