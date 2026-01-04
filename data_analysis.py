import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
import re

## --- NORMOGRAVITY: DECTECTING COMPRESSION PEAKS --- ##

file_path = "raw data/mCPR_ground_data.csv"
mCPR_ground_data = pd.read_csv(file_path)

# Convert pressure from mmHg to Pa (1 mmHg = 133.322 Pa)
mCPR_ground_data['Pressure (Pa)'] = mCPR_ground_data['Pressure (mmHg)'] * 133.322

# Sort data entries by UTC
mCPR_ground_data['DateTime (UTC)'] = pd.to_datetime(mCPR_ground_data['DateTime (UTC)'])
mCPR_ground_data = mCPR_ground_data.sort_values(by='DateTime (UTC)')

mCPR_ground_data.to_csv(file_path, index=False)

# Visualize normogravity pressure over time
"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(mCPR_ground_data['DateTime (UTC)'], mCPR_ground_data['Pressure (Pa)'], label='Pressure (Pa)', color='blue')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure (Pa) Over Time (UTC)')
ax1.grid(True)
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()"""

#Sort data entries by EDT
mCPR_ground_data['DateTime (EDT)'] = pd.to_datetime(mCPR_ground_data['DateTime (EDT)'])
mCPR_ground_data = mCPR_ground_data.sort_values(by='DateTime (EDT)')

# Define 4cm and 5cm normogravity compression windows in EDT
four_cm_indices_ground = (mCPR_ground_data['DateTime (EDT)'] >= pd.to_datetime('2023-08-04 14:35:28')) & \
                          (mCPR_ground_data['DateTime (EDT)'] <= pd.to_datetime('2023-08-04 14:37:11'))
five_cm_indices_ground = (mCPR_ground_data['DateTime (EDT)'] >= pd.to_datetime('2023-08-04 14:37:55')) & \
                          (mCPR_ground_data['DateTime (EDT)'] <= pd.to_datetime('2023-08-04 14:39:38'))

# Detect compression peaks for 4cm and 5cm normogravity compression windows
four_cm_peaks_ground, _ = find_peaks(mCPR_ground_data.loc[four_cm_indices_ground, 'Pressure (mmHg)'], height=40, distance=25)
five_cm_peaks_ground, _ = find_peaks(mCPR_ground_data.loc[five_cm_indices_ground, 'Pressure (mmHg)'], height=40, distance=25)

# Compute average peak amplitude for 4cm and 5cm normogravity compression windows
average_peak_four_cm_ground = mCPR_ground_data.loc[four_cm_indices_ground].iloc[four_cm_peaks_ground]['Pressure (mmHg)'].mean()
average_peak_five_cm_ground = mCPR_ground_data.loc[five_cm_indices_ground].iloc[five_cm_peaks_ground]['Pressure (mmHg)'].mean()
#print(f"Average peak value for 4cm ground compressions: {average_peak_four_cm_ground:.2f} mmHg")
#print(f"Average peak value for 5cm ground compressions: {average_peak_five_cm_ground:.2f} mmHg")
#print(f"Average peak value for 4cm ground compressions: {average_peak_four_cm_ground - pressure_difference_four_cm:.2f} mmHg")
#print(f"Average peak value for 5cm ground compressions: {average_peak_five_cm_ground - pressure_difference_five_cm:.2f} mmHg")

# Define 4cm and 5cm normogravity compression windows for visualization
four_cm_xlim_start_ground = pd.to_datetime('2023-08-04 14:35:28')
four_cm_xlim_end_ground = pd.to_datetime('2023-08-04 14:37:11') 
five_cm_xlim_start_ground = pd.to_datetime('2023-08-04 14:37:55')
five_cm_xlim_end_ground = pd.to_datetime('2023-08-04 14:39:38')  

# Visualize 4cm normogravity compression window
"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(mCPR_ground_data.loc[four_cm_indices_ground, 'DateTime (EDT)'], 
         mCPR_ground_data.loc[four_cm_indices_ground, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax1.scatter(mCPR_ground_data.loc[four_cm_indices_ground].iloc[four_cm_peaks_ground]['DateTime (EDT)'],
            mCPR_ground_data.loc[four_cm_indices_ground].iloc[four_cm_peaks_ground]['Pressure (mmHg)'], 
            color='red', label='Detected Peaks - 4cm mCPR Ground')

ax1.set_xlim([four_cm_xlim_start_ground, four_cm_xlim_end_ground])
ax1.set_xlabel('DateTime (EDT)')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('4cm mCPR Compressions in Normogravity: Pressure (mmHg) Over Time (EDT) with Detected Peaks')
ax1.legend()
ax1.grid(True)"""

# Visualize 5cm normogravity compression window
"""ax2.plot(mCPR_ground_data.loc[five_cm_indices_ground, 'DateTime (EDT)'], 
         mCPR_ground_data.loc[five_cm_indices_ground, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax2.scatter(mCPR_ground_data.loc[five_cm_indices_ground].iloc[five_cm_peaks_ground]['DateTime (EDT)'],
            mCPR_ground_data.loc[five_cm_indices_ground].iloc[five_cm_peaks_ground]['Pressure (mmHg)'], 
            color='green', label='Detected Peaks - 5cm mCPR Ground')

ax2.set_xlim([five_cm_xlim_start_ground, five_cm_xlim_end_ground])
ax2.set_xlabel('DateTime (EDT)')
ax2.set_ylabel('Pressure (mmHg)')
ax2.set_title('5cm mCPR Compressions in Normogravity: Pressure (mmHg) Over Time (EDT) with Detected Peaks')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()"""

# Isolate data points for 4cm and 5cm normogravity compression windows
def mCPR_compression_ground(mCPR_ground_data, mCPR_compression_indices_ground):
    return mCPR_ground_data.loc[mCPR_compression_indices_ground]

mCPR_compression_depth_ground = {'four_cm_ground': [], 'five_cm_ground': []}
mCPR_compression_depth_ground['four_cm_ground'] = mCPR_compression_ground(mCPR_ground_data, four_cm_indices_ground)
mCPR_compression_depth_ground['five_cm_ground'] = mCPR_compression_ground(mCPR_ground_data, five_cm_indices_ground)

four_cm_mask_ground = [four_cm_indices_ground]
five_cm_mask_ground = [five_cm_indices_ground]

## --- NORMOGRAVITY: DECTECTING SYSTOLIC AND DIASTOLIC PEAKS --- ##

def detect_peaks_and_troughs(ground_data, lower_bound, upper_bound):
    # Detect systolic peaks
    systolic_peaks, _ = find_peaks(ground_data['Pressure (mmHg)'], height=40, distance=25)
    systolic_peak_values = ground_data.iloc[systolic_peaks]['Pressure (mmHg)']
    systolic_peak_times = ground_data.iloc[systolic_peaks]['DateTime (UTC)']

    # Detect diastolic troughs
    diastolic_peaks = []
    for i in range(1, len(ground_data) - 1):
        if 15 <= ground_data['Pressure (mmHg)'].iloc[i] <= 25:
            if ground_data['Pressure (mmHg)'].iloc[i] < ground_data['Pressure (mmHg)'].iloc[i - 1] and ground_data['Pressure (mmHg)'].iloc[i] < ground_data['Pressure (mmHg)'].iloc[i + 1]:
                diastolic_peaks.append(i)

    filtered_diastolic_peaks = []
    if len(systolic_peaks) > 0:
        first_systolic_peak = systolic_peaks[0]
        initial_diastolic_troughs = [peak for peak in diastolic_peaks if peak < first_systolic_peak]
        if initial_diastolic_troughs:
            min_initial_trough = min(initial_diastolic_troughs, key=lambda x: ground_data['Pressure (mmHg)'].iloc[x])
            filtered_diastolic_peaks.append(min_initial_trough)

    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        segment_peaks = [peak for peak in diastolic_peaks if start < peak < end]
        if segment_peaks:
            min_peak = min(segment_peaks, key=lambda x: ground_data['Pressure (mmHg)'].iloc[x])
            filtered_diastolic_peaks.append(min_peak)

    diastolic_peak_values = ground_data.iloc[filtered_diastolic_peaks]['Pressure (mmHg)']
    diastolic_peak_times = ground_data.iloc[filtered_diastolic_peaks]['DateTime (UTC)']

    # Detect dicrotic notches
    all_notches, _ = find_peaks(-ground_data['Pressure (mmHg)'], distance=5)
    dicrotic_notches = []
    dicrotic_notch_times = []
    last_systolic_peak = -1

    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        segment_notches = [notch for notch in all_notches if start < notch < end]

        if segment_notches:
            first_notch = segment_notches[0]
            if lower_bound <= ground_data['Pressure (mmHg)'].iloc[first_notch] <= upper_bound:
                dicrotic_notches.append(first_notch)
                dicrotic_notch_times.append(ground_data['DateTime (UTC)'].iloc[first_notch])

    if len(systolic_peaks) > 0:
        last_systolic_peak = systolic_peaks[-1]
        segment_notches = [notch for notch in all_notches if notch > last_systolic_peak]
        if segment_notches:
            first_notch = segment_notches[0]
            if lower_bound <= ground_data['Pressure (mmHg)'].iloc[first_notch] <= upper_bound:
                dicrotic_notches.append(first_notch)
                dicrotic_notch_times.append(ground_data['DateTime (UTC)'].iloc[first_notch])

    return systolic_peaks, systolic_peak_values, systolic_peak_times, filtered_diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times

def create_ground_analysis_df(analysis):
    return pd.DataFrame({
        'average_systolic_pressure (mmHg)': [analysis['average_systolic_pressure (mmHg)']],
        'average_diastolic_pressure (mmHg)': [analysis['average_diastolic_pressure (mmHg)']],
    })

def analyze_compression_ground(ground_data, lower_bound, upper_bound, compression_depth_ground):
    systolic_peaks, systolic_peak_values, systolic_peak_times, diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times = detect_peaks_and_troughs(ground_data, lower_bound, upper_bound)

    if compression_depth_ground == 'four_cm_ground':
        if len(diastolic_peaks) > 0:
            pass  
        if len(dicrotic_notches) > 0:
            pass  

    if compression_depth_ground == 'five_cm_ground':
        if len(diastolic_peaks) > 2:
            pass
        if len(dicrotic_notches) > 1:
            pass
        if len(diastolic_peaks) > 0:
            diastolic_peaks = diastolic_peaks[:-1]
            diastolic_peak_values = diastolic_peak_values.iloc[:-1]
            diastolic_peak_times = diastolic_peak_times.iloc[:-1]
        if len(systolic_peaks) > 0:
            systolic_peaks = systolic_peaks[:-1]
            systolic_peak_values = systolic_peak_values.iloc[:-1]
            systolic_peak_times = systolic_peak_times.iloc[:-1]
        if len(dicrotic_notches) > 0:
            dicrotic_notches = dicrotic_notches[:-1]
            dicrotic_notch_times = dicrotic_notch_times[:-1]
            
    ground_analysis = {
        'average_systolic_pressure (mmHg)': systolic_peak_values.mean(),
        'average_diastolic_pressure (mmHg)': diastolic_peak_values.mean(),
        'systolic_peaks (mmHg)': systolic_peak_values,
        'systolic_peak_times': systolic_peak_times,
        'diastolic_troughs (mmHg)': diastolic_peak_values,
        'diastolic_trough_times': diastolic_peak_times,
        'dicrotic_notches (mmHg)': ground_data['Pressure (mmHg)'].iloc[dicrotic_notches],
        'dicrotic_notch_times': dicrotic_notch_times,
        'systolic_peak_indices': systolic_peaks,
        'diastolic_trough_indices': diastolic_peaks,
        'dicrotic_notch_indices': dicrotic_notches,
    }

    ground_analysis_df = create_ground_analysis_df(ground_analysis)
    systolic_peaks_df = pd.DataFrame({'systolic_peaks (mmHg)': ground_analysis['systolic_peaks (mmHg)'].tolist(), 'systolic_peak_times': ground_analysis['systolic_peak_times'].tolist()})
    diastolic_troughs_df = pd.DataFrame({'diastolic_troughs (mmHg)': ground_analysis['diastolic_troughs (mmHg)'].tolist(), 'diastolic_trough_times': ground_analysis['diastolic_trough_times'].tolist()})
    dicrotic_notches_df = pd.DataFrame({'dicrotic_notches (mmHg)': ground_analysis['dicrotic_notches (mmHg)'].tolist(), 'dicrotic_notch_times': ground_analysis['dicrotic_notch_times']})
    ground_analysis_df = pd.concat([ground_analysis_df, systolic_peaks_df, diastolic_troughs_df, dicrotic_notches_df], axis=1)

    # Compute MAP for each compression cycle
    ground_analysis_df['MAP (mmHg)'] = (2/3) * ground_analysis_df['diastolic_troughs (mmHg)'].shift(-1) + (1/3) * ground_analysis_df['systolic_peaks (mmHg)']
    average_MAP = ground_analysis_df['MAP (mmHg)'].mean()
    ground_analysis_df['average_MAP (mmHg)'] = [average_MAP] + [None] * (len(ground_analysis_df) - 1)

    # Compute systole duration (dicrotic notch to diastolic trough)
    ground_analysis_df['systole_duration (s)'] = (pd.to_datetime(ground_analysis_df['diastolic_trough_times']) - pd.to_datetime(ground_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_systole_duration = ground_analysis_df['systole_duration (s)'].mean()
    ground_analysis_df['average_systole_duration (s)'] = [average_systole_duration] + [None] * (len(ground_analysis_df) - 1)

    # Compute diastole duration (dicrotic notch to diastolic trough)
    ground_analysis_df['diastole_duration (s)'] = (pd.to_datetime(ground_analysis_df['diastolic_trough_times'].shift(-1)) - pd.to_datetime(ground_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_diastole_duration = ground_analysis_df['diastole_duration (s)'].mean()
    ground_analysis_df['average_diastole_duration (s)'] = [average_diastole_duration] + [None] * (len(ground_analysis_df) - 1)

    # Compute compression rate (systolic-to-systolic peak interval)
    ground_analysis_df['compression_rate'] = (pd.to_datetime(ground_analysis_df['systolic_peak_times'].shift(-1)) - pd.to_datetime(ground_analysis_df['systolic_peak_times'])).abs().dt.total_seconds()
    average_time_interval = ground_analysis_df['compression_rate'].mean()
    average_compression_rate = (1 / average_time_interval) * 60 if average_time_interval != 0 else None
    ground_analysis_df['average_compression_rate (compressions/min)'] = [average_compression_rate] + [None] * (len(ground_analysis_df) - 1)

    # Compute pulse pressure for each compression cycle
    ground_analysis_df['pulse_pressure (mmHg)'] = ground_analysis_df['systolic_peaks (mmHg)'] - ground_analysis_df['diastolic_troughs (mmHg)']
    average_pulse_pressure = ground_analysis_df['pulse_pressure (mmHg)'].mean()
    ground_analysis_df['average_pulse_pressure (mmHg)'] = [average_pulse_pressure] + [None] * (len(ground_analysis_df) - 1)

    ground_analysis_df.to_csv(f'processed data/mCPR compressions normogravity/{compression_depth_ground}_analysis.csv', index=False)
    #print(f"{compression_depth_ground.capitalize()} Analysis:")
    #print(ground_analysis_df)

    return ground_analysis

# Visualize systolic, diastolic, and dicrotic features in sliding 5s windows
"""def plot_ground_results(ground_data, analysis, compression_depth_ground):
    start_time = ground_data['DateTime (UTC)'].min()
    end_time = ground_data['DateTime (UTC)'].max()
    interval = pd.Timedelta(seconds=5)
    
    current_time = start_time
    while current_time < end_time:
        window_end = current_time + interval
        plt.figure(figsize=(12, 6))
        plt.plot(ground_data['DateTime (UTC)'], ground_data['Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')
        plt.scatter(ground_data.iloc[analysis['systolic_peak_indices']]['DateTime (UTC)'], analysis['systolic_peaks (mmHg)'], color='red', label='Systolic Peaks')
        plt.scatter(ground_data.iloc[analysis['diastolic_trough_indices']]['DateTime (UTC)'], analysis['diastolic_troughs (mmHg)'], color='green', label='Diastolic Troughs')
        plt.scatter(ground_data.iloc[analysis['dicrotic_notch_indices']]['DateTime (UTC)'], analysis['dicrotic_notches (mmHg)'], color='orange', label='Dicrotic Notches')
        plt.xlabel('DateTime (UTC)')
        plt.ylabel('Pressure (mmHg)')
        plt.title(f'{compression_depth_ground.capitalize()}: Pressure (mmHg) Over Time (UTC) with Systolic Peaks, Diastolic Troughs, and Dicrotic Notches')
        plt.legend()
        plt.grid(True)
        plt.xlim(current_time, window_end)
        plt.show()
        
        current_time = window_end"""

# Visualize 4cm and 5cm normogravity compression windows
four_cm_analysis_ground = analyze_compression_ground(mCPR_compression_depth_ground['four_cm_ground'], lower_bound=30, upper_bound=50, compression_depth_ground='four_cm_ground')
#plot_ground_results(mCPR_compression_depth_ground['four_cm_ground'], four_cm_analysis_ground, compression_depth_ground='four_cm_ground')
five_cm_analysis_ground = analyze_compression_ground(mCPR_compression_depth_ground['five_cm_ground'], lower_bound=30, upper_bound=50, compression_depth_ground='five_cm_ground')
#plot_ground_results(mCPR_compression_depth_ground['five_cm_ground'], five_cm_analysis_ground, compression_depth_ground='five_cm_ground')



## --- HYPOGRAVITY: DECTECTING COMPRESSION PEAKS --- ##

file_path = r"raw data/mCPR_flight_data.csv"
mCPR_flight_data = pd.read_csv(file_path)

# Convert pressure from mmHg to Pa (1 mmHg = 133.322 Pa)
mCPR_flight_data['Pressure (Pa)'] = mCPR_flight_data['Pressure (mmHg)'] * 133.322

mCPR_flight_data.to_csv(file_path, index=False)

# Sort data entries by UTC
mCPR_flight_data['DateTime (UTC)'] = pd.to_datetime(mCPR_flight_data['DateTime (UTC)'])

# Visualize hypogravity pressure over time with g-force overlay
"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(mCPR_flight_data['DateTime (UTC)'], mCPR_flight_data['Pressure (Pa)'], label='Pressure (Pa)', color='blue')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure (Pa) Over Time (UTC)')
ax1.grid(True)
ax1_gforce = ax1.twinx()
ax1_gforce.plot(mCPR_flight_data['DateTime (UTC)'], mCPR_flight_data['G-Force (G)'], label='G-Force (G)', color='red', alpha=0.7)
ax1_gforce.set_ylabel('G-Force (G)')
ax1.legend(loc='upper left')
ax1_gforce.legend(loc='upper right')

plt.tight_layout()
plt.show()"""

# Detect 4cm and 5cm hypogravity compression windows during parabolic flight
def identify_hypogravity_parabolas(pressure_values, gforce_values, time_values, threshold_gforce=0.10, min_points=5, max_duration_seconds=30):
    hypogravity_parabolas = {'four_cm_hypogravity': [], 'five_cm_hypogravity': []}
    current_hypogravity_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
    is_within_hypogravity_parabola = False
    hypogravity_parabola_count = 0
    start_time = None
    
    for i in range(len(pressure_values)):
        if gforce_values[i] < threshold_gforce:
            if not is_within_hypogravity_parabola:
                if len(current_hypogravity_parabola['pressure']) > min_points:
                    current_hypogravity_parabola['number'] = hypogravity_parabola_count + 1
                    if (hypogravity_parabola_count + 1) % 2 == 0: # Even-numbered hypogravity parabolas (2, 4, 6, 8, 10) with 4cm hypogravity compression windows 
                        hypogravity_parabolas['four_cm_hypogravity'].append(current_hypogravity_parabola)
                    else: # Odd-numbered hypogravity parabolas (1, 3, 5, 7, 9) with 5cm hypogravity compression windows 
                        hypogravity_parabolas['five_cm_hypogravity'].append(current_hypogravity_parabola)

                current_hypogravity_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
                is_within_hypogravity_parabola = True
                hypogravity_parabola_count += 1
                start_time = time_values[i]
            
            current_hypogravity_parabola['pressure'].append(pressure_values[i])
            current_hypogravity_parabola['gforce'].append(gforce_values[i])
            current_hypogravity_parabola['indices'].append(i)
        
        else:
            if is_within_hypogravity_parabola:
                time_diff = (time_values[i] - start_time).astype('timedelta64[s]').item().total_seconds()
                if time_diff > max_duration_seconds:
                    is_within_hypogravity_parabola = False

    if len(current_hypogravity_parabola['pressure']) > min_points:
        current_hypogravity_parabola['number'] = hypogravity_parabola_count + 1
        if (hypogravity_parabola_count + 1) % 2 == 0: # Even-numbered hypogravity parabolas (2, 4, 6, 8, 10) with 4cm hypogravity compression windows 
            hypogravity_parabolas['four_cm_hypogravity'].append(current_hypogravity_parabola)
        else: # Odd-numbered hypogravity parabolas (1, 3, 5, 7, 9) with 5cm hypogravity compression windows 
            hypogravity_parabolas['five_cm_hypogravity'].append(current_hypogravity_parabola)

    return hypogravity_parabolas

def renumber_hypogravity_parabolas(hypogravity_parabolas):
    for i, hypogravity_parabola in enumerate(hypogravity_parabolas['four_cm_hypogravity']):
        hypogravity_parabola['number'] = 2 * i + 1 # Odd-numbered hypogravity parabolas (1, 3, 5, 7, 9) with 5cm hypogravity compression windows 
    for i, hypogravity_parabola in enumerate(hypogravity_parabolas['five_cm_hypogravity']):
        hypogravity_parabola['number'] = 2 * (i + 1) # Even-numbered hypogravity parabolas (2, 4, 6, 8, 10) with 4cm hypogravity compression windows 
    return hypogravity_parabolas

pressure_values = mCPR_flight_data['Pressure (Pa)'].values
gforce_values = mCPR_flight_data['G-Force (G)'].values
time_values = pd.to_datetime(mCPR_flight_data['DateTime (UTC)']).values

hypogravity_parabolas_dict = identify_hypogravity_parabolas(pressure_values, gforce_values, time_values)
hypogravity_parabolas_dict = renumber_hypogravity_parabolas(hypogravity_parabolas_dict)

four_cm_hypogravity_parabolas = hypogravity_parabolas_dict['four_cm_hypogravity']
five_cm_hypogravity_parabolas = hypogravity_parabolas_dict['five_cm_hypogravity']

#print(f"Number of hypogravity parabolas with 4cm mCPR compressions: {len(four_cm_hypogravity_parabolas)}")
#print(f"Number of hypogravity parabolas with 5cm mCPR compressions: {len(five_cm_hypogravity_parabolas)}")

# Isolate data points for 4cm and 5cm hypogravity compression windows
def mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola):
    start_index = hypogravity_parabola['indices'][0]
    end_index = hypogravity_parabola['indices'][-1]
    return mCPR_flight_data.iloc[start_index:end_index + 1]

mCPR_compression_depth_hypogravity = {'four_cm_hypogravity': [], 'five_cm_hypogravity': []}

for hypogravity_parabola in four_cm_hypogravity_parabolas:
    hypogravity_parabola_df = mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola)
    mCPR_compression_depth_hypogravity['four_cm_hypogravity'].append(hypogravity_parabola_df)

for hypogravity_parabola in five_cm_hypogravity_parabolas:
    hypogravity_parabola_df = mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola)
    mCPR_compression_depth_hypogravity['five_cm_hypogravity'].append(hypogravity_parabola_df)

# Visualize 4cm and 5cm hypogravity compression windows
"""plt.figure(figsize=(12, 8))
plt.plot(time_values, pressure_values, label='Pressure (Pa)', color='lightgray', alpha=0.5)

for hypogravity_parabola in four_cm_hypogravity_parabolas:
    plt.plot(time_values[hypogravity_parabola['indices']], hypogravity_parabola['pressure'], color='blue', label='4 cm mCPR Compression Window' if hypogravity_parabola == four_cm_hypogravity_parabolas[0] else "")
    mid_index = len(hypogravity_parabola['indices']) // 2
    plt.text(time_values[hypogravity_parabola['indices'][mid_index]], hypogravity_parabola['pressure'][mid_index], f"{hypogravity_parabola['number']}", color='blue')

for hypogravity_parabola in five_cm_hypogravity_parabolas:
    plt.plot(time_values[hypogravity_parabola['indices']], hypogravity_parabola['pressure'], color='green', label='5 cm mCPR Compression Window' if hypogravity_parabola == five_cm_hypogravity_parabolas[0] else "")
    mid_index = len(hypogravity_parabola['indices']) // 2
    plt.text(time_values[hypogravity_parabola['indices'][mid_index]], hypogravity_parabola['pressure'][mid_index], f"{hypogravity_parabola['number']}", color='green')

plt.xlabel('Time (UTC)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure (Pa) Over Time (UTC) with Color-Coded Hypogravity Compression Windows Across Parabolas')
plt.legend()
plt.grid(True)

plt.show()"""

# Print full time-series for each 4cm and 5cm hypogravity compression window
"""print("\nAll parabolas with 4cm compression compression windows:")
for hypogravity_parabola in hypogravity_parabolas_dict['four_cm_hypogravity']:
    print(f"Hypogravity phase in parabola {hypogravity_parabola['number']}:")
    print(mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola))
    print()
    
print("\nAll parabolas with 5cm hypogravity compression windows:")
for hypogravity_parabola in five_cm_hypogravity_parabolas:
    print(f"Hypogravity phase in parabola {hypogravity_parabola['number']}:")
    print(mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola))
    print()"""
    
## --- HYPOGRAVITY: DECTECTING SYSTOLIC AND DIASTOLIC PEAKS --- ##

def detect_peaks_and_troughs(hypogravity_data, lower_bound, upper_bound):
    # Detect systolic peaks
    systolic_peaks, _ = find_peaks(hypogravity_data['Pressure (mmHg)'], height=40, distance=25)
    systolic_peak_values = hypogravity_data.iloc[systolic_peaks]['Pressure (mmHg)']
    systolic_peak_times = hypogravity_data.iloc[systolic_peaks]['DateTime (UTC)']

    # Detect diastolic troughs
    diastolic_peaks_50, _ = find_peaks(-hypogravity_data['Pressure (mmHg)'], height=-30, distance=20)
    diastolic_peaks_70, _ = find_peaks(-hypogravity_data['Pressure (mmHg)'], height=-30, distance=80)
    combined_diastolic_peaks = np.unique(np.concatenate((diastolic_peaks_50, diastolic_peaks_70)))

    filtered_diastolic_peaks = [peak for peak in combined_diastolic_peaks if hypogravity_data['Pressure (mmHg)'].iloc[peak] <= 30]

    first_systolic_peak = systolic_peaks[0]
    troughs_before_first_systolic = [peak for peak in filtered_diastolic_peaks if peak < first_systolic_peak]
    if troughs_before_first_systolic:
        first_diastolic_trough = min(troughs_before_first_systolic, key=lambda x: hypogravity_data['Pressure (mmHg)'].iloc[x])
    else:
        first_diastolic_trough = None

    # Detect dicrotic notches
    all_notches, _ = find_peaks(-hypogravity_data['Pressure (mmHg)'], distance=5)
    dicrotic_notches = []
    dicrotic_notch_times = []

    for peak in systolic_peaks:
        for notch in all_notches:
            if notch > peak and lower_bound <= hypogravity_data['Pressure (mmHg)'].iloc[notch] <= upper_bound:
                dicrotic_notches.append(notch)
                dicrotic_notch_times.append(hypogravity_data['DateTime (UTC)'].iloc[notch])
                break 
    
    final_diastolic_peaks = [first_diastolic_trough] if first_diastolic_trough is not None else []
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        troughs_in_range = [peak for peak in filtered_diastolic_peaks if start < peak < end]
        if troughs_in_range:
            lowest_trough = min(troughs_in_range, key=lambda x: hypogravity_data['Pressure (mmHg)'].iloc[x])
            final_diastolic_peaks.append(lowest_trough)

    diastolic_peak_values = hypogravity_data.iloc[final_diastolic_peaks]['Pressure (mmHg)']
    diastolic_peak_times = hypogravity_data.iloc[final_diastolic_peaks]['DateTime (UTC)']

    return systolic_peaks, systolic_peak_values, systolic_peak_times, final_diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times

def create_hypogravity_analysis_df(analysis):
    return pd.DataFrame({
        'average_systolic_pressure (mmHg)': [analysis['average_systolic_pressure (mmHg)']],
        'average_diastolic_pressure (mmHg)': [analysis['average_diastolic_pressure (mmHg)']],
    })

def analyze_compression_hypogravity(hypogravity_data, lower_bound, upper_bound, hypogravity_parabola_index):
    systolic_peaks, systolic_peak_values, systolic_peak_times, diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times = detect_peaks_and_troughs(hypogravity_data, lower_bound, upper_bound)

    if hypogravity_parabola_index in ['five_cm_hypogravity_parabola_6', 'four_cm_hypogravity_parabola_5']:
        if len(systolic_peaks) > 0:
            systolic_peaks = systolic_peaks[1:]
            systolic_peak_values = systolic_peak_values[1:]
            systolic_peak_times = systolic_peak_times[1:]
        if len(dicrotic_notches) > 0:
            dicrotic_notches = dicrotic_notches[1:]
            dicrotic_notch_times = dicrotic_notch_times[1:]

    hypogravity_analysis = {
        'average_systolic_pressure (mmHg)': systolic_peak_values.mean(),
        'average_diastolic_pressure (mmHg)': diastolic_peak_values.mean(),
        'systolic_peaks (mmHg)': systolic_peak_values,
        'systolic_peak_times': systolic_peak_times,
        'diastolic_troughs (mmHg)': diastolic_peak_values,
        'diastolic_trough_times': diastolic_peak_times,
        'dicrotic_notches (mmHg)': hypogravity_data['Pressure (mmHg)'].iloc[dicrotic_notches],
        'dicrotic_notch_times': dicrotic_notch_times,
        'systolic_peak_indices': systolic_peaks,
        'diastolic_trough_indices': diastolic_peaks,
        'dicrotic_notch_indices': dicrotic_notches
    }

    hypogravity_analysis_df = create_hypogravity_analysis_df(hypogravity_analysis)
    systolic_peaks_df = pd.DataFrame({'systolic_peaks (mmHg)': hypogravity_analysis['systolic_peaks (mmHg)'].tolist(), 'systolic_peak_times': hypogravity_analysis['systolic_peak_times'].tolist()})
    diastolic_troughs_df = pd.DataFrame({'diastolic_troughs (mmHg)': hypogravity_analysis['diastolic_troughs (mmHg)'].tolist(), 'diastolic_trough_times': hypogravity_analysis['diastolic_trough_times'].tolist()})
    dicrotic_notches_df = pd.DataFrame({'dicrotic_notches (mmHg)': hypogravity_analysis['dicrotic_notches (mmHg)'].tolist(), 'dicrotic_notch_times': hypogravity_analysis['dicrotic_notch_times']})
    hypogravity_analysis_df = pd.concat([hypogravity_analysis_df, systolic_peaks_df, diastolic_troughs_df, dicrotic_notches_df], axis=1)

    # Compute MAP
    hypogravity_analysis_df['MAP (mmHg)'] = (2/3) * hypogravity_analysis_df['diastolic_troughs (mmHg)'].shift(-1) + (1/3) * hypogravity_analysis_df['systolic_peaks (mmHg)']
    average_MAP = hypogravity_analysis_df['MAP (mmHg)'].mean()
    hypogravity_analysis_df['average_MAP (mmHg)'] = [average_MAP] + [None] * (len(hypogravity_analysis_df) - 1)

    # Compute systole duration (dicrotic notch to diastolic trough)
    hypogravity_analysis_df['systole_duration (s)'] = (pd.to_datetime(hypogravity_analysis_df['diastolic_trough_times']) - pd.to_datetime(hypogravity_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_systole_duration = hypogravity_analysis_df['systole_duration (s)'].mean()
    hypogravity_analysis_df['average_systole_duration (s)'] = [average_systole_duration] + [None] * (len(hypogravity_analysis_df) - 1)

    # Compute diastole duration (dicrotic notch to diastolic trough)
    hypogravity_analysis_df['diastole_duration (s)'] = (pd.to_datetime(hypogravity_analysis_df['diastolic_trough_times'].shift(-1)) - pd.to_datetime(hypogravity_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_diastole_duration = hypogravity_analysis_df['diastole_duration (s)'].mean()
    hypogravity_analysis_df['average_diastole_duration (s)'] = [average_diastole_duration] + [None] * (len(hypogravity_analysis_df) - 1)

    # Compute compression rate (systolic-to-systolic peak interval)
    hypogravity_analysis_df['compression_rate'] = (pd.to_datetime(hypogravity_analysis_df['systolic_peak_times'].shift(-1)) - pd.to_datetime(hypogravity_analysis_df['systolic_peak_times'])).abs().dt.total_seconds()
    average_time_interval = hypogravity_analysis_df['compression_rate'].mean()
    average_compression_rate = (1 / average_time_interval) * 60 if average_time_interval != 0 else None
    hypogravity_analysis_df['average_compression_rate (compressions/min)'] = [average_compression_rate] + [None] * (len(hypogravity_analysis_df) - 1)

    # Compute pulse pressure for each compression cycle
    hypogravity_analysis_df['pulse_pressure (mmHg)'] = hypogravity_analysis_df['systolic_peaks (mmHg)'] - hypogravity_analysis_df['diastolic_troughs (mmHg)']
    average_pulse_pressure = hypogravity_analysis_df['pulse_pressure (mmHg)'].mean()
    hypogravity_analysis_df['average_pulse_pressure (mmHg)'] = [average_pulse_pressure] + [None] * (len(hypogravity_analysis_df) - 1)

    hypogravity_analysis_df.to_csv(f'processed data/mCPR compressions hypogravity/{hypogravity_parabola_index}_analysis.csv', index=False)

    return hypogravity_analysis

# Visualize systolic, diastolic, and dicrotic features in sliding 5s windows
"""def plot_hypogravity_results(hypogravity_data, hypogravity_analysis, hypogravity_parabola_index):
    plt.figure(figsize=(16, 10))
    plt.plot(hypogravity_data['DateTime (UTC)'], hypogravity_data['Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')
    plt.scatter(hypogravity_data.iloc[hypogravity_analysis['systolic_peak_indices']]['DateTime (UTC)'], hypogravity_analysis['systolic_peaks (mmHg)'], color='red', label='Systolic Peaks')
    plt.scatter(hypogravity_data.iloc[hypogravity_analysis['diastolic_trough_indices']]['DateTime (UTC)'], hypogravity_analysis['diastolic_troughs (mmHg)'], color='green', label='Diastolic Troughs')
    plt.scatter(hypogravity_data.iloc[hypogravity_analysis['dicrotic_notch_indices']]['DateTime (UTC)'], hypogravity_analysis['dicrotic_notches (mmHg)'], color='orange', label='Dicrotic Notches')
    plt.xlabel('DateTime (UTC)')
    plt.ylabel('Pressure (mmHg)')
    plt.title(f'{hypogravity_parabola_index.capitalize()}: Pressure (mmHg) Over Time (UTC) with Systolic Peaks, Diastolic Troughs, and Dicrotic Notches')
    plt.legend()
    plt.grid(True)
    plt.show()"""

lower_bound = 30
upper_bound = 55

# Visualize 4cm and 5cm hypogravity compression windows
for i, hypogravity_parabola in enumerate(four_cm_hypogravity_parabolas):
    hypogravity_data = mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola)
    hypogravity_parabola_index = f'four_cm_hypogravity_parabola_{i+1}'
    #print(f"Analyzing hypogravity parabola index: {i}, four_cm_hypogravity_parabola_{i+1}")
    four_cm_analysis_hypogravity = analyze_compression_hypogravity(hypogravity_data, lower_bound, upper_bound, hypogravity_parabola_index=hypogravity_parabola_index)
    #plot_hypogravity_results(hypogravity_data, four_cm_analysis_hypogravity, hypogravity_parabola_index=hypogravity_parabola_index)
    
for i, hypogravity_parabola in enumerate(five_cm_hypogravity_parabolas):
    hypogravity_data = mCPR_compression_hypogravity(mCPR_flight_data, hypogravity_parabola)
    #print(f"Analyzing hypogravity parabola index: {i}, five_cm_hypogravity_parabola_{i+6}")
    five_cm_analysis_hypogravity = analyze_compression_hypogravity(hypogravity_data, lower_bound, upper_bound, hypogravity_parabola_index=f'five_cm_hypogravity_parabola_{i+6}')
    #plot_hypogravity_results(hypogravity_data, five_cm_analysis_hypogravity, hypogravity_parabola_index=f'five_cm_hypogravity_parabola_{i+6}')
    
    
    
# --- GENERATE CSV FILES FOR ANALYSIS --- #

def flight_analysis(file_patterns, num_files_list, output_file, compression_depth_cm):
    flight_analysis_df = pd.DataFrame()

    def extract_hypogravity_parabola(file_path):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        match = re.search(r'parabola_(\d+)', stem)
        return int(match.group(1)) if match else None

    for pattern, num_files in zip(file_patterns, num_files_list):
        for i in range(1, num_files + 1):
            
            if 'five_cm_hypogravity_parabola' in pattern:
                file_path = pattern.format(i=i+5)
            else:
                file_path = pattern.format(i=i)

            if os.path.exists(file_path):
                #print(f"Reading file: {file_path}")
                hypogravity_parabola_index_df = pd.read_csv(file_path)

                hypogravity_parabola_num = extract_hypogravity_parabola(file_path)
                hypogravity_parabola_index_df['hypogravity_parabola'] = hypogravity_parabola_num

                # Compute MAP for each compression cycle
                hypogravity_parabola_index_df['MAP (mmHg)'] = (
                    (2/3) * pd.to_numeric(hypogravity_parabola_index_df['diastolic_troughs (mmHg)']).shift(-1)
                    + (1/3) * pd.to_numeric(hypogravity_parabola_index_df['systolic_peaks (mmHg)'])
                )

                flight_analysis_df = pd.concat([flight_analysis_df, hypogravity_parabola_index_df], ignore_index=True)

            else:
                pass
                #print(f"File not found: {file_path}")

    # Compute averages for each hypogravity parabola
    if 'hypogravity_parabola' in flight_analysis_df.columns:
        for hypogravity_parabola_num in flight_analysis_df['hypogravity_parabola'].dropna().unique():
            hypogravity_df = flight_analysis_df[flight_analysis_df['hypogravity_parabola'] == hypogravity_parabola_num]

            averages = {
                'average_systolic_pressure (mmHg)': hypogravity_df['systolic_peaks (mmHg)'].mean(),
                'average_diastolic_pressure (mmHg)': hypogravity_df['diastolic_troughs (mmHg)'].mean(),
                'average_MAP (mmHg)': hypogravity_df['MAP (mmHg)'].mean(),
                'average_systole_duration (s)': hypogravity_df['systole_duration (s)'].mean(),
                'average_diastole_duration (s)': hypogravity_df['diastole_duration (s)'].mean(),
                'average_compression_rate (compressions/min)': (lambda avg_interval: (60 / avg_interval) if pd.notna(avg_interval) and avg_interval != 0 else None)(hypogravity_df['compression_rate'].mean()),
                'average_pulse_pressure (mmHg)': hypogravity_df['pulse_pressure (mmHg)'].mean()
            }

            first_row_idx = flight_analysis_df[flight_analysis_df['hypogravity_parabola'] == hypogravity_parabola_num].index[0]
            for col, avg_val in averages.items():
                flight_analysis_df.at[first_row_idx, col] = avg_val

    # Export per hypogravity parabola CSV file
    flight_analysis_df.to_csv(output_file, index=False)
    #print(f"Data saved to {output_file}")  

flight_analysis(
    ['processed data/mCPR compressions hypogravity/four_cm_hypogravity_parabola_{i}_analysis.csv'],
    [5],
    'processed data/mCPR compressions hypogravity/four_cm_hypogravity_analysis.csv',
    compression_depth_cm=4
)

flight_analysis(
    ['processed data/mCPR compressions hypogravity/five_cm_hypogravity_parabola_{i}_analysis.csv'],
    [5],
    'processed data/mCPR compressions hypogravity/five_cm_hypogravity_analysis.csv',
    compression_depth_cm=5
)

# Export hypogravity and normogravity analysis into master CSV file
four_cm_flight_analysis_df = pd.read_csv('processed data/mCPR compressions hypogravity/four_cm_hypogravity_analysis.csv')
five_cm_flight_analysis_df = pd.read_csv('processed data/mCPR compressions hypogravity/five_cm_hypogravity_analysis.csv')

four_cm_flight_analysis_df['gravitational_condition'] = 'hypogravity'
four_cm_flight_analysis_df['compression_depth (cm)'] = 4

five_cm_flight_analysis_df['gravitational_condition'] = 'hypogravity'
five_cm_flight_analysis_df['compression_depth (cm)'] = 5

four_cm_ground_analysis_df = pd.read_csv('processed data/mCPR compressions normogravity/four_cm_ground_analysis.csv')
five_cm_ground_analysis_df = pd.read_csv('processed data/mCPR compressions normogravity/five_cm_ground_analysis.csv')
four_cm_ground_analysis_df['gravitational_condition'] = 'normogravity'
four_cm_ground_analysis_df['compression_depth (cm)'] = 4
four_cm_ground_analysis_df['hypogravity_parabola'] = pd.NA
five_cm_ground_analysis_df['gravitational_condition'] = 'normogravity'
five_cm_ground_analysis_df['compression_depth (cm)'] = 5
five_cm_ground_analysis_df['hypogravity_parabola'] = pd.NA

merged_df = pd.concat(
    [five_cm_flight_analysis_df, five_cm_ground_analysis_df, four_cm_flight_analysis_df, four_cm_ground_analysis_df],
    ignore_index=True
)

merged_df = merged_df.sort_values(
    by=['compression_depth (cm)', 'gravitational_condition', 'hypogravity_parabola'],
    na_position='last',
    kind='mergesort'
)

cols_right = ['gravitational_condition', 'compression_depth (cm)', 'hypogravity_parabola']
merged_df = merged_df[[c for c in merged_df.columns if c not in cols_right] + cols_right]
merged_df.to_csv('processed data/mCPR_compressions_analysis.csv', index=False)
#print("Data saved to mCPR_compressions_analysis.csv")