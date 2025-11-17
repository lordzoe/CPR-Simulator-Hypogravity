import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os

## --- NORMOGRAVITY: DECTECTING COMPRESSION PEAKS --- ##

file_path = "data/Query3.csv"
Query3 = pd.read_csv(file_path)

# Convert pressure from mmHg to Pa (1 mmHg = 133.322 Pa)
Query3['Pressure (Pa)'] = Query3['Pressure (mmHg)'] * 133.322

# Sort samples by UTC time for consistent time-series processing
Query3['DateTimeUTC'] = pd.to_datetime(Query3['DateTimeUTC'])
Query3 = Query3.sort_values(by='DateTimeUTC')

Query3.to_csv(file_path, index=False)

# Create a figure for visual inspection of normogravity pressure over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(Query3['DateTimeUTC'], Query3['Pressure (Pa)'], label='Pressure (Pa)', color='blue')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure Over Time')
ax1.grid(True)
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()

#Sort samples by EDT time for defining normogravity compression windows
Query3['DateTimeEDT'] = pd.to_datetime(Query3['DateTimeEDT'])
Query3 = Query3.sort_values(by='DateTimeEDT')

# (Optional) Compute baseline average pressure up to onset of first normogravity compression window
#start_time_query3 = Query3['DateTimeEDT'].iloc[0]  # First time point in the dataset
#end_time_query3 = pd.to_datetime('2023-08-04 14:35:28')  # Adjust this as needed
#filtered_data_query3 = Query3[(Query3['DateTimeEDT'] >= start_time_query3) & (Query3['DateTimeEDT'] <= end_time_query3)]
#average_pressure_query3 = filtered_data_query3['Pressure (mmHg)'].mean()
#print(f"Average pressure value from onset until {end_time_query3}: {average_pressure_query3:.2f} mmHg")

# Define first and second normogravity compression windows in EDT
cluster1_indices_query3 = (Query3['DateTimeEDT'] >= pd.to_datetime('2023-08-04 14:35:28')) & \
                          (Query3['DateTimeEDT'] <= pd.to_datetime('2023-08-04 14:37:11'))
cluster2_indices_query3 = (Query3['DateTimeEDT'] >= pd.to_datetime('2023-08-04 14:37:55')) & \
                          (Query3['DateTimeEDT'] <= pd.to_datetime('2023-08-04 14:39:38'))

# Detect compression peaks within each normogravity compression window
cluster1_peaks_query3, _ = find_peaks(Query3.loc[cluster1_indices_query3, 'Pressure (mmHg)'], height=40, distance=25)
cluster2_peaks_query3, _ = find_peaks(Query3.loc[cluster2_indices_query3, 'Pressure (mmHg)'], height=40, distance=25)

# Compute average peak amplitude within each normogravity compression window
average_peak_cluster1_query3 = Query3.loc[cluster1_indices_query3].iloc[cluster1_peaks_query3]['Pressure (mmHg)'].mean()
average_peak_cluster2_query3 = Query3.loc[cluster2_indices_query3].iloc[cluster2_peaks_query3]['Pressure (mmHg)'].mean()

print(f"Average peak value for the first cluster of spikes: {average_peak_cluster1_query3:.2f} mmHg")
print(f"Average peak value for the second cluster of spikes: {average_peak_cluster2_query3:.2f} mmHg")

#print(f"Average peak value for the first cluster of spikes: {average_peak_cluster1_query3 - pressure_difference_cluster1:.2f} mmHg")
#print(f"Average peak value for the second cluster of spikes: {average_peak_cluster2_query3 - pressure_difference_cluster2:.2f} mmHg")

# Define x-axis limits for plotting each normogravity compression window
cluster1_xlim_start_query3 = pd.to_datetime('2023-08-04 14:35:28')
cluster1_xlim_end_query3 = pd.to_datetime('2023-08-04 14:37:11')  # Set x-axis limit for Cluster 1
cluster2_xlim_start_query3 = pd.to_datetime('2023-08-04 14:37:55')
cluster2_xlim_end_query3 = pd.to_datetime('2023-08-04 14:39:38')  # Set x-axis limit for Cluster 2

# Create subplots for visualizing normogravity compression windows with detected peaks
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot first normogravity compression window 
ax1.plot(Query3.loc[cluster1_indices_query3, 'DateTimeEDT'], 
         Query3.loc[cluster1_indices_query3, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax1.scatter(Query3.loc[cluster1_indices_query3].iloc[cluster1_peaks_query3]['DateTimeEDT'],
            Query3.loc[cluster1_indices_query3].iloc[cluster1_peaks_query3]['Pressure (mmHg)'], 
            color='red', label='Detected Peaks - Cluster 1')

ax1.set_xlim([cluster1_xlim_start_query3, cluster1_xlim_end_query3])
ax1.set_xlabel('DateTimeEDT')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('Cluster 1: Pressure Over Time with Detected Peaks')
ax1.legend()
ax1.grid(True)

# Plot second normogravity compression window
ax2.plot(Query3.loc[cluster2_indices_query3, 'DateTimeEDT'], 
         Query3.loc[cluster2_indices_query3, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax2.scatter(Query3.loc[cluster2_indices_query3].iloc[cluster2_peaks_query3]['DateTimeEDT'],
            Query3.loc[cluster2_indices_query3].iloc[cluster2_peaks_query3]['Pressure (mmHg)'], 
            color='green', label='Detected Peaks - Cluster 2')

ax2.set_xlim([cluster2_xlim_start_query3, cluster2_xlim_end_query3])
ax2.set_xlabel('DateTimeEDT')
ax2.set_ylabel('Pressure (mmHg)')
ax2.set_title('Cluster 2: Pressure Over Time with Detected Peaks')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Define helper to isolate data points for each normogravity compression window
def isolate_cluster_data(Query3, cluster_indices):
    return Query3.loc[cluster_indices]

# Combine cluster data into a dictionary for downstream analysis
isolated_clusters_query3 = {'cluster1': [], 'cluster2': []}
isolated_clusters_query3['cluster1'] = isolate_cluster_data(Query3, cluster1_indices_query3)
isolated_clusters_query3['cluster2'] = isolate_cluster_data(Query3, cluster2_indices_query3)

# Preserve masks for potential external use (4 cm vs 5 cm depths)
four_cm_cluster1_query3 = [cluster1_indices_query3]
five_cm_cluster2_query3 = [cluster2_indices_query3]

## --- NORMOGRAVITY: DECTECTING SYSTOLIC AND DIASTOLIC PEAKS --- ##

def detect_peaks_and_troughs(cluster_data, lower_bound, upper_bound):
    # Detect systolic peaks (compression maxima) in normogravity windows
    systolic_peaks, _ = find_peaks(cluster_data['Pressure (mmHg)'], height=40, distance=25)
    systolic_peak_values = cluster_data.iloc[systolic_peaks]['Pressure (mmHg)']
    systolic_peak_times = cluster_data.iloc[systolic_peaks]['DateTimeUTC']

    # Detect diastolic troughs (relaxation minima) between 15–25 mmHg
    diastolic_peaks = []
    for i in range(1, len(cluster_data) - 1):
        if 15 <= cluster_data['Pressure (mmHg)'].iloc[i] <= 25:
            if cluster_data['Pressure (mmHg)'].iloc[i] < cluster_data['Pressure (mmHg)'].iloc[i - 1] and cluster_data['Pressure (mmHg)'].iloc[i] < cluster_data['Pressure (mmHg)'].iloc[i + 1]:
                diastolic_peaks.append(i)

    # Keep only the lowest diastolic trough before the first systolic peak
    filtered_diastolic_peaks = []
    if len(systolic_peaks) > 0:
        first_systolic_peak = systolic_peaks[0]
        initial_diastolic_troughs = [peak for peak in diastolic_peaks if peak < first_systolic_peak]
        if initial_diastolic_troughs:
            min_initial_trough = min(initial_diastolic_troughs, key=lambda x: cluster_data['Pressure (mmHg)'].iloc[x])
            filtered_diastolic_peaks.append(min_initial_trough)

    # Keep the lowest diastolic trough between successive systolic peaks
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        segment_peaks = [peak for peak in diastolic_peaks if start < peak < end]
        if segment_peaks:
            min_peak = min(segment_peaks, key=lambda x: cluster_data['Pressure (mmHg)'].iloc[x])
            filtered_diastolic_peaks.append(min_peak)

    diastolic_peak_values = cluster_data.iloc[filtered_diastolic_peaks]['Pressure (mmHg)']
    diastolic_peak_times = cluster_data.iloc[filtered_diastolic_peaks]['DateTimeUTC']

    # Detect dicrotic notches (secondary minima) between systolic peaks
    all_notches, _ = find_peaks(-cluster_data['Pressure (mmHg)'], distance=5)
    dicrotic_notches = []
    dicrotic_notch_times = []
    last_systolic_peak = -1

    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        segment_notches = [notch for notch in all_notches if start < notch < end]

        # Keep only the first notch after each systolic peak within bounds
        if segment_notches:
            first_notch = segment_notches[0]
            if lower_bound <= cluster_data['Pressure (mmHg)'].iloc[first_notch] <= upper_bound:
                dicrotic_notches.append(first_notch)
                dicrotic_notch_times.append(cluster_data['DateTimeUTC'].iloc[first_notch])

    # Handle potential dicrotic notch after the last systolic peak
    if len(systolic_peaks) > 0:
        last_systolic_peak = systolic_peaks[-1]
        segment_notches = [notch for notch in all_notches if notch > last_systolic_peak]
        if segment_notches:
            first_notch = segment_notches[0]
            if lower_bound <= cluster_data['Pressure (mmHg)'].iloc[first_notch] <= upper_bound:
                dicrotic_notches.append(first_notch)
                dicrotic_notch_times.append(cluster_data['DateTimeUTC'].iloc[first_notch])

    return systolic_peaks, systolic_peak_values, systolic_peak_times, filtered_diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times

def create_analysis_df(analysis):
    # Combine summary metrics into a one-row DataFrame
    return pd.DataFrame({
        'average_systolic_pressure (mmHg)': [analysis['average_systolic_pressure (mmHg)']],
        'average_diastolic_pressure (mmHg)': [analysis['average_diastolic_pressure (mmHg)']],
    })

def analyze_cluster(cluster_data, lower_bound, upper_bound, cluster_name):
    # Analyze systolic/diastolic behaviour and dicrotic notches for a normogravity cluster
    systolic_peaks, systolic_peak_values, systolic_peak_times, diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times = detect_peaks_and_troughs(cluster_data, lower_bound, upper_bound)

    if cluster_name == 'cluster1':
         # Optionally adjust specific outliers for first compression window (left as pass to preserve raw logic)
        if len(diastolic_peaks) > 0:
            pass  
        if len(dicrotic_notches) > 0:
            pass  

    if cluster_name == 'cluster2':
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
            
    onground_analysis = {
        'average_systolic_pressure (mmHg)': systolic_peak_values.mean(),
        'average_diastolic_pressure (mmHg)': diastolic_peak_values.mean(),
        'systolic_peaks (mmHg)': systolic_peak_values,
        'systolic_peak_times': systolic_peak_times,
        'diastolic_troughs (mmHg)': diastolic_peak_values,
        'diastolic_trough_times': diastolic_peak_times,
        'dicrotic_notches (mmHg)': cluster_data['Pressure (mmHg)'].iloc[dicrotic_notches],
        'dicrotic_notch_times': dicrotic_notch_times,
        'systolic_peak_indices': systolic_peaks,
        'diastolic_trough_indices': diastolic_peaks,
        'dicrotic_notch_indices': dicrotic_notches,
    }

    onground_analysis_df = create_analysis_df(onground_analysis)
    systolic_peaks_df = pd.DataFrame({'systolic_peaks (mmHg)': onground_analysis['systolic_peaks (mmHg)'].tolist(), 'systolic_peak_times': onground_analysis['systolic_peak_times'].tolist()})
    diastolic_troughs_df = pd.DataFrame({'diastolic_troughs (mmHg)': onground_analysis['diastolic_troughs (mmHg)'].tolist(), 'diastolic_trough_times': onground_analysis['diastolic_trough_times'].tolist()})
    dicrotic_notches_df = pd.DataFrame({'dicrotic_notches (mmHg)': onground_analysis['dicrotic_notches (mmHg)'].tolist(), 'dicrotic_notch_times': onground_analysis['dicrotic_notch_times']})
    onground_analysis_df = pd.concat([onground_analysis_df, systolic_peaks_df, diastolic_troughs_df, dicrotic_notches_df], axis=1)

    # Compute MAP (mean arterial pressure) for each compression cycle, then average
    onground_analysis_df['MAP (mmHg)'] = (2/3) * onground_analysis_df['diastolic_troughs (mmHg)'].shift(-1) + (1/3) * onground_analysis_df['systolic_peaks (mmHg)']
    average_MAP = onground_analysis_df['MAP (mmHg)'].mean()
    onground_analysis_df['average_MAP (mmHg)'] = [average_MAP] + [None] * (len(onground_analysis_df) - 1)

    # Compute systole duration (dicrotic notch to following diastolic trough)
    onground_analysis_df['systole_duration (s)'] = (pd.to_datetime(onground_analysis_df['diastolic_trough_times']) - pd.to_datetime(onground_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_systole_duration = onground_analysis_df['systole_duration (s)'].mean()
    onground_analysis_df['average_systole_duration (s)'] = [average_systole_duration] + [None] * (len(onground_analysis_df) - 1)

    # Compute diastole duration (dicrotic notch to next diastolic trough)
    onground_analysis_df['diastole_duration (s)'] = (pd.to_datetime(onground_analysis_df['diastolic_trough_times'].shift(-1)) - pd.to_datetime(onground_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_diastole_duration = onground_analysis_df['diastole_duration (s)'].mean()
    onground_analysis_df['average_diastole_duration (s)'] = [average_diastole_duration] + [None] * (len(onground_analysis_df) - 1)

    # Compute compression rate (systolic-to-systolic interval, converted to compressions/min)
    onground_analysis_df['compression_rate'] = (pd.to_datetime(onground_analysis_df['systolic_peak_times'].shift(-1)) - pd.to_datetime(onground_analysis_df['systolic_peak_times'])).abs().dt.total_seconds()
    average_time_interval = onground_analysis_df['compression_rate'].mean()
    average_compression_rate = (1 / average_time_interval) * 60 if average_time_interval != 0 else None
    onground_analysis_df['average_compression_rate (compressions/min)'] = [average_compression_rate] + [None] * (len(onground_analysis_df) - 1)

    # Compute pulse pressure for each compression
    onground_analysis_df['pulse_pressure (mmHg)'] = onground_analysis_df['systolic_peaks (mmHg)'] - onground_analysis_df['diastolic_troughs (mmHg)']
    average_pulse_pressure = onground_analysis_df['pulse_pressure (mmHg)'].mean()
    onground_analysis_df['average_pulse_pressure (mmHg)'] = [average_pulse_pressure] + [None] * (len(onground_analysis_df) - 1)

    onground_analysis_df.to_csv(f'{cluster_name}_analysis.csv', index=False)
    print(f"{cluster_name.capitalize()} Analysis:")
    print(onground_analysis_df)

    return onground_analysis

def plot_results(cluster_data, analysis, cluster_name):
    # Plot systolic, diastolic, and dicrotic features in sliding 5 s windows
    start_time = cluster_data['DateTimeUTC'].min()
    end_time = cluster_data['DateTimeUTC'].max()
    interval = pd.Timedelta(seconds=5)
    
    current_time = start_time
    while current_time < end_time:
        window_end = current_time + interval
        plt.figure(figsize=(12, 6))
        plt.plot(cluster_data['DateTimeUTC'], cluster_data['Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')
        plt.scatter(cluster_data.iloc[analysis['systolic_peak_indices']]['DateTimeUTC'], analysis['systolic_peaks (mmHg)'], color='red', label='Systolic Peaks')
        plt.scatter(cluster_data.iloc[analysis['diastolic_trough_indices']]['DateTimeUTC'], analysis['diastolic_troughs (mmHg)'], color='green', label='Diastolic Troughs')
        plt.scatter(cluster_data.iloc[analysis['dicrotic_notch_indices']]['DateTimeUTC'], analysis['dicrotic_notches (mmHg)'], color='orange', label='Dicrotic Notches')
        plt.xlabel('DateTimeUTC')
        plt.ylabel('Pressure (mmHg)')
        plt.title(f'{cluster_name.capitalize()}: Pressure Over Time with Systolic Peaks, Diastolic Troughs, and Dicrotic Notches')
        plt.legend()
        plt.grid(True)
        plt.xlim(current_time, window_end)
        plt.show()
        
        current_time = window_end

# Analyze and visualize normogravity compression windows (first and second compression windows)
cluster1_analysis = analyze_cluster(isolated_clusters_query3['cluster1'], lower_bound=30, upper_bound=50, cluster_name='cluster1')
plot_results(isolated_clusters_query3['cluster1'], cluster1_analysis, cluster_name='cluster1')
cluster2_analysis = analyze_cluster(isolated_clusters_query3['cluster2'], lower_bound=30, upper_bound=50, cluster_name='cluster2')
plot_results(isolated_clusters_query3['cluster2'], cluster2_analysis, cluster_name='cluster2')

## --- HYPOGRAVITY FLIGHT 1: DECTECTING COMPRESSION PEAKS --- ##

file_path = r"data/Query1.csv"
Query1 = pd.read_csv(file_path)

# Convert pressure from mmHg to Pa (1 mmHg = 133.322 Pa)
Query1['Pressure (Pa)'] = Query1['Pressure (mmHg)'] * 133.322

# Sort Flight 1 samples by UTC time
Query1['DateTimeUTC'] = pd.to_datetime(Query1['DateTimeUTC'])
Query1 = Query1.sort_values(by='DateTimeUTC')
Query1.to_csv(file_path, index=False)

# Visualize hypogravity Flight 1 pressure trace with g-force
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(Query1['DateTimeUTC'], Query1['Pressure (Pa)'], label='Pressure (Pa)', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure Over Time')
ax1.grid(True)

ax1_gforce = ax1.twinx()
ax1_gforce.plot(Query1['DateTimeUTC'], Query1['Gforce'], label='G-force', color='red', alpha=0.7)
ax1_gforce.set_ylabel('G-force')
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Identify hypogravity parabolas in Flight 1 based on g-force profile
def identify_parabolas(pressure_values, gforce_values, time_values, threshold_gforce=0.10, min_points=5, max_duration_seconds=30):
    parabolas = {'5cm_compression': []}
    
    current_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
    is_within_parabola = False
    parabola_count = 0
    start_time = None
    
    for i in range(len(pressure_values)):
        if gforce_values[i] < threshold_gforce:
            if not is_within_parabola:
                if len(current_parabola['pressure']) > min_points:
                    current_parabola['number'] = parabola_count + 1
                    parabolas['5cm_compression'].append(current_parabola)

                current_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
                is_within_parabola = True
                parabola_count += 1
                start_time = time_values[i]
            
            current_parabola['pressure'].append(pressure_values[i])
            current_parabola['gforce'].append(gforce_values[i])
            current_parabola['indices'].append(i)
        
        else:
            if is_within_parabola:
                time_diff = (time_values[i] - start_time).astype('timedelta64[s]').item().total_seconds()
                if time_diff > max_duration_seconds:
                    is_within_parabola = False

    if len(current_parabola['pressure']) > min_points:
        current_parabola['number'] = parabola_count + 1
        parabolas['5cm_compression'].append(current_parabola)

    return parabolas

# Renumber parabolas in Flight 1 (5 cm compression only) consecutively
def renumber_parabolas(parabolas):
    for i, parabola in enumerate(parabolas['5cm_compression']):
        parabola['number'] = i + 1  # 1, 2, 3, 4, ...
    return parabolas

# Extract pressure, g-force, and time vectors for Flight 1
pressure_values = Query1['Pressure (Pa)'].values
gforce_values = Query1['Gforce'].values
time_values = pd.to_datetime(Query1['DateTimeUTC']).values

# Detect hypogravity parabolas and renumber
parabolas_query1 = identify_parabolas(pressure_values, gforce_values, time_values)
parabolas_query1 = renumber_parabolas(parabolas_query1)
parabolas_query1['5cm_compression'] = parabolas_query1['5cm_compression'][2:]
parabolas_query1 = renumber_parabolas(parabolas_query1)

# Store 5 cm compression parabolas for Flight 1
five_cm_parabolas_query1 = parabolas_query1['5cm_compression']

print(f"Number of 5cm compression parabolas: {len(five_cm_parabolas_query1)}")

# Isolate time-series for each identified hypogravity parabola (Flight 1)
def isolate_parabola_data(Query1, parabola):
    start_index = parabola['indices'][0]
    end_index = parabola['indices'][-1]
    return Query1.iloc[start_index:end_index + 1]

isolated_parabolas = {'5cm_compression': []}

for parabola in five_cm_parabolas_query1:
    isolated_data = isolate_parabola_data(Query1, parabola)
    isolated_parabolas['5cm_compression'].append(isolated_data)

# Plot Flight 1 pressure with color-coded 5 cm hypogravity parabolas
plt.figure(figsize=(12, 8))
plt.plot(time_values, pressure_values, label='Pressure (Pa)', color='lightgray', alpha=0.5)

for parabola in five_cm_parabolas_query1:
    plt.plot(time_values[parabola['indices']], parabola['pressure'], color='green', label='5cm Compression' if parabola == five_cm_parabolas_query1[0] else "")
    mid_index = len(parabola['indices']) // 2
    plt.text(time_values[parabola['indices'][mid_index]], parabola['pressure'][mid_index], f"{parabola['number']}", color='green')

plt.xlabel('Time')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Over Time with Color-Coded Parabolas')
plt.legend()
plt.grid(True)

plt.show()

# Print all 5 cm compression parabolas for Flight 1 (full time-series)
print("\nAll parabolas in 5cm compression:")
for parabola in parabolas_query1['5cm_compression']:
    print(f"Parabola {parabola['number']}:")
    print(isolate_parabola_data(Query1, parabola))
    print()
    
## --- HYPOGRAVITY FLIGHT 1: DECTECTING SYSTOLIC AND DIASTOLIC PEAKS --- ##

# Function to detect peaks and troughs
def detect_peaks_and_troughs(parabola_data, lower_bound, upper_bound):
    systolic_peaks, _ = find_peaks(parabola_data['Pressure (mmHg)'], height=45, distance=25)
    systolic_peak_values = parabola_data.iloc[systolic_peaks]['Pressure (mmHg)']
    systolic_peak_times = parabola_data.iloc[systolic_peaks]['DateTimeUTC']

    # Diastolic troughs detection using two different distances
    diastolic_peaks_50, _ = find_peaks(-parabola_data['Pressure (mmHg)'], height=-23, distance=50)
    diastolic_peaks_70, _ = find_peaks(-parabola_data['Pressure (mmHg)'], height=-23, distance=70)
    combined_diastolic_peaks = np.unique(np.concatenate((diastolic_peaks_50, diastolic_peaks_70)))

    # Filter diastolic peaks based on the pressure threshold (<= 27 mmHg)
    filtered_diastolic_peaks = [peak for peak in combined_diastolic_peaks if parabola_data['Pressure (mmHg)'].iloc[peak] <= 27]

    # Capture the first diastolic trough before the first systolic peak
    first_systolic_peak = systolic_peaks[0]
    troughs_before_first_systolic = [peak for peak in filtered_diastolic_peaks if peak < first_systolic_peak]
    if troughs_before_first_systolic:
        first_diastolic_trough = min(troughs_before_first_systolic, key=lambda x: parabola_data['Pressure (mmHg)'].iloc[x])
    else:
        first_diastolic_trough = None

    # Dicrotic notches detection (as troughs)
    all_notches, _ = find_peaks(-parabola_data['Pressure (mmHg)'], distance=5)  # Detect troughs
    dicrotic_notches = []
    dicrotic_notch_times = []

    for peak in systolic_peaks:
        for notch in all_notches:
            if notch > peak and lower_bound <= parabola_data['Pressure (mmHg)'].iloc[notch] <= upper_bound:
                dicrotic_notches.append(notch)
                dicrotic_notch_times.append(parabola_data['DateTimeUTC'].iloc[notch])
                break  # Only take the first notch after the systolic peak

    # Keep only the lowest diastolic trough between each systolic peak and dicrotic notch
    final_diastolic_peaks = [first_diastolic_trough] if first_diastolic_trough is not None else []
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        troughs_in_range = [peak for peak in filtered_diastolic_peaks if start < peak < end]
        if troughs_in_range:
            lowest_trough = min(troughs_in_range, key=lambda x: parabola_data['Pressure (mmHg)'].iloc[x])
            final_diastolic_peaks.append(lowest_trough)

    diastolic_peak_values = parabola_data.iloc[final_diastolic_peaks]['Pressure (mmHg)']
    diastolic_peak_times = parabola_data.iloc[final_diastolic_peaks]['DateTimeUTC']

    return systolic_peaks, systolic_peak_values, systolic_peak_times, final_diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times

def create_analysis_df(flight1_analysis):
    return pd.DataFrame({
        'average_systolic_pressure (mmHg)': [flight1_analysis['average_systolic_pressure (mmHg)']],
        'average_diastolic_pressure (mmHg)': [flight1_analysis['average_diastolic_pressure (mmHg)']],
    })

def analyze_parabola(parabola_data, lower_bound, upper_bound, parabola_name):
    systolic_peaks, systolic_peak_values, systolic_peak_times, diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times = detect_peaks_and_troughs(parabola_data, lower_bound, upper_bound)

    flight1_analysis = {
        'average_systolic_pressure (mmHg)': systolic_peak_values.mean(),
        'average_diastolic_pressure (mmHg)': diastolic_peak_values.mean(),
        'systolic_peaks (mmHg)': systolic_peak_values,
        'systolic_peak_times': systolic_peak_times,
        'diastolic_troughs (mmHg)': diastolic_peak_values,
        'diastolic_trough_times': diastolic_peak_times,
        'dicrotic_notches (mmHg)': parabola_data['Pressure (mmHg)'].iloc[dicrotic_notches],
        'dicrotic_notch_times': dicrotic_notch_times,
        'systolic_peak_indices': systolic_peaks,
        'diastolic_trough_indices': diastolic_peaks,
        'dicrotic_notch_indices': dicrotic_notches
    }

    flight1_analysis_df = create_analysis_df(flight1_analysis)
    systolic_peaks_df = pd.DataFrame({'systolic_peaks (mmHg)': flight1_analysis['systolic_peaks (mmHg)'].tolist(), 'systolic_peak_times': flight1_analysis['systolic_peak_times'].tolist()})
    diastolic_troughs_df = pd.DataFrame({'diastolic_troughs (mmHg)': flight1_analysis['diastolic_troughs (mmHg)'].tolist(), 'diastolic_trough_times': flight1_analysis['diastolic_trough_times'].tolist()})
    dicrotic_notches_df = pd.DataFrame({'dicrotic_notches (mmHg)': flight1_analysis['dicrotic_notches (mmHg)'].tolist(), 'dicrotic_notch_times': flight1_analysis['dicrotic_notch_times']})
    flight1_analysis_df = pd.concat([flight1_analysis_df, systolic_peaks_df, diastolic_troughs_df, dicrotic_notches_df], axis=1)

    # Calculate MAP 
    flight1_analysis_df['MAP (mmHg)'] = (2/3) * flight1_analysis_df['diastolic_troughs (mmHg)'].shift(-1) + (1/3) * flight1_analysis_df['systolic_peaks (mmHg)']
    average_MAP = flight1_analysis_df['MAP (mmHg)'].mean()
    flight1_analysis_df['average_MAP (mmHg)'] = [average_MAP] + [None] * (len(flight1_analysis_df) - 1)

    # Calculate systole duration in seconds and take the absolute value
    flight1_analysis_df['systole_duration (s)'] = (pd.to_datetime(flight1_analysis_df['diastolic_trough_times']) - pd.to_datetime(flight1_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_systole_duration = flight1_analysis_df['systole_duration (s)'].mean()
    flight1_analysis_df['average_systole_duration (s)'] = [average_systole_duration] + [None] * (len(flight1_analysis_df) - 1)

    # Calculate diastole duration in seconds and take the absolute value
    flight1_analysis_df['diastole_duration (s)'] = (pd.to_datetime(flight1_analysis_df['diastolic_trough_times'].shift(-1)) - pd.to_datetime(flight1_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_diastole_duration = flight1_analysis_df['diastole_duration (s)'].mean()
    flight1_analysis_df['average_diastole_duration (s)'] = [average_diastole_duration] + [None] * (len(flight1_analysis_df) - 1)

    # Calculate compression rate from systolic peak to systolic peak
    flight1_analysis_df['compression_rate'] = (pd.to_datetime(flight1_analysis_df['systolic_peak_times'].shift(-1)) - pd.to_datetime(flight1_analysis_df['systolic_peak_times'])).abs().dt.total_seconds()
    average_time_interval = flight1_analysis_df['compression_rate'].mean()
    average_compression_rate = (1 / average_time_interval) * 60 if average_time_interval != 0 else None
    flight1_analysis_df['average_compression_rate (compressions/min)'] = [average_compression_rate] + [None] * (len(flight1_analysis_df) - 1)

    flight1_analysis_df.to_csv(f'{parabola_name}_analysis.csv', index=False)

    return flight1_analysis

# Function to plot results for a parabola
def plot_parabola_results(parabola_data, flight1_analysis, parabola_name):
    plt.figure(figsize=(12, 6))
    plt.plot(parabola_data['DateTimeUTC'], parabola_data['Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')
    plt.scatter(parabola_data.iloc[flight1_analysis['systolic_peak_indices']]['DateTimeUTC'], flight1_analysis['systolic_peaks (mmHg)'], color='red', label='Systolic Peaks')
    plt.scatter(parabola_data.iloc[flight1_analysis['diastolic_trough_indices']]['DateTimeUTC'], flight1_analysis['diastolic_troughs (mmHg)'], color='green', label='Diastolic Troughs')
    plt.scatter(parabola_data.iloc[flight1_analysis['dicrotic_notch_indices']]['DateTimeUTC'], flight1_analysis['dicrotic_notches (mmHg)'], color='orange', label='Dicrotic Notches')
    plt.xlabel('DateTimeUTC')
    plt.ylabel('Pressure (mmHg)')
    plt.title(f'{parabola_name.capitalize()}: Pressure Over Time with Systolic Peaks, Diastolic Troughs, and Dicrotic Notches')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Assuming five_cm_parabolas_query1 and Query1 are already defined
lower_bound = 35
upper_bound = 49

# Analyze and plot all parabolas in the list
for i, parabola in enumerate(five_cm_parabolas_query1):
    # Isolate data for each parabola
    parabola_data = isolate_parabola_data(Query1, parabola)

    # Analyze the parabola
    parabola_analysis = analyze_parabola(parabola_data, lower_bound, upper_bound, parabola_name=f'parabola{i+1}')

    # Plot the results for each parabola with zoomed-in view on the specified 1-second interval
    plot_parabola_results(parabola_data, parabola_analysis, parabola_name=f'parabola{i+1}')
    
## --- HYPOGRAVITY FLIGHT 2: DECTECTING COMPRESSION PEAKS --- ##

file_path = r"data/Query2.csv"
Query2 = pd.read_csv(file_path)

# Convert pressure from mmHg to Pa (1 mmHg = 133.322 Pa)
Query2['Pressure (Pa)'] = Query2['Pressure (mmHg)'] * 133.322

Query2.to_csv(file_path, index=False)

# Convert DateTimeUTC to a datetime object
Query2['DateTimeUTC'] = pd.to_datetime(Query2['DateTimeUTC'])

# Visualize hypogravity Flight 2 pressure trace with g-force overlay
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(Query2['DateTimeUTC'], Query2['Pressure (Pa)'], label='Pressure (Pa)', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure Over Time')
ax1.grid(True)
s
ax1_gforce = ax1.twinx()
ax1_gforce.plot(Query2['DateTimeUTC'], Query2['Gforce'], label='G-force', color='red', alpha=0.7)
ax1_gforce.set_ylabel('G-force')
ax1.legend(loc='upper left')
ax1_gforce.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Identify hypogravity parabolas in Flight 2 and classify by compression depth (4 cm vs 5 cm)
def identify_parabolas(pressure_values, gforce_values, time_values, threshold_gforce=0.10, min_points=5, max_duration_seconds=30):
    parabolas = {'4cm_compression': [], '5cm_compression': []}
    
    current_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
    is_within_parabola = False
    parabola_count = 0
    start_time = None
    
    for i in range(len(pressure_values)):
        if gforce_values[i] < threshold_gforce:
            if not is_within_parabola:
                if len(current_parabola['pressure']) > min_points:
                    current_parabola['number'] = parabola_count + 1
                    if (parabola_count + 1) % 2 == 0:  # Even-numbered parabolas (2, 4, 6, 8, 10)
                        parabolas['4cm_compression'].append(current_parabola)
                    else:  # Odd-numbered parabolas (1, 3, 5, 7, 9)
                        parabolas['5cm_compression'].append(current_parabola)

                current_parabola = {'pressure': [], 'gforce': [], 'indices': [], 'number': None}
                is_within_parabola = True
                parabola_count += 1
                start_time = time_values[i]
            
            current_parabola['pressure'].append(pressure_values[i])
            current_parabola['gforce'].append(gforce_values[i])
            current_parabola['indices'].append(i)
        
        else:
            if is_within_parabola:
                time_diff = (time_values[i] - start_time).astype('timedelta64[s]').item().total_seconds()
                if time_diff > max_duration_seconds:
                    is_within_parabola = False

    if len(current_parabola['pressure']) > min_points:
        current_parabola['number'] = parabola_count + 1
        if (parabola_count + 1) % 2 == 0:  # Even-numbered parabolas (2, 4, 6, 8, 10)
            parabolas['4cm_compression'].append(current_parabola)
        else:  # Odd-numbered parabolas (1, 3, 5, 7, 9)
            parabolas['5cm_compression'].append(current_parabola)

    return parabolas

# Renumber Flight 2 parabolas while preserving 4 cm / 5 cm pattern
def renumber_parabolas(parabolas):
    for i, parabola in enumerate(parabolas['4cm_compression']):
        parabola['number'] = 2 * i + 1  # 1, 3, 5, 7, 9, ...
    for i, parabola in enumerate(parabolas['5cm_compression']):
        parabola['number'] = 2 * (i + 1)  # 2, 4, 6, 8, 10, ...
    return parabolas

# Extract vectors for Flight 2 parabola detection
pressure_values = Query2['Pressure (Pa)'].values
gforce_values = Query2['Gforce'].values
time_values = pd.to_datetime(Query2['DateTimeUTC']).values

# Detect and renumber parabolas for Flight 2
parabolas_query2 = identify_parabolas(pressure_values, gforce_values, time_values)
parabolas_query2 = renumber_parabolas(parabolas_query2)

four_cm_parabolas_query2 = parabolas_query2['4cm_compression']
five_cm_parabolas_query2 = parabolas_query2['5cm_compression']

print(f"Number of 4cm compression parabolas: {len(four_cm_parabolas_query2)}")
print(f"Number of 5cm compression parabolas: {len(five_cm_parabolas_query2)}")

# Isolate time-series for each hypogravity parabola
def isolate_parabola_data(Query2, parabola):
    start_index = parabola['indices'][0]
    end_index = parabola['indices'][-1]
    return Query2.iloc[start_index:end_index + 1]

isolated_parabolas = {'4cm_compression': [], '5cm_compression': []}

for parabola in four_cm_parabolas_query2:
    isolated_data = isolate_parabola_data(Query2, parabola)
    isolated_parabolas['4cm_compression'].append(isolated_data)

for parabola in five_cm_parabolas_query2:
    isolated_data = isolate_parabola_data(Query2, parabola)
    isolated_parabolas['5cm_compression'].append(isolated_data)

# Plot Flight 2 pressure trace with 4 cm and 5 cm hypogravity parabolas color-coded
plt.figure(figsize=(12, 8))

plt.plot(time_values, pressure_values, label='Pressure (Pa)', color='lightgray', alpha=0.5)

# Color code the 4cm compression parabolas
for parabola in four_cm_parabolas_query2:
    plt.plot(time_values[parabola['indices']], parabola['pressure'], color='blue', label='4cm Compression' if parabola == four_cm_parabolas_query2[0] else "")
    mid_index = len(parabola['indices']) // 2
    plt.text(time_values[parabola['indices'][mid_index]], parabola['pressure'][mid_index], f"{parabola['number']}", color='blue')

# Color code the 5cm compression parabolas
for parabola in five_cm_parabolas_query2:
    plt.plot(time_values[parabola['indices']], parabola['pressure'], color='green', label='5cm Compression' if parabola == five_cm_parabolas_query2[0] else "")
    mid_index = len(parabola['indices']) // 2
    plt.text(time_values[parabola['indices'][mid_index]], parabola['pressure'][mid_index], f"{parabola['number']}", color='green')

plt.xlabel('Time')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Over Time with Color-Coded Parabolas')
plt.legend()
plt.grid(True)

plt.show()

# Print full time-series for each 4 cm parabola
print("\nAll parabolas in 4cm compression:")
for parabola in parabolas_query2['4cm_compression']:
    print(f"Parabola {parabola['number']}:")
    print(isolate_parabola_data(Query2, parabola))
    print()
    
# Print full time-series for each 5 cm parabola
print("\nAll parabolas in 5cm compression:")
for parabola in five_cm_parabolas_query2:
    print(f"Parabola {parabola['number']}:")
    print(isolate_parabola_data(Query2, parabola))
    print()
    
## --- HYPOGRAVITY FLIGHT 2: DECTECTING SYSTOLIC AND DIASTOLIC PEAKS --- ##

# Detect systolic, diastolic, and dicrotic features within Flight 2 parabolas
def detect_peaks_and_troughs(parabola_data, lower_bound, upper_bound):
    systolic_peaks, _ = find_peaks(parabola_data['Pressure (mmHg)'], height=40, distance=25)
    systolic_peak_values = parabola_data.iloc[systolic_peaks]['Pressure (mmHg)']
    systolic_peak_times = parabola_data.iloc[systolic_peaks]['DateTimeUTC']

    # Detect diastolic troughs using two spacing heuristics and combine
    diastolic_peaks_50, _ = find_peaks(-parabola_data['Pressure (mmHg)'], height=-30, distance=20)
    diastolic_peaks_70, _ = find_peaks(-parabola_data['Pressure (mmHg)'], height=-30, distance=80)
    combined_diastolic_peaks = np.unique(np.concatenate((diastolic_peaks_50, diastolic_peaks_70)))

    # Filter diastolic peaks based on upper pressure threshold (<= 30 mmHg)
    filtered_diastolic_peaks = [peak for peak in combined_diastolic_peaks if parabola_data['Pressure (mmHg)'].iloc[peak] <= 30]

    # Capture lowest diastolic trough before first systolic peak
    first_systolic_peak = systolic_peaks[0]
    troughs_before_first_systolic = [peak for peak in filtered_diastolic_peaks if peak < first_systolic_peak]
    if troughs_before_first_systolic:
        first_diastolic_trough = min(troughs_before_first_systolic, key=lambda x: parabola_data['Pressure (mmHg)'].iloc[x])
    else:
        first_diastolic_trough = None

    # Detect dicrotic notches after each systolic peak
    all_notches, _ = find_peaks(-parabola_data['Pressure (mmHg)'], distance=5)
    dicrotic_notches = []
    dicrotic_notch_times = []

    for peak in systolic_peaks:
        for notch in all_notches:
            if notch > peak and lower_bound <= parabola_data['Pressure (mmHg)'].iloc[notch] <= upper_bound:
                dicrotic_notches.append(notch)
                dicrotic_notch_times.append(parabola_data['DateTimeUTC'].iloc[notch])
                break 
    
    # Keep lowest diastolic trough between each systolic peak and next dicrotic notch
    final_diastolic_peaks = [first_diastolic_trough] if first_diastolic_trough is not None else []
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        troughs_in_range = [peak for peak in filtered_diastolic_peaks if start < peak < end]
        if troughs_in_range:
            lowest_trough = min(troughs_in_range, key=lambda x: parabola_data['Pressure (mmHg)'].iloc[x])
            final_diastolic_peaks.append(lowest_trough)

    diastolic_peak_values = parabola_data.iloc[final_diastolic_peaks]['Pressure (mmHg)']
    diastolic_peak_times = parabola_data.iloc[final_diastolic_peaks]['DateTimeUTC']

    return systolic_peaks, systolic_peak_values, systolic_peak_times, final_diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times

def create_analysis_df(analysis):
    # Build one-row summary DataFrame for a single Flight 2 parabola
    return pd.DataFrame({
        'average_systolic_pressure (mmHg)': [analysis['average_systolic_pressure (mmHg)']],
        'average_diastolic_pressure (mmHg)': [analysis['average_diastolic_pressure (mmHg)']],
    })

def analyze_parabola(parabola_data, lower_bound, upper_bound, parabola_name):
    # Analyze systolic/diastolic behaviour and dicrotic notches for a Flight 2 parabola
    systolic_peaks, systolic_peak_values, systolic_peak_times, diastolic_peaks, diastolic_peak_values, diastolic_peak_times, dicrotic_notches, dicrotic_notch_times = detect_peaks_and_troughs(parabola_data, lower_bound, upper_bound)

    # Optionally trim extra peaks for specific parabolas (empirical cleanup)
    if parabola_name in ['5cm_parabola6', '4cm_parabola5']:
        if len(systolic_peaks) > 0:
            systolic_peaks = systolic_peaks[1:]
            systolic_peak_values = systolic_peak_values[1:]
            systolic_peak_times = systolic_peak_times[1:]
        if len(dicrotic_notches) > 0:
            dicrotic_notches = dicrotic_notches[1:]
            dicrotic_notch_times = dicrotic_notch_times[1:]

    flight2_analysis = {
        'average_systolic_pressure (mmHg)': systolic_peak_values.mean(),
        'average_diastolic_pressure (mmHg)': diastolic_peak_values.mean(),
        'systolic_peaks (mmHg)': systolic_peak_values,
        'systolic_peak_times': systolic_peak_times,
        'diastolic_troughs (mmHg)': diastolic_peak_values,
        'diastolic_trough_times': diastolic_peak_times,
        'dicrotic_notches (mmHg)': parabola_data['Pressure (mmHg)'].iloc[dicrotic_notches],
        'dicrotic_notch_times': dicrotic_notch_times,
        'systolic_peak_indices': systolic_peaks,
        'diastolic_trough_indices': diastolic_peaks,
        'dicrotic_notch_indices': dicrotic_notches
    }

    flight2_analysis_df = create_analysis_df(flight2_analysis)
    systolic_peaks_df = pd.DataFrame({'systolic_peaks (mmHg)': flight2_analysis['systolic_peaks (mmHg)'].tolist(), 'systolic_peak_times': flight2_analysis['systolic_peak_times'].tolist()})
    diastolic_troughs_df = pd.DataFrame({'diastolic_troughs (mmHg)': flight2_analysis['diastolic_troughs (mmHg)'].tolist(), 'diastolic_trough_times': flight2_analysis['diastolic_trough_times'].tolist()})
    dicrotic_notches_df = pd.DataFrame({'dicrotic_notches (mmHg)': flight2_analysis['dicrotic_notches (mmHg)'].tolist(), 'dicrotic_notch_times': flight2_analysis['dicrotic_notch_times']})
    flight2_analysis_df = pd.concat([flight2_analysis_df, systolic_peaks_df, diastolic_troughs_df, dicrotic_notches_df], axis=1)

    # Compute MAP, systole/diastole durations, and compression rate for this Flight 2 parabola
    flight2_analysis_df['MAP (mmHg)'] = (2/3) * flight2_analysis_df['diastolic_troughs (mmHg)'].shift(-1) + (1/3) * flight2_analysis_df['systolic_peaks (mmHg)']
    average_MAP = flight2_analysis_df['MAP (mmHg)'].mean()
    flight2_analysis_df['average_MAP (mmHg)'] = [average_MAP] + [None] * (len(flight2_analysis_df) - 1)

    flight2_analysis_df['systole_duration (s)'] = (pd.to_datetime(flight2_analysis_df['diastolic_trough_times']) - pd.to_datetime(flight2_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_systole_duration = flight2_analysis_df['systole_duration (s)'].mean()
    flight2_analysis_df['average_systole_duration (s)'] = [average_systole_duration] + [None] * (len(flight2_analysis_df) - 1)

    flight2_analysis_df['diastole_duration (s)'] = (pd.to_datetime(flight2_analysis_df['diastolic_trough_times'].shift(-1)) - pd.to_datetime(flight2_analysis_df['dicrotic_notch_times'])).abs().dt.total_seconds()
    average_diastole_duration = flight2_analysis_df['diastole_duration (s)'].mean()
    flight2_analysis_df['average_diastole_duration (s)'] = [average_diastole_duration] + [None] * (len(flight2_analysis_df) - 1)

    flight2_analysis_df['compression_rate'] = (pd.to_datetime(flight2_analysis_df['systolic_peak_times'].shift(-1)) - pd.to_datetime(flight2_analysis_df['systolic_peak_times'])).abs().dt.total_seconds()
    average_time_interval = flight2_analysis_df['compression_rate'].mean()
    average_compression_rate = (1 / average_time_interval) * 60 if average_time_interval != 0 else None
    flight2_analysis_df['average_compression_rate (compressions/min)'] = [average_compression_rate] + [None] * (len(flight2_analysis_df) - 1)

    flight2_analysis_df.to_csv(f'{parabola_name}_analysis.csv', index=False)

    return flight2_analysis

# Plot systolic, diastolic, and dicrotic features for a single Flight 2 parabola
def plot_parabola_results(parabola_data, flight2_analysis, parabola_name):
    plt.figure(figsize=(16, 10))
    plt.plot(parabola_data['DateTimeUTC'], parabola_data['Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')
    plt.scatter(parabola_data.iloc[flight2_analysis['systolic_peak_indices']]['DateTimeUTC'], flight2_analysis['systolic_peaks (mmHg)'], color='red', label='Systolic Peaks')
    plt.scatter(parabola_data.iloc[flight2_analysis['diastolic_trough_indices']]['DateTimeUTC'], flight2_analysis['diastolic_troughs (mmHg)'], color='green', label='Diastolic Troughs')
    plt.scatter(parabola_data.iloc[flight2_analysis['dicrotic_notch_indices']]['DateTimeUTC'], flight2_analysis['dicrotic_notches (mmHg)'], color='orange', label='Dicrotic Notches')
    plt.xlabel('DateTimeUTC')
    plt.ylabel('Pressure (mmHg)')
    plt.title(f'{parabola_name.capitalize()}: Pressure Over Time with Systolic Peaks, Diastolic Troughs, and Dicrotic Notches')
    plt.legend()
    plt.grid(True)
    plt.show()

lower_bound = 30
upper_bound = 55 

# Analyze and plot all 4 cm compression parabolas in Flight 2
for i, parabola in enumerate(four_cm_parabolas_query2):
    parabola_data = isolate_parabola_data(Query2, parabola)
    parabola_name = f'4cm_parabola{i+1}'
    print(f"Analyzing parabola index: {i}, parabola name: 4cm_parabola{i+1}")
    parabola_analysis = analyze_parabola(parabola_data, lower_bound, upper_bound, parabola_name=parabola_name)
    plot_parabola_results(parabola_data, parabola_analysis, parabola_name=parabola_name)
    
# Analyze and plot all new 5 cm compression parabolas from Flight 2
for i, parabola in enumerate(five_cm_parabolas_query2):
    parabola_data = isolate_parabola_data(Query2, parabola)
    print(f"Analyzing parabola index: {i}, parabola name: 5cm_parabola{i+6}")

    parabola_analysis = analyze_parabola(parabola_data, lower_bound, upper_bound, parabola_name=f'5cm_parabola{i+6}')
    plot_parabola_results(parabola_data, parabola_analysis, parabola_name=f'5cm_parabola{i+6}')
    
# --- GENERATE CSV FILES FOR ANALYSIS --- #

# Build per-parabola summary files (Earth vs microgravity, 4 cm vs 5 cm) and merge them into a single dataset for downstream analysis.
def flight_analysis(file_patterns, num_files_list, output_file):
    flight_analysis_df = pd.DataFrame()

    for pattern, num_files in zip(file_patterns, num_files_list):
        for i in range(1, num_files + 1):
            
            # 5 cm parabolas use indices offset by +5 in their filenames
            if '5cm_parabola' in pattern:
                file_path = pattern.format(i=i+5)  
            else:
                file_path = pattern.format(i=i)
            
            # Only process files that exist on disk
            if os.path.exists(file_path):
                print(f"Reading file: {file_path}")  
                parabola_type_df = pd.read_csv(file_path)
                
                # Track the source file name for each row (without the .csv extension)
                parabola_source = os.path.splitext(os.path.basename(file_path))[0]  
                parabola_type_df['parabola_source'] = parabola_source  
                
                # Compute mean arterial pressure (MAP) per beat
                parabola_type_df['MAP (mmHg)'] = parabola_type_df['diastolic_troughs (mmHg)'] + \
                    (1/3) * (parabola_type_df['systolic_peaks (mmHg)'] - parabola_type_df['diastolic_troughs (mmHg)'])
                
                # Append rows from this parabola file to the full dataset
                flight_analysis_df = flight_analysis_df.append(parabola_type_df, ignore_index=True)
            else:
                print(f"File not found: {file_path}")  

    # List of metrics to compute per-parabola averages
    columns_to_average = [
        'systolic_peaks (mmHg)', 'diastolic_troughs (mmHg)', 'MAP (mmHg)',
        'systole_duration (s)', 'diastole_duration (s)', 'compression_rate', 
    ]
    
    # Compute summary metrics for each unique parabola source
    for parabola_source in flight_analysis_df['parabola_source'].unique():
        parabola_df = flight_analysis_df[flight_analysis_df['parabola_source'] == parabola_source]
        
        averages = {
            'average_systolic_pressure (mmHg)': parabola_df['systolic_peaks (mmHg)'].mean(),
            'average_diastolic_pressure (mmHg)': parabola_df['diastolic_troughs (mmHg)'].mean(),
            'MAP (mmHg)': parabola_df['MAP (mmHg)'].mean(),
            'average_systole_duration (s)': parabola_df['systole_duration (s)'].mean(),
            'average_diastole_duration (s)': parabola_df['diastole_duration (s)'].mean(),
            'average_compression_rate (compressions/min)': (1 / parabola_df['compression_rate']).mean() * 60,
        }

        # Store per-parabola averages in the first row of that parabola's block
        first_row_idx = flight_analysis_df[flight_analysis_df['parabola_source'] == parabola_source].index[0]
        for col, avg_val in averages.items():
            flight_analysis_df.at[first_row_idx, col] = avg_val
        
    # Export the per-parabola dataset for this configuration
    flight_analysis_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")  

# Process 4cm and 5cm parabolas
flight_analysis(['4cm_parabola{i}_analysis.csv'], [5], '4cm_flight_analysis.csv')
flight_analysis(['parabola{i}_analysis.csv', '5cm_parabola{i}_analysis.csv'], [4, 5], '5cm_flight_analysis.csv')

# Read per-parabola summaries and normogravity-based analyses
flight_analysis_5cm_df = pd.read_csv('5cm_flight_analysis.csv')
cluster_analysis_2_df = pd.read_csv('cluster2_analysis.csv')
flight_analysis_4cm_df = pd.read_csv('4cm_flight_analysis.csv')
cluster_analysis_1_df = pd.read_csv('cluster1_analysis.csv')

# Tag each dataset with its source (Earth vs microgravity; 4 cm vs 5 cm)
cluster_analysis_1_df['parabola_source'] = 'cluster1_analysis'
cluster_analysis_2_df['parabola_source'] = 'cluster2_analysis'
cluster_analysis_1_df['source'] = '4cm Earth'
flight_analysis_4cm_df['source'] = '4cm Microgravity'
cluster_analysis_2_df['source'] = '5cm Earth'
flight_analysis_5cm_df['source'] = '5cm Microgravity'

# Concatenate all tagged datasets into one master DataFrame and export into CSV
merged_df = pd.concat([flight_analysis_5cm_df, cluster_analysis_2_df, flight_analysis_4cm_df, cluster_analysis_1_df], ignore_index=True)
merged_df.to_csv('CRISiS_analysis.csv', index=False)
print(f"Data saved to CRISiS_analysis.csv")
