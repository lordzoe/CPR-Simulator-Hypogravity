import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

## --- VISCOSITY CALCULATOR --- ##

# Compute the density and viscosity of a glycerol-water mixture given temperature and volumes
def calculate_glycerol_properties(T, waterVol, glycerolVol):
    glycerolDen = (1273.3 - 0.6121 * T) / 1000  # Density of Glycerol (g/cm3)
    waterDen = (1 - math.pow(((abs(T - 4)) / 622), 1.7))  # Density of water (g/cm3)

    glycerolMass = glycerolDen * glycerolVol
    waterMass = waterDen * waterVol
    totalMass = glycerolMass + waterMass
    mass_fraction = glycerolMass / totalMass
    vol_fraction = glycerolVol / (glycerolVol + waterVol)

    print("Mass fraction of mixture =", round(mass_fraction, 5))
    print("Volume fraction of mixture =", round(vol_fraction, 5))

    # Density calculator
    contraction_av = 1 - math.pow(3.520E-8 * ((mass_fraction * 100)), 3) + math.pow(1.027E-6 * ((mass_fraction * 100)), 2) + 2.5E-4 * (mass_fraction * 100) - 1.691E-4
    contraction = 1 + contraction_av / 100
    density_mix = (glycerolDen * vol_fraction + waterDen * (1 - vol_fraction)) * contraction
    print("Density of mixture =", round(density_mix, 5), "g/cm3")

    # Viscosity calculator
    glycerolVisc = 0.001 * 12100 * numpy.exp((-1233 + T) * T / (9900 + 70 * T))
    waterVisc = 0.001 * 1.790 * numpy.exp((-1230 - T) * T / (36100 + 360 * T))

    a = 0.705 - 0.0017 * T
    b = (4.9 + 0.036 * T) * numpy.power(a, 2.5)
    alpha = 1 - mass_fraction + (a * b * mass_fraction * (1 - mass_fraction)) / (a * mass_fraction + b * (1 - mass_fraction))
    A = numpy.log(waterVisc / glycerolVisc)

    viscosity_mix = glycerolVisc * numpy.exp(A * alpha)

    print("Viscosity of mixture =", round(viscosity_mix, 5), "Ns/m2")

calculate_glycerol_properties(25.99332232, 1400, 2600)  # Flight 1
calculate_glycerol_properties(28.47262121, 1400, 2600)  # Flight 2
calculate_glycerol_properties(20, 1400, 2600)           # Calibration

## --- CALIBRATION: NORMOGRAVITY CONDITIONS (NULL GLYCEROL) --- ##

file_path = r"data/2023-08-04_14-57-32_on-ground-null-glycerol.txt"
null_glycerol = pd.read_csv(file_path, skiprows=47, delim_whitespace=True, names=["Date", "Time", "Pressure (mmHg)"])

# Combine Date and Time into a single datetime column
null_glycerol['DateTime (EDT)'] = pd.to_datetime(null_glycerol['Date'] + ' ' + null_glycerol['Time'])
null_glycerol.drop(columns=['Date', 'Time'], inplace=True)

# Define first normogravity compression window (continuous compressions between 14:57:49 and 14:59:31)
# Compute mean and minimum pressures over this window
start_time_cluster1 = pd.to_datetime('2023-08-04 14:57:49')
end_time_cluster1 = pd.to_datetime('2023-08-04 14:59:31')
filtered_data_cluster1 = null_glycerol[(null_glycerol['DateTime (EDT)'] >= start_time_cluster1) & (null_glycerol['DateTime (EDT)'] <= end_time_cluster1)]
average_pressure_cluster1 = filtered_data_cluster1['Pressure (mmHg)'].mean()
negative_pressure_cluster1 = filtered_data_cluster1['Pressure (mmHg)'].min()
print(f"Average cluster 1 pressure value from onset until {end_time_cluster1}: {average_pressure_cluster1:.2f} mmHg")
print(f"Negative cluster 1 pressure value: {negative_pressure_cluster1:.2f} mmHg")

# Define second normogravity compression window (continuous compressions between 15:00:09 and 15:01:52)
# Compute mean and minimum pressures over this window
start_time_cluster2 = pd.to_datetime('2023-08-04 15:00:09')
end_time_cluster2 = pd.to_datetime('2023-08-04 15:01:52')
filtered_data_cluster2 = null_glycerol[(null_glycerol['DateTime (EDT)'] >= start_time_cluster2) & (null_glycerol['DateTime (EDT)'] <= end_time_cluster2)]
average_pressure_cluster2 = filtered_data_cluster2['Pressure (mmHg)'].mean()
negative_pressure_cluster2 = filtered_data_cluster2['Pressure (mmHg)'].min()
print(f"Average cluster 2 pressure value from onset until {end_time_cluster2}: {average_pressure_cluster2:.2f} mmHg")
print(f"Negative cluster 2 pressure value: {negative_pressure_cluster2:.2f} mmHg")

# Define boolean masks for the two normogravity compression windows in the null_glycerol dataset
cluster1_indices = (null_glycerol['DateTime (EDT)'] >= '2023-08-04 14:57:49') & (null_glycerol['DateTime (EDT)'] <= '2023-08-04 14:59:31')
cluster2_indices = (null_glycerol['DateTime (EDT)'] > '2023-08-04 15:00:09') & (null_glycerol['DateTime (EDT)'] <= '2023-08-04 15:01:52')

# Detect compression peaks within each normogravity compression window
cluster1_peaks, _ = find_peaks(null_glycerol.loc[cluster1_indices, 'Pressure (mmHg)'], height=18, distance=25)
cluster2_peaks, _ = find_peaks(null_glycerol.loc[cluster2_indices, 'Pressure (mmHg)'], height=18, distance=25)

# Compute mean peak amplitude within each normogravity compression window
average_peak_cluster1 = null_glycerol.loc[cluster1_indices].iloc[cluster1_peaks]['Pressure (mmHg)'].mean()
average_peak_cluster2 = null_glycerol.loc[cluster2_indices].iloc[cluster2_peaks]['Pressure (mmHg)'].mean()
print(f"Average peak value for the first cluster of spikes: {average_peak_cluster1:.2f} mmHg")
print(f"Average peak value for the second cluster of spikes: {average_peak_cluster2:.2f} mmHg")

# Convert string time limits to datetime objects for plotting each normogravity compression window
cluster1_xlim_start = pd.to_datetime('2023-08-04 14:57:49')
cluster1_xlim_end = pd.to_datetime('2023-08-04 14:59:31')
cluster2_xlim_start = pd.to_datetime('2023-08-04 15:00:09')
cluster2_xlim_end = pd.to_datetime('2023-08-04 15:01:52')

# Create subplots for visual inspection of both normogravity compression windows
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot first normogravity compression window 
ax1.plot(null_glycerol.loc[cluster1_indices, 'DateTime (EDT)'], 
         null_glycerol.loc[cluster1_indices, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax1.scatter(null_glycerol.loc[cluster1_indices].iloc[cluster1_peaks]['DateTime (EDT)'],
            null_glycerol.loc[cluster1_indices].iloc[cluster1_peaks]['Pressure (mmHg)'], 
            color='red', label='Detected Peaks - Cluster 1')

ax1.set_xlim([cluster1_xlim_start, cluster1_xlim_end])  #
ax1.set_xlabel('DateTime (EDT)')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('Cluster 1: Pressure Over Time with Detected Peaks')
ax1.legend()
ax1.grid(True)

# Plot second normogravity compression window
ax2.plot(null_glycerol.loc[cluster2_indices, 'DateTime (EDT)'], 
         null_glycerol.loc[cluster2_indices, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax2.scatter(null_glycerol.loc[cluster2_indices].iloc[cluster2_peaks]['DateTime (EDT)'],
            null_glycerol.loc[cluster2_indices].iloc[cluster2_peaks]['Pressure (mmHg)'], 
            color='green', label='Detected Peaks - Cluster 2')

ax2.set_xlim([cluster2_xlim_start, cluster2_xlim_end])  
ax2.set_xlabel('DateTime (EDT)')
ax2.set_ylabel('Pressure (mmHg)')
ax2.set_title('Cluster 2: Pressure Over Time with Detected Peaks')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

## --- CALIBRATION FOR NORMOGRAVITY CONDITIONS (NO NULL GLYCEROL) --- ##

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = r"data/2023-08-04_14-34-16_on-ground-no-null-glycerol.txt"
no_null_glycerol = pd.read_csv(file_path, skiprows=47, delim_whitespace=True, names=["Date", "Time", "Pressure (mmHg)"])

# Combine Date and Time into a single datetime column
no_null_glycerol['DateTime (EDT)'] = pd.to_datetime(no_null_glycerol['Date'] + ' ' + no_null_glycerol['Time'])
no_null_glycerol.drop(columns=['Date', 'Time'], inplace=True)

# Define first normogravity compression window without null glycerol (continuous compressions between 14:35:28 and 14:37:11)
# Compute mean and minimum pressures over this window
start_time_cluster1_no_null = pd.to_datetime('2023-08-04 14:35:28')
end_time_cluster1_no_null = pd.to_datetime('2023-08-04 14:37:11')
filtered_data_cluster1_no_null = no_null_glycerol[(no_null_glycerol['DateTime (EDT)'] >= start_time_cluster1_no_null) & (no_null_glycerol['DateTime (EDT)'] <= end_time_cluster1_no_null)]
average_pressure_cluster1_no_null = filtered_data_cluster1_no_null['Pressure (mmHg)'].mean()
negative_pressure_cluster1_no_null = filtered_data_cluster1_no_null['Pressure (mmHg)'].min()
print(f"Average cluster 1 pressure value from onset until {end_time_cluster1_no_null}: {average_pressure_cluster1_no_null:.2f} mmHg")
print(f"Negative cluster 1 pressure value from onset until {end_time_cluster1_no_null}: {negative_pressure_cluster1_no_null:.2f} mmHg")

# Define second normogravity compression window without null glycerol (continuous compressions between 14:37:55 and 14:39:38).
# Compute mean and minimum pressures over this window.
start_time_cluster2_no_null = pd.to_datetime('2023-08-04 14:37:55')
end_time_cluster2_no_null = pd.to_datetime('2023-08-04 14:39:38')
filtered_data_cluster2_no_null = no_null_glycerol[(no_null_glycerol['DateTime (EDT)'] >= start_time_cluster2_no_null) & (no_null_glycerol['DateTime (EDT)'] <= end_time_cluster2_no_null)]
average_pressure_cluster2_no_null = filtered_data_cluster2_no_null['Pressure (mmHg)'].mean()
negative_pressure_cluster2_no_null = filtered_data_cluster2_no_null['Pressure (mmHg)'].min()
print(f"Average cluster 2 pressure value from onset until {end_time_cluster2_no_null}: {average_pressure_cluster2_no_null:.2f} mmHg")
print(f"Negative cluster 2 pressure value from onset until {end_time_cluster2_no_null}: {negative_pressure_cluster2_no_null:.2f} mmHg")

# Define boolean masks for the two normogravity compression windows in the no_null_glycerol dataset
cluster1_indices_no_null = (no_null_glycerol['DateTime (EDT)'] >= pd.to_datetime('2023-08-04 14:35:28')) & \
                           (no_null_glycerol['DateTime (EDT)'] <= pd.to_datetime('2023-08-04 14:37:11'))
cluster2_indices_no_null = (no_null_glycerol['DateTime (EDT)'] >= pd.to_datetime('2023-08-04 14:37:55')) & \
                           (no_null_glycerol['DateTime (EDT)'] <= pd.to_datetime('2023-08-04 14:39:38'))

# Detect compression peaks within each normogravity compression window
cluster1_peaks_no_null, _ = find_peaks(no_null_glycerol.loc[cluster1_indices_no_null, 'Pressure (mmHg)'], height=4, distance=25)
cluster2_peaks_no_null, _ = find_peaks(no_null_glycerol.loc[cluster2_indices_no_null, 'Pressure (mmHg)'], height=4, distance=25)

# Compute mean peak amplitude within each normogravity compression window
average_peak_cluster1_no_null = no_null_glycerol.loc[cluster1_indices_no_null].iloc[cluster1_peaks_no_null]['Pressure (mmHg)'].mean()
average_peak_cluster2_no_null = no_null_glycerol.loc[cluster2_indices_no_null].iloc[cluster2_peaks_no_null]['Pressure (mmHg)'].mean()
print(f"Average peak value for the first cluster of spikes: {average_peak_cluster1_no_null:.2f} mmHg")
print(f"Average peak value for the second cluster of spikes: {average_peak_cluster2_no_null:.2f} mmHg")

# Convert string time limits to datetime objects for plotting each normogravity compression window
cluster1_xlim_start_no_null = pd.to_datetime('2023-08-04 14:35:28')
cluster1_xlim_end_no_null = pd.to_datetime('2023-08-04 14:37:11') 
cluster2_xlim_start_no_null = pd.to_datetime('2023-08-04 14:37:55')
cluster2_xlim_end_no_null = pd.to_datetime('2023-08-04 14:39:38')  

# Create subplots for visual inspection of both normogravity compression windows (no null glycerol)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot first normogravity compression window
ax1.plot(no_null_glycerol.loc[cluster1_indices_no_null, 'DateTime (EDT)'], 
         no_null_glycerol.loc[cluster1_indices_no_null, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax1.scatter(no_null_glycerol.loc[cluster1_indices_no_null].iloc[cluster1_peaks_no_null]['DateTime (EDT)'],
            no_null_glycerol.loc[cluster1_indices_no_null].iloc[cluster1_peaks_no_null]['Pressure (mmHg)'], 
            color='red', label='Detected Peaks - Cluster 1')

ax1.set_xlim([cluster1_xlim_start_no_null, cluster1_xlim_end_no_null])
ax1.set_xlabel('DateTime (EDT)')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('Cluster 1: Pressure Over Time with Detected Peaks')
ax1.legend()
ax1.grid(True)

# Plot second normogravity compression window
ax2.plot(no_null_glycerol.loc[cluster2_indices_no_null, 'DateTime (EDT)'], 
         no_null_glycerol.loc[cluster2_indices_no_null, 'Pressure (mmHg)'], label='Pressure (mmHg)', color='blue')

ax2.scatter(no_null_glycerol.loc[cluster2_indices_no_null].iloc[cluster2_peaks_no_null]['DateTime (EDT)'],
            no_null_glycerol.loc[cluster2_indices_no_null].iloc[cluster2_peaks_no_null]['Pressure (mmHg)'], 
            color='green', label='Detected Peaks - Cluster 2')

ax2.set_xlim([cluster2_xlim_start_no_null, cluster2_xlim_end_no_null])
ax2.set_xlabel('DateTime (EDT)')
ax2.set_ylabel('Pressure (mmHg)')
ax2.set_title('Cluster 2: Pressure Over Time with Detected Peaks')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

## --- CALCULATE DIFFERENCES BETWEEN NORMOGRAVITY CONDITIONS --- ##

# Compute differences in mean pressures between the null_glycerol and no_null_glycerol normogravity conditions for each compression window
pressure_difference_cluster1 = average_pressure_cluster1 - average_pressure_cluster1_no_null
pressure_difference_cluster2 = average_pressure_cluster2 - average_pressure_cluster2_no_null
print(f"Average pressure value for the first cluster for null_glycerol: {average_pressure_cluster1:.2f} mmHg")
print(f"Average pressure value for the first cluster for no_null_glycerol: {average_pressure_cluster1_no_null:.2f} mmHg")
print(f"Average pressure value for the second cluster for null_glycerol: {average_pressure_cluster2:.2f} mmHg")
print(f"Average pressure value for the second cluster for no_null_glycerol: {average_pressure_cluster2_no_null:.2f} mmHg")
print(f"Difference (null_glycerol - no_null_glycerol) for the first cluster: {pressure_difference_cluster1:.2f} mmHg")
print(f"Difference (null_glycerol - no_null_glycerol) for the second cluster: {pressure_difference_cluster2:.2f} mmHg")

# Compute differences in mean peak pressures between normogravity conditions for the first compression window
peak_difference_cluster1 = average_peak_cluster1 - average_peak_cluster1_no_null
#print(f"Average peak value for the first cluster of spikes for null_glycerol: {average_peak_cluster1:.2f} mmHg")
#print(f"Average peak value for the first cluster of spikes for no_null_glycerol: {average_peak_cluster1_no_null:.2f} mmHg")
#print(f"Difference (null_glycerol - no_null_glycerol) for the first cluster: {peak_difference_cluster1:.2f} mmHg")

# Compute differences in mean peak pressures between normogravity conditions for the second compression window
peak_difference_cluster2 = average_peak_cluster2 - average_peak_cluster2_no_null
#print(f"Average peak value for the second cluster of spikes for null_glycerol: {average_peak_cluster2:.2f} mmHg")
#print(f"Average peak value for the second cluster of spikes for no_null_glycerol: {average_peak_cluster2_no_null:.2f} mmHg")
#print(f"Difference (null_glycerol - no_null_glycerol) for the second cluster: {peak_difference_cluster2:.2f} mmHg")

# Compute differences between the most negative pressure values for each normogravity compression window
negative_pressure_difference_cluster1 = negative_pressure_cluster1 - negative_pressure_cluster1_no_null
print(f"Negative pressure value for the first cluster for null_glycerol: {negative_pressure_cluster1:.2f} mmHg")
print(f"Negative pressure value for the first cluster for no_null_glycerol: {negative_pressure_cluster1_no_null:.2f} mmHg")
print(f"Negative pressure value for the second cluster for null_glycerol: {negative_pressure_cluster2:.2f} mmHg")
print(f"Negative pressure value for the second cluster for no_null_glycerol: {negative_pressure_cluster2_no_null:.2f} mmHg")
print(f"Difference (null_glycerol - no_null_glycerol) for the most negative pressure value in cluster 1: {negative_pressure_difference_cluster1:.2f} mmHg")
negative_pressure_difference_cluster2 = negative_pressure_cluster2 - negative_pressure_cluster2_no_null
print(f"Difference (null_glycerol - no_null_glycerol) for the most negative pressure value in cluster 2: {negative_pressure_difference_cluster2:.2f} mmHg")

file_path = r"data/"
filenames = ["Query1.csv", "Query2.csv", "Query3.csv"]

#for filename in filenames:
    #query_data = pd.read_csv(file_path + filename)
    
    # Add the pressure_difference to the 'Pressure (Pa)' column
   # query_data['Pressure (mmHg)'] += pressure_difference

    # Save the updated file
    #query_data.to_csv(file_path + filename, index=False)

#print("Pressure values updated in Query1.csv, Query2.csv, Query3.csv successfully.")

file_path = r"data/"

query1 = pd.read_csv(file_path + "Query1.csv")
query2 = pd.read_csv(file_path + "Query2.csv")
query3 = pd.read_csv(file_path + "Query3.csv")

# Calibrate no_null_glycerol baseline using null_glycerol (align lowest pressure in second normogravity window to zero)
baseline_offset_no_null = abs(negative_pressure_cluster2_no_null)
no_null_glycerol['Pressure (mmHg)'] += baseline_offset_no_null

# Recompute pressure differences after baseline alignment
pressure_difference_cluster1 = average_pressure_cluster1 - average_pressure_cluster1_no_null
pressure_difference_cluster2 = average_pressure_cluster2 - average_pressure_cluster2_no_null
print(f"Pressure difference for Cluster 1: {pressure_difference_cluster1:.2f} mmHg")
print(f"Pressure difference for Cluster 2: {pressure_difference_cluster2:.2f} mmHg")

# Calibrate Query1, Query2, Query3 using the pressure differences derived from normogravity calibrations
queries = [query1, query2, query3]
query_names = ["Query1", "Query2", "Query3"]

for query, filename in zip(queries, query_names):
    # Calibrate each query baseline by shifting minimum pressure to zero
    negative_pressure_query = query['Pressure (mmHg)'].min()
    baseline_offset_query = abs(negative_pressure_query)
    query['Pressure (mmHg)'] += baseline_offset_query

    # Apply normogravity-derived pressure offsets
    if filename == "Query1":
        query['Pressure (mmHg)'] += pressure_difference_cluster2
    elif filename == "Query2":
        query['DateTimeUTC'] = pd.to_datetime(query['DateTimeUTC'])
        
        # Apply first normogravity compression window offset to the corresponding UTC windows
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:16:43.006')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:17:00.294')), 'Pressure (mmHg)'] += pressure_difference_cluster1
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:27:08.502')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:27:27.494')), 'Pressure (mmHg)'] += pressure_difference_cluster1
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:41:08.006')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:41:25.894')), 'Pressure (mmHg)'] += pressure_difference_cluster1
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:49:55.806')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:50:14.598')), 'Pressure (mmHg)'] += pressure_difference_cluster1
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:58:49.806')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:59:07.894')), 'Pressure (mmHg)'] += pressure_difference_cluster1

        # Apply second normogravity compression window offset to the corresponding UTC windows
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:22:10.606')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:22:29.894')), 'Pressure (mmHg)'] += pressure_difference_cluster2
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:33:33.902')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:33:51.294')), 'Pressure (mmHg)'] += pressure_difference_cluster2
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:45:59.902')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:46:18.998')), 'Pressure (mmHg)'] += pressure_difference_cluster2
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 16:54:09.102')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 16:54:28.198')), 'Pressure (mmHg)'] += pressure_difference_cluster2
        query.loc[(query['DateTimeUTC'] >= pd.to_datetime('2023-08-04 17:02:47.406')) & (query['DateTimeUTC'] <= pd.to_datetime('2023-08-04 17:03:05.894')), 'Pressure (mmHg)'] += pressure_difference_cluster2

    elif filename == "Query3":
        query['DateTimeEDT'] = pd.to_datetime(query['DateTimeEDT'])
        query.loc[(query['DateTimeEDT'] >= pd.to_datetime('2023-08-04 14:35:28')) & (query['DateTimeEDT'] <= pd.to_datetime('2023-08-04 14:37:11')), 'Pressure (mmHg)'] += pressure_difference_cluster1
        query.loc[(query['DateTimeEDT'] >= pd.to_datetime('2023-08-04 14:37:55')) & (query['DateTimeEDT'] <= pd.to_datetime('2023-08-04 14:39:38')), 'Pressure (mmHg)'] += pressure_difference_cluster2

    # Save the updated file
    query.to_csv(file_path + filename + ".csv", index=False)
    print(f"Calibrated {filename}:")
    print(query)

print("Pressure values updated in Query1.csv, Query2.csv, Query3.csv successfully.")