# Cardiopulmonary resuscitation (CPR) for human spaceflight: A high-fidelity CPR simulator for hypogravity environments

## Abstract

With the emergence of long-duration space travel, providing specialized healthcare to astronauts is crucial for their survival. Prolonged space exploration missions pose a major concern due to the heightened risk of medical emergencies, such as sudden cardiac arrest. While several cardiopulmonary resuscitation (CPR) methods are proposed, their reliability and effectiveness remain uncertain, as none of these methods have been validated through physiological metrics. To address this gap, a high-fidelity CPR simulator was developed to simulate blood circulation and deliver real-time hemodynamic feedback. Herein, we show that in normogravity, the CPR simulator generated accurate compression-decompression waveforms that aligned with published animal and test bench studies. Hypogravity environments resulted in significantly elevated systolic, diastolic, mean, and pulse pressures compared to normogravity, while mechanical CPR compressions remained stable. These findings highlight the critical importance of internal physiological responses in evaluating CPR effectiveness under spaceflight conditions. The CPR simulator proves to be an effective tool for evaluating the performance of different CPR methods in hypogravity. Thus, the current study provides a foundational step toward supporting the identification and validation of a gold standard CPR protocol, contributing to the complex health challenges surrounding long-duration spaceflight.

## Quickstart

1. **Set up your environment**

   You can use your base Python environment or create a conda environment (Python 3.9+):

   ```
   conda create -n cpr_env python=3.9
   conda activate cpr_env
   ```

2. **Set the working directory**

    ```
    git clone <https://github.com/lordzoe/CPR-Simulator-Hypogravity.git>
    cd CPR-Simulator-Hypogravity
    ```

3. **Install dependencies**

    ```
    conda install pip
    pip install pandas numpy matplotlib scipy
    ```

4. **Run the analysis**

    ```
    python data_analysis.py
    ```

This will generate all intermediate analysis CSVs and the final merged dataset used in the manuscript.

---

## Usage

1. Make sure the following input files are present in the `raw data/` folder:

    - `raw data/mCPR_flight_data.csv` – Parabolic flight (hypogravity) dataset
    - `raw data/mCPR_ground_data.csv` – Ground (normogravity) dataset

2. From `CPR-Simulator-Hypogravity`, run:

        python data_analysis.py

3. The script will:

    - Identify normogravity compression windows and compute hemodynamic metrics for 4 cm and 5 cm mCPR compressions.  
        -  Outputs will be saved in the `processed data/mCPR compressions normogravity` folder:  
        `four_cm_ground_analysis.csv`,  
        `five_cm_ground_analysis.csv`

    - Detect hypogravity compression windows and compute per-parabola hemodynamic metrics for 4 cm and 5 cm mCPR compressions.  
        -  Outputs will be saved in the `processed data/mCPR compressions hypogravity` folder:  
        `four_cm_hypogravity_parabola_{i}_analysis.csv` for i = 1... 5,  
        `five_cm_hypogravity_parabola_{i}_analysis.csv` for i = 6... 10

    - Aggregate per-parabola hemodynamic metrics by compression depth.  
        -  Outputs will be saved in the `processed data/mCPR compressions hypogravity` folder:  
        `four_cm_hypogravity_analysis.csv`,  
        `five_cm_hypogravity_analysis.csv`

    - Merge normogravity and hypogravity datasets into a master file used for statistics and figures.  
        -  Outputs will be saved in the `processed data` folder:  
        `mCPR_compressions_analysis.csv`

Diagnostic figures (pressure vs. time with detected systolic peaks, diastolic troughs, and dicrotic notches) are generated for visual verification.

---

## Community Support

We encourage users to contribute by:

- Reporting issues or bugs via the GitHub **Issues** tab.
- Submitting feature requests or suggestions for improving the analysis pipeline.
- Sharing adaptations of the code for other CPR simulators, gravitational conditions, or experimental designs.

For scientific or methodological questions regarding the CPR simulator or study design, please contact the authors at 22zvl@queensu.ca.

---

## Citing CPR-Simulator-Hypogravity

If you use this code or the associated CPR hypogravity methodology in your research, please cite our manuscript (details to be updated upon publication).

**Manuscript**  
Citation details will be added here once the article is available.

```
@article{cpr_simulator_hypogravity_2025,
    title   = {Cardiopulmonary resuscitation (CPR) for human spaceflight: A high-fidelity CPR simulator for hypogravity environments},
    author  = {Lord, Z., Andrande, C., Leroux, L., Kadem, L.},
    journal = {...},
    year    = {...},
    volume  = {...},
    number  = {...},
    pages   = {...}
}
```