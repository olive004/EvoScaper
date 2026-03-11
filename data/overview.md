# Overview of datasets

## Sequence datasets
1. `data/raw/summarise_simulation/2024_11_21_160955/tabulated_mutation_info.csv`
    - Simulated with dt0 = 0.001
    - Includes many mutations
    - Was re-generated in `notebooks/data/simulate_circuits/2025_02_11__13_56_51/config.json` (probably because of a bug)
    - Ruggedness was simulated in `notebooks/data/07_ruggedness/2025_02_07__15_33_55`
2. `data/raw/summarise_simulation/2024_11_27_145142/tabulated_mutation_info.csv`
    - Re-balanced version of dataset `2024_11_21_160955`
        - Uses lower dt0 = 0.01
    - Includes many mutations too
    - Resimulated with a different starting copy number and updated code `notebooks/data/simulate_circuits/2026_01_12__18_50_13/config.json`

## Regenerated from source energies
1. `notebooks/data/simulate_circuits/2025_02_11__13_56_51/`
    - Main training dataset
    - Re-simulated version of dataset `2024_11_21_160955`
        - Some circuits were left out though
    - Ruggedness is the same as for `2024_11_21_160955`, since this was simulated with the same binding energies
2. `notebooks/data/simulate_circuits/2026_01_12__18_50_13/analytics.json`
    - Re-simulation of dataset `2024_11_27_145142`
    - starting copynumbers = 100
3. `notebooks/data/simulate_circuits/2026_01_14__22_38_47/analytics.json`
    - Re-simulation of dataset `2024_11_27_145142`
    - starting copynumbers = 200

## Parameter-sampled datasets 
1. `notebooks/data/simulate_circuits/2025_01_29__18_12_38/tabulated_mutation_info.json`
    - Simulated with sampled binding energies
    - Includes ruggedness
2. `data/raw/generate_sequence_batch/2025_09_20_103744`