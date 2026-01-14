# Overview of datasets

## Sequence datasets
1. `data/raw/summarise_simulation/2024_11_21_160955/tabulated_mutation_info.csv`
    - Main training dataset
    - Trained with dt0 = 0.001
    - Includes many mutations
    - Was re-generated in `notebooks/data/simulate_circuits/2025_02_11__13_56_51/config.json` (probably because of a bug?)
    - Ruggedness was generated in `notebooks/data/07_ruggedness/2025_02_07__15_33_55`
2. `data/raw/summarise_simulation/2024_11_27_145142/tabulated_mutation_info.csv`
    - Re-balanced version of dataset `2024_11_21_160955`
        - Uses lower dt0 = 0.01
    - Includes many mutations too
    - Resimulated with a different starting copy number and updated code `notebooks/data/simulate_circuits/2026_01_12__18_50_13/config.json`
3. `notebooks/data/simulate_circuits/2025_02_11__13_56_51/`
    - Re-simulated version of dataset `2024_11_21_160955`

## Parameter-sampled datasets 


