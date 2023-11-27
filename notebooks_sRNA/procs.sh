#!/bin/bash

# Set the main directory to the current directory
main_directory="notebooks_sRNA/data/04_distribution_mutational_robustness/mutations"

# Loop through all of the directories in the main directory
for directory in $(ls $main_directory); do
    rm -r "$main_directory/$directory/binding_rates_dissociation"
    # if "eqconstants" in $(ls $directory); then # [ -d "$directory/energies" ]; then
    #     echo "$directory/eqconstants"
    # fi
done