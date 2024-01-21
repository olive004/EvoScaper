#!/bin/bash

# Specify the Jupyter notebook file
notebook_file="04_distribution_mutational_robustness.ipynb"

# Specify the Docker container name or ID
container_name="gcg"

i=23

cd notebooks_sRNA
cp "$notebook_file" "modified_notebook_$i.ipynb"
# jq --arg new_value "$i" '.cells[] | select(.cell_type == "code") | .source |= gsub("i = [0-9]+"; "i = " + $new_value)' "temp_notebook_$i.ipynb" > "modified_notebook_$i.ipynb"
pattern="i = 0"
sed -i "s/$pattern/i = $i/" "modified_notebook_$i.ipynb"

# Copy the modified notebook to the Docker container
docker cp "modified_notebook_$i.ipynb" "$container_name:workdir/notebooks_sRNA/modified_notebook_$i.ipynb"

# Execute the rest of the loop inside the Docker container
# docker exec -it "$container_name" bash -c "cd notebooks_sRNA && jupyter nbconvert --to notebook --inplace --execute modified_notebook_$i.ipynb"

docker exec -it "$container_name" bash -c "cd notebooks_sRNA && jupyter nbconvert --to notebook --execute "modified_notebook_$i.ipynb" --output="dmodified_notebook_$i.ipynb" --ExecutePreprocessor.timeout=-1"


# Create a temporary copy of the notebook
# cp "$notebook_file" "temp_notebook_$i.ipynb"

# Use jq to modify the variable in the notebook
# jq --arg new_value "$i" '.cells[] | select(.cell_type == "code") | .source |= gsub("i = 0"; "i = " + $new_value)' "temp_notebook_$i.ipynb" > "modified_notebook_$i.ipynb"

# Use jq to modify the variable in the notebook
# jq --argjson new_value $i '.cells[] | select(.cell_type == "code") | .source |= gsub("your_variable = [0-9]+"; "your_variable = " + ($new_value | tostring))' "temp_notebook_$i.ipynb" > "modified_notebook_$i.ipynb"

# Convert the modified notebook back to the original format
# jupyter nbconvert --to notebook --inplace --execute "modified_notebook_$i.ipynb"

# jupyter nbconvert --to notebook --execute "modified_notebook_$i.ipynb" --output="modified_notebook_$i.ipynb" --ExecutePreprocessor.timeout=-1

# Optional: Clean up temporary files
# rm "temp_notebook_$i.ipynb"  # "modified_notebook_$i.ipynb"
# done
