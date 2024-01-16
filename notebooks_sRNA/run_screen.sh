#!/bin/bash

# Number of iterations
num_iterations=1
file_to_modify="run_iterate_script.sh"
cp $file_to_modify "temp_mod_file.sh"

# Loop to run each iteration within a screen session
for ((i=0; i<=$num_iterations; i++)); do
    screen_name="screen_$i"

    screen -dmS $screen_name

    sed -i "s/i=0/i=$i/" "temp_mod_file.sh"

    # Send the commands to the screen session
    screen -S "$screen_name" -X stuff "bash -c '
        # Within the screen session, put the rest of your loop logic here
        # For example, you can execute your modified notebook script
        cd notebooks_sRNA
        echo \"1212\" | sudo bash temp_mod_file.sh
    '$(printf \\r)"

    echo "Started loop $i in screen $screen_name"
done

#!/bin/bash
