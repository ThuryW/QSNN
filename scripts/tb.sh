#!/bin/bash

output_dir=./log

exp_dirs=($(ls -td "$output_dir"/*/))

if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "[ERROR] No experiment directories found in '$output_dir'."
    exit 1
fi

echo "Found the following experiment directories:"
for i in "${!exp_dirs[@]}"; do
    echo "[$i] ${exp_dirs[$i]}"
done

read -p "Select a directory to launch TensorBoard [default: 0]: " selected_index
selected_index=${selected_index:-0}

if ! [[ "$selected_index" =~ ^[0-9]+$ ]] || [ "$selected_index" -ge "${#exp_dirs[@]}" ]; then
    echo "[ERROR] Invalid selection."
    exit 1
fi

selected_dir="${exp_dirs[$selected_index]}/tensorboard"
echo "Launching TensorBoard with logdir: $selected_dir"

tensorboard \
    --logdir "$selected_dir" \
    --host 127.0.0.1 \
    --port 6006