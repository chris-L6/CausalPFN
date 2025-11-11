conda activate reproduce
echo starting...
python linear_backdoor_bin_optimization.py
conda deactivate
echo "finished"
