# Run the following from CLI with source, e.g. 
# source run-python-script.sh <arg>

conda activate reproduce
echo starting...
python $1
conda deactivate
echo "finished"
