export PYTHONPATH=$(pwd)
echo "Device is ${1}. Training on task ${2}";
#CUDA_VISIBLE_DEVICES=${1} python -u ./codes/main.py --config "./config/${2}.yaml"
# Check if the third argument is provided
if [ -n "$3" ]; then
    # If the third argument is provided, include it in the Python call
    CUDA_VISIBLE_DEVICES=${1} python -u ./codes/main.py --config "./config/${2}.yaml" --seed $3
else
    # If the third argument is not provided, make the Python call without it
    CUDA_VISIBLE_DEVICES=${1} python -u ./codes/main.py --config "./config/${2}.yaml"
fi