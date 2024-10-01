# Adaptation of MIMIC for MELART

This repository contains the adaptation of the original [MIMIC Repository](https://github.com/pengfei-luo/MIMIC) for the experiments using the MELArt dataset.

Clone the original repository, prepare the environment accordingly and copy the files from this repository to the original one.

Set the paths in the configuration files of the `config` folder to the correct paths in your system.

Run the experiments using the `run.sh` script.

```
./run.sh <config_file> <seed>
```

For example, to run the experiments without images using the seed 0:

```
./run.sh melart_no_imges 42
```

## Changes

- `run.sh`, `codes/main.py` and `codes/utils/functions.py`: Added optional argument with the seed for the random number generator.
- `config`: Configuration files for the experiments
    - `melart_no_candidate_img.yaml`: Experiments without using the entity images for the candidate entities.
    - `melart_no_imges.yaml`: Experiments without using any images.
    - `melart.yaml`: Experiments using the entity images if available.
