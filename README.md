# Supervised Brain Lesion Classification BRaTS2021

This is the repository for a Supervised Brain Lesion Classification Model using 3D Convolutionals layers. The project was done in the module Bigdata in Bioinformatics in the Summer Term 2022 from the Ruhr-University-Bochum.

## How to run

1. Clone this Repo
2. Install the requirements (recommended Conda)
```conda create --name <env> --file requirements.txt```
3. Run data_processing.py to chunk the Nifti Images
4. Run train.py (Note: We didn't implement arguments, so you need to specify Paths in the scripts.)
5. Optionally, run evaluate.py to save predicted and target labels and then analysis.py to compute metrics and figures.

## Data

We used the BRaTS2021 Dataset

## Contact

E-Mail: malik.mueller@ruhr-uni-bochum.de