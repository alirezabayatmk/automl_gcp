#!/bin/bash

echo "Cloning the repository"
git clone https://github.com/alirezabayatmk/automl_gcp.git
cd automl_gcp

echo "Installing requirements"
pip install -r requirements.txt

echo "downloading datasets"
python datasets.py

echo "Preparing balanced datasets"
rm -rf data/deepweedsx/test_balanced/*
rm -rf data/deepweedsx/train_balanced/*
cp -r data/deepweedsx/test/* data/deepweedsx/test_balanced/
cp -r data/deepweedsx/train/* data/deepweedsx/train_balanced/
rm -rf data/deepweedsx/test_balanced/8
rm -rf data/deepweedsx/train_balanced/8

echo "making visualizations directory"
mkdir visualizations

echo "Running final script in the background"
nohup python final_from_optuna.py &

echo "Setup completed"
