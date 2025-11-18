# CC Default Prediction

## How to setup Environment
1. Install Anaconda and create new conda environment
2. Conda install uv
3. Run uv sync with conda activated

```bash
conda create -n example_env
conda activate example_env
conda install uv
uv sync --locked
```

## Problem Statement
The objective of this project is to predict whether client will be default or no in the next month using up to 6 months past data. The class was imbalance with only 22% are default. To predict this cases 3 model trained using available data, that are:
1. Logistic Regression
2. Random Forest Classifier
3. SVC

All the details can be found on `/notebook/notebook.ipynb`

## Docker Deployment
You can run the docker and test the script using this command below, with directory in main directory, this is to build docker images
```bash
docker build -t predict-default:latest .
```
after that you can run the images with this script
```bash
docker run -p 9696:9696 predict-default:latest
```
And you can test the prediction in web services using notebook that available in `/notebook/test_endpoint.ipynb`