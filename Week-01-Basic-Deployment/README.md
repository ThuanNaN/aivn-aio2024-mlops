# Week 1: Basic Deployment

In this first week, we build a simple application (web application) using 2 python packages: Streamlit and Gradio for serving 2 our models.

## Requirements

To run the code, you need to install the required packages. You can install the required packages using the following command:

```bash
conda create -n aio-mlops-w1 python=3.9.11 --y
conda activate aio-mlops-w1
pip install -r requirements.txt
```

## Streamlit

Before deploying the model, we need to train the model and save it. We will use the Random Forest Regressor model to predict the price of a house based on the number of bedrooms and bathrooms and the area of the house.

Follow the steps below to deploy the model using Streamlit:

### Step 1: Training a Model

The base code training is available in the notebook folder. You can run the code and train the model.

### Step 2: Saving the Model

Run the cell ***Save the model*** to save the model in the model folder.

```python
import pickle

with open("rf_regressor.pkl", "wb") as model_file:
    pickle.dump(rf_regressor, model_file)
```

### Step 3: Building a Web Application

First you need copy the model file (checkpoint) in notebook folder to this directory.

To start the streamlit application, run the following command in the terminal:

```bash
streamlit run streamlit_app.py
```

## Gradio

In the second example, we will use the Gradio package to deploy a model that predicts the text occured in the image. This task is called Optical Character Recognition (OCR).

We will use the pre-trained model from the EasyOCR package so we don't need to train the model. The model file (checkpoint) will be downloaded automatically when the application is run.

To start the Gradio application, run the following command in the terminal:

```bash
python gradio_app.py
```