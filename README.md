# Federated Transfer Learning on Predictive Maintenance

In this study, the datasets have been downloaded from the links provided below:

Nasa turbofan get engine dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Microsoft Azure Predictive Maintenance dataset: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

The files in this project are as below:

1. preprocesing_NASA_turbofan.ipynb

   This file is for preprocessing the NASA dataset into a multivariate time series dataset, where in each time frame the machine would fail or not.

2. preprocesing_Azure.ipynb

     This file is for preprocessing the Microsoft Azure predictive maintenance dataset. The purpose of this file is to transform the raw data into a multivariate time series dataset where each time frame for each machine of the models would fail or not.

3. FTL_one_class.ipynb

     In this code, the unified approach experiment has been done where one of the Azure's clients is used for transfer learning in each run.

4. FTL_unified_train.ipynb

      In this code, the experiment using the unified approach has been done where a subset of all Azure's clients is used for transfer learning.

To run the code first run the following line

`pip install -r requirements.txt`

After that run *preprocesing_NASA_turbofan.ipynb* and *preprocesing_NASA_turbofan.ipynb* where they will produce the datasets for FTL. By running *FTL_one_class.ipynb* and *FTL_unified_train.ipynb* you will be able to see the whole FTL pipeline.

Also the python version must be 3.10
   
