CODTECH IT SOLUTIONS INTERNSHIP
5TH JUNE 2025 --> 5TH JULY 2025

TASK 4 - 
Create a predictive model using scikit learn to classify or predict outcomes from a dataset (eg spam email detection)

Deliverable: a jupyter notebook showcasing the model’s implementation and evaluation

NAME - CHRYSL SHECKINA
INTERN ID - CT04DG328
DOMAIN - PYTHON PROGRAMMING
DURATION - 4 WEEKS
MENTOR - MS. NEELA SANTHOSH KUMAR



OUTPUT SCREENSHOTS

![alt text](images/Screenshot%202025-06-27%20230054.png)
![alt text](images/Screenshot%202025-06-27%20230116-1.png)
![alt text](images/Screenshot%202025-06-27%20230132.png)
![alt text](images/Screenshot%202025-06-27%20230146.png)
![alt text](images/Screenshot%202025-06-27%20230159.png)
![alt text](images/Screenshot%202025-06-27%20230213-1.png)
![alt text](images/Screenshot%202025-06-27%20230227.png)
![alt text](images/Screenshot%202025-06-27%20230239-1.png)
![alt text](images/Screenshot%202025-06-27%20230255.png)
![alt text](images/Screenshot%202025-06-27%20230308.png)
![alt text](images/Screenshot%202025-06-27%20230319.png)
![alt text](images/Screenshot%202025-06-27%20230330.png)
![alt text](images/Screenshot%202025-06-27%20230339.png)

This project focuses on building a predictive machine learning model to detect breast cancer base3d on real diagnostic data. Uding the Breast Cancer Wisconsin(Diagnostic) Data Set from Kaggle, the model aims to classify tumors as either benign(non-cancerous) or malignant(cancerous). The task falls under binary classification, which is a fundamental use in healthcare-focused machine learning. 

This project was developed using Python, Jupyter Notebook, and scikit-learn library.It is part of CodTech Internship Task that requires creating a complete predictive model pipeline -- from data loading and cleaning, to training, evaluating , and comparing different models. 

Dataset
The dataset used was obtained from Kaggle. It contains 569 samples of breat cancer diagnostic measurements with 30 numerical features computed from digitalized images of fine needle aspirate(FNA) of breat masses. 

Each sample is labeled as either:
- M(Malignant) -- Cancerous
- B(Benign) -- Non-cancerous

Key Colums:

- radius_mean, texture_mean, area_mean, etc. 
- diagnosis(target)

Workflow Steps
The notebook is structured step by step to reflect a real world ML pipeline:
1. Data Loading
The dataset (data.csv) is read into a pandas DataFrame and the first few rows are inspected. 
2. Data Cleaning and Preprocessing 
- Unnecessary columns such as 'id' and empty columns were dropped. 
- The target column 'diagnosis' was converted from categorical ("M","B") to binary values(1,0).
- Featured and labels were split. 
- The dataset was divided into training (80%) and testing(20%) sets. 
- StandardSCaler was applied to normalize feature values. 

3. Model Building 
A Random Forest Classifier was initially selected for its robustness and high performance on classification problems. The model was trained using the training dataset. 

4. Model Evaluation
Predictions were made on the test set, and evaluation metrics like accuracy , confusion matrix, and classification report were used. The model achieved over 97% accuracy, indicating excellent predictive performance. 

5. Model Comparison
Two more models-- Logistic Regression and Support Vector Mchine(SVM)-- were added and trained on the same data. Their performances were compared using accuracy scores and visualized in a bar chart.

6. ROC Curve & AUC Score
To further evaluate the models, ROC curves were plotted and the AUC scores were calculated. All models achieved AUC> 0.98, confirming that they effectively distinguish between malignant and benign cases. 

Technologies Used
- Python 
- Jupyter Notebook
- Pandas
- NumPy
- Sci-kit Learn
- Matplotlib
- Seaborn

This project demonstrated the full lifecycle of building a predictive model using real world healthcare data. Among the models tested, Random Forest provided the best overall performance in terms of accuracy and AUC. The visualizations like ROC curves and confusion matrices helped in clearly interpreting model performance. 

With proper tuning and deployment , suvh a model could serve as a valuable tool in assisting medical professionals with early diagnosis of breast cancer. However, it is important to note that machine learning models are meant to support, not replace, clinical decision making. 

Project Files
- breast_cancer_model.ipynb -- Full notebook with code and outputs
- data.csv -- Dataset used(from Kaggle)
- README.md -- Project description and instructions

