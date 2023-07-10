# Heart_Disease_Prediction
## Heart Disease Prediction using Logistic Regression
## Introduction:

World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. Cardiovascular disease or heart disease is the leading cause of death amongst women and men and amongst most racial/ethnic groups in the United States. Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease. From the CDC, roughly every 1 in 4 deaths each year are due to heart disease. The WHO states that human life style is the main reason behind this heart problem. Apart from this there are many key factors which warns that the person may/may not getting chance of heart disease.

The term heart disease is often used interchangeably with the term cardiovascular disease. Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke.The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This project intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.

## Solution:

The classification goal is to predict whether the patient has risk of future heart disease.

I have implemented Logistic Regression Model to predict Heart Disease.

What is Logistic Regression ?

Logistic Regression is a statistical and machine-learning techniques classifying records of a dataset based on the values of the input fields . It predicts a dependent variable based on one or more set of independent variables to predict outcomes . It can be used both for binary classification and multi-class classification.

BASE ON GIVEN FEATURE AND MESUREMENT PREDICT WHETHER PATIENT WILL HAVE HEART DISEASE OR NOT.

## Workflow of model

1. Data collection
2. Split Features and Target set
3. Train-Test split
4. Model Training
5. Model Evaluation
6. Predicting Results


## Data collection
The dataset is available on the Kaggle website, and it provides the patients’ information with over 1,000 records and 14 attributes.

Columns Information

   1. age
   2. sex
   3. Chest pain type (4 values)
   4. Resting blood pressure
   5. Serum cholestoral in mg/dl
   6. Fasting blood sugar > 120 mg/dl
   7. Resting electrocardiographic results (values 0,1,2)
   8. Maximum heart rate achieved
   9. Exercise induced angina
  10. Oldpeak = ST depression induced by exercise relative to rest
  11. The slope of the peak exercise ST segment
  12. Number of major vessels (0-3) colored by flourosopy
  13. Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

1.AGE:
age in years

2.SEX:
sex (1 = male; 0 = female)

3.CHEST PAIN TYPE:
Most of the chest pain causes are not dangerous to health, but some are serious, while the least cases are life-threatening.[TA: typical angina(1), ATA: Atypical angina(2), NAP: non-anginal pain(3), ASY: asymptomatic (4) ]

4.RESTING BLOOD PRESSURE:
Blood pressure tells a lot about your general health. High blood pressure or hypertension can lead to several heart related issues and other medical conditions. Uncontrolled high blood pressure can lead to stroke.

5.SERUM CHOLESTEROL:
A person’s serum cholesterol level represents the amount of total cholesterol in their blood. A person’s serum cholesterol level comprises the amount of high-density lipoprotein (HDL), low-density lipoprotein (LDL), and triglycerides in the blood. Triglycerides are a type of fat bundled with cholesterol.

6.FASTING BLOOD SUGAR:
Your Fasting blood sugar level of 120 is a High Fasting blood sugar level. If your Fasting blood sugar is in between 74 mg/dL and 99 mg/dL, then you need not worry as 74-99 mg/dL is the normal range for Fasting blood sugar. But if your Fasting blood sugar is lesser or greater than the above values, then there may be some problem in your body. (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

7.RESTING EKG/ECG RESULT:
The electrocardiogram (ECG or EKG) is a test that measures the heart’s electrical activity, and a resting ECG is administered when the patient is at rest. It involves noninvasive recording with adhesive skin electrodes placed on specially prepared spots on the skin, and it plots out the heart's activity on a graph. It is used to determine the health of the heart and circulatory system and to help diagnose issues with associated body systems.[0: normal, 1:having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2:showing probable or definite left ventricular hypertrophy by Estes’ criteria]

8.MAX HEART RATE:
It has been shown that an increase in heart rate by 10 beats per minute was associated with an increase in the risk of cardiac death by at least 20%, and this increase in the risk is similar to the one observed with an increase in systolic blood pressure by 10 mm Hg.[Average heart rate: 60 to 100 bpm]

9.EXERCISE INDUCED ANGINA:
Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood.[0: no, 1: yes]

10. oldpeak_eq_st_depression:
oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms

11.SLOPE OF PEAK EXERCISE ST SEGMENT:
While a high ST depression is considered normal & healthy. The “ slope ” hue, refers to the peak exercise ST segment, with values: 1: upsloping, 2: flat, 3: down-sloping). Both positive & negative heart disease patients exhibit equal distributions of the 3 slope categories.

12.NUM OF MAJOR VESSELS:
Major Blood Vessels of the Heart: Blood exits the right ventricle through the pulmonary trunk artery. Approximately two inches superior to the base of the heart, this vessel branches into the left and right pulmonary arteries, which transport blood into the lungs.[number of major vessels: 0 to 3]

13.THAL:
A blood disorder called thalassemia,[normal, reversible defect, fixed defect]

## Libraries Used -

   1. Pandas (for data manipulation)
   2. Matplotlib (for data visualization)
   3. Seaborn (for data visualization)
   4. Scikit-Learn (for data modeling)

## Contents:

 1. Importing the required libraries.
 2. Importing and Reading the dataset.
 3. Exploratory Data Analysis (EDA)
 4. Data-Preprocessing
 5. Data Visualization
 6.Data Modeling
    - Separating the data into features and target variable.
    - Splitting the data into training and test sets.
    - Modeling/ Training the data
    - Predicting the data
    - Calculating the prediction scores
    - Getting the model's accuracy


## Dependencies

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    %matplotlib inline
    import seaborn as sns
    
## EDA Observations:
-IMPACT OF SLOPE OF PEAK TO HEART DISEASE:

   If slope of peak is flat the chance of heart disease is more than upsloping.
    Downslope st segment patient is also chance to get heart diseaase
    With the follwing observation we can say that slope of peak st segment has major impact on heart disease.

-IMPACT OF THAL TO HEART DISEASE:

  Normal blood disorder patient has less chance of heart disease than other thal.
  Reversible defect blood disorder has more chance of heart disease and fixed defect blood disorder has 50-50 chance of heart disease.

-IMPACT OF CPT TO HEART DISEASE:

  If the patient have asymtomatic(4) chest paint the chance of heart disease is more high.
    Non-anginal pain(3),typical angina(1), Atypical angina(2) chest pain have less chances of heart disease.
    But all chest pain types are impacting heart disease.

-IMPACT OF FASTING BLOOD SUGAR TO HEART DISEASE:

  If fasting blood suagar is less than 120mg/dl the chance of heart disease is high.
    If fasting blood sugar is greter than 120mg/dl the chace of heart disease is slightly less.

-IMPACT OF SEX TO HEART DISEASE:

  Male patient has more chance of heart disease than female.

-IMPACT OF MAJOR VESSELS TO HEART DISEASE:

  If the major vessels is zero the chance of heart disease is less but zero major vessels are also chance of heart disease.
    1,2, and 3 major vessels have more high chances of heart disease.

-IMPACT OF EKG RESULT TO HEART DISEASE:

  If the ekg/ecg result is normal(0) the chance of heart disease is less.
    If ekg/ecg result is 1 the 100% patient has heart disease,
    2 ekg/ecg result is 50-50% chance of heart disease.

-IMPACT OF EXERCISE INDUCED ANGINA:

  If the patient has no chest pain the chance of heart disease is less.
    If patient has chest pain the chance of heart disease is more.

## Split Features and Target set

    X = heart_data.drop(columns = 'target', axis = 1)
    X.head()
    #now X contains table without target column which will help for training the dataset

## Train Test Split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 1 )

    #here we have test data size is 20 percent of total data which is evenly distributed with degree of randomness = 1

## Model Training

    model = LogisticRegression()
    model.fit(X_train.values, Y_train)

## Model Evaluation

    # accuracy of traning data
    # accuracy function measures accuracy of model

    X_train_prediction = model.predict(X_train.values)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    print("The accuracy of training data : ", training_data_accuracy)

    # accuracy of test data

    X_test_prediction = model.predict(X_test.values)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

## Predicting Results

    # input feature values
    input_data = (42,1,0,136,315,0,1,125,1,1.8,1,0,1)
    
    # change the input data into a numpy array 
    input_data_arr = np.array(input_data)
    
    # reshape the array to predict data for only one instance
    reshaped_array = input_data_aarr.reshape(1,-1)

## Printing Results

    # predicting the result and printing it

    prediction = model.predict(reshaped_array)
    
    print(prediction)
    
    if(prediction[0] == 0):
        print("The Patient has a healthy heart")
    
    else:
        print("The Patient has an unhealthy heart")

    print("The accuracy of test data : ", test_data_accuracy)
