# Classification
## Introduction
<img align="right" src="https://github.com/Mitra-Pidaparti/Classification/assets/110911635/82f04a85-bd78-4282-af97-d50e257d6499" width="380">   

- The fundamental objective of classification is to make `predictions about the category or class` (represented by y) based on given inputs (represented by x).
- It is one of the most widely used techniques in machine learning, with a broad array of applications, including sentiment analysis, ad targeting, spam detection, risk assessment, medical diagnosis, and image classification.


![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/a442485c-44fa-44e7-b82d-d6ca8ec4df8c)

#### Feature Extraction: 
In the classification of dogs and cats, feature extraction involves analyzing distinguishing characteristics such as ear texture, fur texture, snout, and nose structure. These features are identified and extracted to capture unique attributes that differentiate between the two animal classes.
#### Training and Learning: 
A machine learning algorithm is employed to learn from a carefully labeled dataset of dog and cat images, where each instance is associated with the corresponding class label. By training on this dataset, the algorithm discovers patterns and correlations between the extracted features and the respective animal classes.
#### Prediction: 
Once the model is trained, it can be utilized to classify new, unseen images of dogs and cats. By examining the distinct features of these images, the model applies its learned knowledge to accurately predict whether an image depicts a dog or a cat, enabling seamless and reliable classification.



## Binary Classification
- Binary classification is a common machine learning technique that categorizes inputs into two distinct categories, making it the simplest form of classification
- The following are a few binary classification applications, where the 0 and 1 columns are two possible classes for each observation:
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/85046681-b0cc-4554-bfab-c58cb7f27cde)
- Some popular model algorithms commonly used for binary classification tasks:
  
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Gradient Boosting models (e.g., XGBoost, LightGBM)
  
### Evaluation Metrics:
Now that the prediction of the class is done, how do we evaluate the model?

Before that lets learn some new terms: `True Positives, True Negatives, False Positives, False Negatives`:
Lets use the above Medical Diagnosis example:

<img align="right" src="https://github.com/Mitra-Pidaparti/Classification/assets/110911635/6efc7fa5-d636-43dc-9a2a-1b72700212f9" width="500"> 

- True Positive (TP): The patient is diseased and the model predicts "diseased"
- False Positive (FP): The patient is healthy but the model predicts "diseased"
- True Negative (TN): The patient is healthy and the model predicts "healthy"
- False Negative (FN): The patient is diseased and the model predicts "healthy"

  Now, lets check out some metrics which use the above:
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/836327cd-f818-4e39-9e3d-9f9e8733e218)









 









