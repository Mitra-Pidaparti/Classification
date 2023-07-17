# Classification
## Introduction
<img align="right" src="https://github.com/Mitra-Pidaparti/Classification/assets/110911635/82f04a85-bd78-4282-af97-d50e257d6499" width="380">   

- The fundamental objective of classification is to make `predictions about the category or class` (represented by y) based on given inputs (represented by x).
- It is one of the most widely used techniques in machine learning, with a broad array of applications, including sentiment analysis, ad targeting, spam detection, risk assessment, medical diagnosis, and image classification.

![Dogs vs cats](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/da6a5b78-14e2-4264-a443-110bf933db3a)

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

For a better understanding of Confusion Matrix: [Confusion Matrix](https://youtu.be/Kdsp6soqA7o)

  Now, here are some metrics for evaluation that use the above
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/836327cd-f818-4e39-9e3d-9f9e8733e218)

**When to use which metric?**
- Accuracy is obviously the most straight- forward as it tells us the fraction of how many predictions were right
but accuracy is not necessarily the best. Check this article out: [Accuracy, Precision, Recall or F1?](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)




## Implementation
Lets look at a step by step implmentation to classify animals as dogs or cats

#### Importing Dependencies and Preparing Directories
```{python}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Installing visualkeras
!pip install visualkeras   
import visualkeras
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/kaggle/input/')
os.listdir()
```
#### Setting Data Directories
```{python}
train_dir = '/kaggle/input/cats-and-dogs-image-classification/train'
test_dir = '/kaggle/input/cats-and-dogs-image-classification/test'
```
####  Creating Image Generators for Train and Test Data
- When using the image_dataset_from_directory function, it automatically assigns labels to the images based on the subdirectory structure. Each subdirectory represents a different class or category, and the function assigns a unique label to each class.
```{python}
from keras.utils import image_dataset_from_directory
train_generator = image_dataset_from_directory(train_dir, image_size=(64, 64), batch_size=32)
test_generator = image_dataset_from_directory(test_dir, image_size=(64, 64), batch_size=32)
```
#### Visualizing Samples from the Train Dataset
```{python}
# showing only the first 10 samples of our training data set
plt.figure(figsize=(10, 10))
for images, labels in test_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/9f9c9a09-8b3a-4dd9-bec8-681b660e9710)




#### Building the CNN Model
```{python}
# building CNN
model = keras.Sequential([
    # Conv layer 1:
    keras.layers.Conv2D(128, (3, 3), input_shape=(64,64,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    
    # Conv layer 2:
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    
    # Conv layer 3:
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    
    keras.layers.Flatten(),
    
    # fully connected layers:
    keras.layers.Dense(units = 128, activation = 'relu'),
    keras.layers.Dense(units =1, activation = 'sigmoid')
    
])

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics ='accuracy')

model.summary()

```
#### Visualizing the Model Architecture
```{python}
visualkeras.layered_view(model)
```


#### Training the model
```{python}
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1)
logs = model.fit(train_generator, epochs=50, validation_data=test_generator,validation_steps=2000/32, callbacks=[es, red_lr])
```
#### Plotting Training History
```{python}
plt.title('Training Log')
plt.plot(logs.history['loss'], label='Training Loss')
plt.plot(logs.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()
```
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/0b3de64c-a4d9-4697-a1d2-cf8fa7d997b7)

####  Evaluating on the Test Dataset
```{python}
res = model.evaluate(test_generator)
accuracy = res[1]
print(accuracy)
```
![image](https://github.com/Mitra-Pidaparti/Classification/assets/110911635/42718f2c-bb0c-431f-81ac-7effc05ab36a)

Not satisified with this accuracy? Well, there are ways you can improve it

- Of course, Adding more layers to your model increases its ability to learn your dataset’s features more deeply, helping it to extract more complex features from the image
- Getting more data: Duh :)
- Here is an additional method called Transfer Learning:
  


