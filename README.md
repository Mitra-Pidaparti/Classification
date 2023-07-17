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






### Importing Dependencies and Preparing Directories
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
### Setting Data Directories
```{python}
train_dir = '/kaggle/input/cats-and-dogs-image-classification/train'
test_dir = '/kaggle/input/cats-and-dogs-image-classification/test'
```
### Data Augmentation
```{python}
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
```
###  Creating Image Generators for Train and Test Data
```{python}
from keras.utils import image_dataset_from_directory
train_generator = image_dataset_from_directory(train_dir, image_size=(64, 64), batch_size=32)
test_generator = image_dataset_from_directory(test_dir, image_size=(64, 64), batch_size=32)
```
### Visualizing Samples from the Train Dataset
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
### Building the CNN Model
```{python}
model = keras.Sequential([
    keras.layers.Conv2D(128, (3, 3), input_shape=(64, 64, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
```
### Visualizing the Model Architecture
```{python}
visualkeras.layered_view(model)
```
### Training the model
```{python}
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1)
mc = keras.callbacks.ModelCheckpoint('/kaggle/working/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

logs = model.fit(train_generator,
                 epochs=50,
                 validation_data=test_generator,
                 validation_steps=2000/32,
                 callbacks=[es, red_lr, mc])

```

```{python}
import matplotlib.pyplot as plt
plt.title('Training Log')
plt.plot(logs.history['loss'], label='Training Loss')
plt.plot(logs.history['accuracy'], label='Training Accuracy')
# plt.plot(logs.history['val_loss'], label='Validation Loss', linewidth=3)
# plt.plot(logs.history['val_accuracy'], label='Validation accuracy', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()
```
### Plotting Training History
```{python}
plt.title('Training Log')
plt.plot(logs.history['loss'], label='Training Loss')
plt.plot(logs.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()
```
###  Loading the Best Model and Evaluating on the Test Dataset
```{python}
best_model = keras.models.load_model('/kaggle/working/best_model.h5')

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```
### Predicting on the Test Dataset and Calculating Evaluation Metrics
```{python}
y_true = test_generator.classes
y_pred_probs = best_model.predict(test_generator).flatten()
y_pred = np.round(y_pred_probs)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
```
### Confusion Matrix
```{python}
cm = confusion_matrix(y_true, y_pred)
labels = ['Cat', 'Dog']

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```
###  Predicting on a Single Image
```{python}
img = keras.preprocessing.image.load_img(
    "/kaggle/input/cats-and-dogs-image-classification/train/cats/cat_104.jpg",
    target_size=(64, 64)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = best_model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

```

