Bank Note Authentication
Overview
This project involves authenticating banknotes using machine learning techniques. The dataset consists of images that were captured from genuine and forged banknote-like specimens. An industrial camera, typically used for print inspection, was employed to capture the images. The images were digitized with a resolution of about 660 dpi and are of size 400x400 pixels. Wavelet Transform tools were used to extract features from the images.

Dataset Link
Bank Note Authentication Dataset

Features
The dataset contains the following features:

variance: Variance of Wavelet Transformed image.
skewness: Skewness of Wavelet Transformed image.
curtosis: Curtosis of Wavelet Transformed image.
entropy: Entropy of image.
class: 0 for genuine and 1 for forged.
Data Import and Exploration
The dataset is imported using pandas and displayed for initial exploration.

python
Copy
Edit
import pandas as pd
import numpy as np

df = pd.read_csv('BankNote_Authentication.csv')
df
Sample of the dataset:

variance	skewness	curtosis	entropy	class
3.62160	8.66610	-2.8073	-0.44699	0
4.54590	8.16740	-2.4586	-1.46210	0
3.86600	-2.63830	1.9242	0.10645	0
3.45660	9.52280	-4.0112	-3.59440	0
...	...	...	...	...
Feature Selection
Independent Features (X): All columns except for the last column (class).
Dependent Feature (y): The class column representing the banknote authenticity.
python
Copy
Edit
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
Train-Test Split
The dataset is split into training and testing sets using an 70-30 split.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
Model Implementation
A Random Forest Classifier is used for classification.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
Prediction
Once the model is trained, predictions are made on the test set:

python
Copy
Edit
y_pred = classifier.predict(X_test)
Model Accuracy
The accuracy of the model is evaluated using the accuracy_score from sklearn.metrics.

python
Copy
Edit
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)
print(score)  # Output: 0.9903 (99.03%)
Model Serialization
The trained model is serialized and saved into a .pkl file using pickle.

python
Copy
Edit
import pickle

pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
Example Prediction
To use the saved model to make a prediction on new data:

python
Copy
Edit
import numpy as np
classifier.predict([[2, 3, 4, 1]])  # Example input
Output:

python
Copy
Edit
array([0], dtype=int64)  # Predicted class: 0 (genuine)
Requirements
Python 3.x
pandas
numpy
scikit-learn
pickle
Conclusion
This project demonstrates the use of machine learning for banknote authentication. The Random Forest Classifier achieved a high accuracy of 99.03%, indicating that the model can effectively differentiate between genuine and forged banknotes based on extracted image features.
