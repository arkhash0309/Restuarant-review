# importing the necessary libraries
import numpy as np
import pandas as pd

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

# import the libraries for NLP
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# lets us create an empty array to append the cleaned up text
data = []

# let us run a for loop for the 1000 reviews
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower() # converting all into lower case to keep on one standard
    review = review.split() # split the words using a space

    ps = PorterStemmer() # creating an object of the PorterStemmer class to take the stem of each word

    # let us remove the stopwords and take the stem of the word
    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))] 
    
    # now all the words could be rejoined
    review = ''.join(review)
    
    # let us create the array of the cleaned up text
    data.append(review)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(data).toarray()  # input features
y = df.iloc[:, 1].values  # output features

# now let us split into training and testing split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# lets us try with a Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred 

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)