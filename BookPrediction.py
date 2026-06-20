import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("BookData.csv")
print(data)
print("TotalRows", len(data))
vectorizer = TfidfVectorizer(stop_words="english")
x = vectorizer.fit_transform(data['summary'])
y = data['genre']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("ModelAccuracy", accuracy)
with open('model01.pkl', "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Model and vectorizer saved successfully")
sample_text = ['A young hero discovers magical power and fights dark forces']
sample_vector = vectorizer.transform(sample_text)
prediction = model.predict(sample_vector)
print('SamplePrediction', prediction[0])