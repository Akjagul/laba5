import re
import string
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#загрузка датасета
dataset = pd.read_csv('train.csv')
#удаление значений Nan
dataset.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
#выделение целевого столбца
train_labels = dataset['label']

#очистка данных
def remove_un(data):
    data = data.lower()
    data = re.sub('\[.*?\]','',data)
    data = re.sub('\\W',' ',data)
    data = re.sub('https?://\S+|www.\S+','',data)
    data = re.sub('<.*?>+','',data)
    data = re.sub('[%s]'%re.escape(string.punctuation),'',data)
    data = re.sub('\n','',data)
    data = re.sub('\w*\d\w','',data)
    return data

dataset['text']= dataset['text'].apply(remove_un)

#разделение дататсета на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], train_labels, test_size=0.1, random_state=0)
#векторизация данных
tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

#обучение
#pac = PassiveAggressiveClassifier(max_iter = 50)
#pac.fit(tfidf_train, y_train)
#y_pred = pac.predict(tfidf_test)
#score = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {round(score * 100, 2)}%')

#lg = LogisticRegression()
#lg.fit(tfidf_train, y_train)
#y_pred = lg.predict(tfidf_test)
#score = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {round(score * 100, 2)}%')

nbc = MultinomialNB()
nbc.fit(tfidf_train, y_train)
y_pred = nbc.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')


plt.subplot(1, 2, 1)
plt.hist(y_test)
plt.subplot(1, 2, 2)
plt.hist(y_pred)
plt.show()
