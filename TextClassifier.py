#import sklearn
import nltk
import scipy as sp
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#CorpusDirTrain = r'E:\Datasets\16NepaliNews\16NepaliNews\Train'
#CorpusDirTest = r'E:\Datasets\16NepaliNews\16NepaliNews\Test'

CorpusDir = r'E:\Datasets\16NepaliNews\16NepaliNews\16719\raw'
raw=load_files(CorpusDir, description=None, 
               load_content=True,
               encoding='utf-8',
               decode_error='ignore')

''' Nepali Stop Words '''
# The stop words file is copied into the stopwords directory of nltk.data\corpora folder

stopWords = set(nltk.corpus.stopwords.words('nepali')) 


''' Testing and Training Data '''
xTrain, xTest, yTrain, yTest=train_test_split(raw.data,
                                               raw.target,
                                               test_size=0.1,
                                               random_state=42)
''' feature vector construction '''
''' Vectorizer '''

tfidfVectorizer = TfidfVectorizer(tokenizer= lambda x: x.split(" "),
                                  sublinear_tf=True, encoding='utf-8',
                                  decode_error='ignore',
                                  max_df=0.5,
                                  min_df=10,
                                  stop_words=stopWords)

vectorised = tfidfVectorizer.fit_transform(xTrain)
print('No of Samples , No. of Features ', vectorised.shape)
''' Classifier '''

# Multinomial Naive Bayes
clf1 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', MultinomialNB(alpha=0.01, fit_prior=True))
])

# SVM Linear Kernel
clf2 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='linear'))
])
# SVM RBF Kernel
clf3 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='rbf'))
])

#MLP Neural Network
clf4=Pipeline([
        ('vect', tfidfVectorizer),
         ('clf', MLPClassifier())
])

#RBF Neural Network
#MLP Neural Network
clf4=Pipeline([
        ('vect', tfidfVectorizer),
         ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1))
])


def trainAndEvaluate(clf, xTrain, xTest, yTrain, yTest):
    clf.fit(xTrain, yTrain)
    print("Accuracy on training Set : ")
    print(clf.score(xTrain, yTrain))
    print("Accuracy on Testing Set : ")
    print(clf.score(xTest, yTest))
    yPred = clf.predict(xTest)
    
    print("Classification Report : ")
    print(metrics.classification_report(yTest, yPred))
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(yTest, yPred))


print('Multinominal Naive Bayes \n')
trainAndEvaluate(clf1, xTrain, xTest, yTrain, yTest)
print('Linear SVM \n')
trainAndEvaluate(clf2, xTrain, xTest, yTrain, yTest)
print('RBF SVM \n')
trainAndEvaluate(clf3, xTrain, xTest, yTrain, yTest)
print('MLP Neural Network n')
trainAndEvaluate(clf4, xTrain, xTest, yTrain, yTest)