#import sklearn
import nltk
import scipy as sp
from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


#CorpusDirTrain = r'E:\Datasets\16NepaliNews\16NepaliNews\Train'
#CorpusDirTest = r'E:\Datasets\16NepaliNews\16NepaliNews\Test'

CorpusDir=r'E:\Datasets\16NepaliNews\16NepaliNews'
#trainNews=load_files(CorpusDirTrain, shufffle=True)
#trainNews=load_files(CorpusDirTest, shufffle=True)

trainNews = load_mlcomp('16NepaliNews', 'train', mlcomp_root= CorpusDir)
testNews = load_mlcomp('16NepaliNews', 'test', mlcomp_root= CorpusDir)

''' Nepali Stop Words '''
# The stop words file is copied into the stopwords directory of nltk.data\corpora folder

stopWords = set(nltk.corpus.stopwords.words('nepali'))


''' Testing and Training Data '''
xTrain = trainNews.data
xTest = testNews.data
yTrain = trainNews.target
yTest = testNews.target

''' feature vector construction '''
'''tokenizing text with count'''

count_vect = CountVectorizer(tokenizer= lambda x: x.split(" "),
                                encoding='utf-8',
                                  decode_error='ignore', analyzer='word',
                                  max_df=0.5,
                                  min_df=10,
                                  stop_words=stopWords)
X_train_counts = count_vect.fit_transform(xTrain)
print(X_train_counts.shape)

'''TF-IDF calculation'''
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#multinomial naive Bayes
clf1 = Pipeline([
    ('vect', CountVectorizer),
    ('tfidf',TfidfTransformer)
    ('clf', MultinomialNB())
])

# Linear SVM
clf2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',TfidfTransformer())
    ('clf', SVC(kernel='linear'))
])

# RBF Kernel SVM
clf3 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',TfidfTransformer())
    ('clf', SVC(kernel='rbf'))
])

# MLP Neural Network
clf4 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',TfidfTransformer())
    ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5,
                          hidden_layer_sizes=(5, 2),
                          random_state=1))
])
#Radial Bias Neural Network
#need to write, I couldnt fint it on Sklearn

def trainAndEvaluate(clf, xTrain, xTest, yTrain, yTest):
    clf.fit(xTrain, yTrain)
    print("Accuracy on training Set : ")
    print(clf.score(xTrain, yTrain))
    print("Accuracy on Testing Set : ")
    print(clf.score(xTest, yTest))
    yPred=clf.predict(xTest)
    print("Classification Report : ")
    print(metrics.classification_report(yTest, yPred))
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(yTest, yPred))
    
print('Multinominal Naive Bayes \n')
trainAndEvaluate(clf1, xTrain, xTest, yTrain, yTest)
print('Linear Kernel SVC \n')
trainAndEvaluate(clf2, xTrain, xTest, yTrain, yTest)
print('RBF Kernel SVC \n')
trainAndEvaluate(clf3, xTrain, xTest, yTrain, yTest)
print("MLP Neuaral Network")
trainAndEvaluate(clf4, xTrain, xTest, yTrain, yTest)
