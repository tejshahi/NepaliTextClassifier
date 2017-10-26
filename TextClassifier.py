#import sklearn
import nltk
import scipy as sp


from sklearn.datasets import load_mlcomp
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
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
X_train_counts = count_vect.fit_transform(trainNews.data)
print(X_train_counts.shape)

'''TF-IDF calculation'''
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#multinomial naive Bayes
clf1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',TfidfTransformer())
    ('clf', MultinomialNB(alpha=0.01, fit_prior=True))
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

# SVC Poly Kernel
clf6 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='poly'))
])

