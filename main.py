from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import classification_report
import numpy as np

def read_data(ngram_range):
    fo_neg = open('neg.tok', 'r')
    line_neg = fo_neg.readlines()
    fo_neg.close()
    fo_pos = open('pos.tok', 'r')
    line_pos = fo_pos.readlines()
    fo_pos.close()
    corpus = line_neg + line_pos
    vectorizer = CountVectorizer(binary=True, ngram_range = ngram_range )
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    #print(len(line_neg))  # 1368
    return X

def test_clf(clf,x,y):
   cv_results = cross_validate(clf, x, y,cv=10,return_estimator=False)
   print('The average accuracy is : '+ str(np.mean(cv_results['test_score'])))
   y_pred = cross_val_predict(clf, x, y, cv=10)
   print(classification_report(y,y_pred,target_names = ['neg','pos']))



print(read_data((1,1)))

if __name__ == '__main__':
    y_neg = np.zeros([1368,1])
    y_pos = np.ones([2407,1])
    y = np.vstack((y_neg,y_pos))
    y = np.ravel(y)
    x_word = read_data((1, 1))
    x_ngram = read_data((1, 2))

    bayes_clf = MultinomialNB(alpha=1.0)  # add one smooth
    print('A Naïve Bayes classifier with bag-of-words features')
    test_clf(bayes_clf, x_word, y)
    print('A Naïve Bayes classifier with bag-of-ngram features')
    test_clf(bayes_clf, x_ngram, y)

    lr_clf = LogisticRegression(solver='lbfgs')
    print('A logical regression classifier with bag-of-words features')
    test_clf(lr_clf, x_word, y)
    print('A logical regression classifier with bag-of-ngram features')
    test_clf(lr_clf, x_ngram, y)


    p = x_word[1368:].sum(axis = 0) + 1
    q = x_word[:1368].sum(axis = 0) + 1
    r = np.log((p/p.sum())/(q/q.sum()))
    x_word_bayes_lr = np.multiply(r,x_word)
    print('Bayes and logical regression classifier with bag-of-words features')
    test_clf(lr_clf, x_word_bayes_lr, y)

    p = x_ngram[1368:].sum(axis = 0) + 1
    q = x_ngram[:1368].sum(axis = 0) + 1
    r = np.log((p / p.sum()) / (q / q.sum()))
    x_ngram_bayes_lr = np.multiply(r, x_ngram)
    print('Bayes and logical regression classifier with bag-of-ngram features')
    test_clf(lr_clf, x_ngram_bayes_lr, y)


