from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd

class mlwork:
    def resulter(self, info):
        dataframe = pd.read_csv("C:\\Users\\adhee\\Documents\\Adheesh\\Coding\\Hackathon\\yeehaw\\daty.csv")
        x = dataframe['Inputtext']
        y = dataframe['Label']

        cov = CountVectorizer()
        featureset = cov.fit_transform(x)

        mymodel = svm.SVC()
        mymodel.fit(featureset, y)

        info = [info]
        transformation = cov.transform(info)
        result = mymodel.predict(transformation)
        result = result[0]
        print( result )
    
mm = mlwork
mm.resulter(mm, input())
