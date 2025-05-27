import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma.extras import average
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from scipy import stats


# show the histogram and correlation heatmap for dataset.
def show_insepection(df):
    # Distribution of each datas
    df.hist(bins=50, figsize=(20, 15))
    plt.show()

    # corr_matrix = df.corr()
    # plt.figure(figsize=(14, 10))
    # sns.heatmap(corr_matrix, annot=False, cmap='Greens', fmt='.2f', annot_kws={'size': 10}, linewidths=0.5)
    # plt.show()

# classification object to predict wine quality with LinearRegression
# this contains model evaluation.
def classification(df):
    # to
    x = df.copy()
    y = x['quality']
    x = x.drop(['quality'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)
    lr = LogisticRegression(solver='lbfgs',multi_class='ovr', random_state=123)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    y_proba = lr.predict_proba(x_test)

    # create dataframe to see easier
    y_df = pd.DataFrame(y_test)
    y_df['predict'] = y_pred
    y_df['probability'] = y_proba[:, 1]

    # evaluation
    print("Accuracy : %.3f" % accuracy_score(y_test, y_pred))
    print("Precision : %.3f" % precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Recall : %.3f" % recall_score(y_test, y_pred, average='macro'))
    print("F1-score : %.3f" % f1_score(y_test, y_pred, average='macro'))
    print("AUC : %.3f" % roc_auc_score(y_test, y_proba, multi_class='ovr'))



if __name__ == '__main__':
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', None)

    data = pd.read_csv('dataset/winequalityN.csv')
    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    # drop type. Or, may encode into integers
    # If set the type to target label, object of classification is judge what type is.
    # wine_type = df['type']
    # df = df.drop(['type'], axis=1)
    wine_type = [0 if i == 'red' else 1 for i in df['type']]
    df['type'] = wine_type

    scale_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                     'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    minmaxscaler = MinMaxScaler()
    df[scale_columns] = minmaxscaler.fit_transform(df[scale_columns])
    # show_insepection(df)

    classification(df)
