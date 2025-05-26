import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from scipy import stats


def show_insepection(df):
    # Distribution of each datas
    df.hist(bins=50, figsize=(20, 15))
    plt.show()

    corr_matrix = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='Greens', fmt='.2f', annot_kws={'size': 10}, linewidths=0.5)
    plt.show()


def classification(df):
    print(df.head())


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    data = pd.read_csv('dataset/winequalityN.csv')
    df = pd.DataFrame(data)

    # drop type. Or, may encode into integers
    # If set the type to target label, object of classification is judge what type is.
    # wine_type = df['type']
    # df = df.drop(['type'], axis=1)

    # show_insepection(df)
