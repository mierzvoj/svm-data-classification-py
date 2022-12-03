import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr = LinearRegression ()
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


#Importing the dataset
df = pd.read_csv('winequality-red.csv')
#checking the unique label values
# print(df.quality.unique())
# This is the example of the wine quality and water potability classification labeling them good or bad.
# Wines with quality 7 or higher are classified as good and wines with lower than 7 are classified as bad.

# Converting the labels into only two variables
df.loc[df.quality >= 7, 'quality'] = 1
# print(df.quality.unique())
df.loc[df.quality > 1, 'quality'] = 0
df.shape

# #describing the data visually
# sns.pairplot (df)
# plt.savefig('pairplot.png')
# plt.show()

# Imbalance is observed in many features
# Data is not normally distributed possible skewness*

# *Skewness is a quantifiable measure of how distorted a data sample is from the normal distribution

# splitting features & labels
x = df.drop('quality', axis=1)
y = df.quality

#Using Quantile transformer for skewness removal

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer()
np_array = qt.fit_transform(x) #this will result in numpy array
np_array

#converting array into dataframe
xt = pd.DataFrame(np_array, columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'])


#train test split
for i in range (0,10):
    x_train,x_test,y_train,y_test = train_test_split (xt, y, test_size = 0.2, random_state = i)
    lr.fit (x_train, y_train)
    pred_train = lr.predict (x_train)
    pred_test = lr.predict (x_test)
    print (f"At random state {i}, the training accuracy is: {r2_score (y_train, pred_train)}")
    print (f"At random state {i}, the testing accuracy is: {r2_score (y_test, pred_test)}")
    print ('\n')

x_train, x_test, y_train, y_test = train_test_split (xt, y, test_size = 0.2, random_state = 2)

ovr_spl = SMOTE(0.75)

# SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the imbalance problem.
# It aims to balance class distribution by randomly increasing minority class examples by replicating them. SMOTE synthesises new minority instances between existing minority instances.

x_train_ns, y_train_ns = ovr_spl.fit_resample(x_train, y_train)

#Using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train_ns, y_train_ns)
y_pred = dtc.predict(x_test)


accuracy1 = accuracy_score(y_test, y_pred)
print("Decision Tree accuracy = ", accuracy1)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_ns, y_train_ns)
y_pred = svc.predict (x_test)

accuracy2 = accuracy_score(y_test, y_pred)
print("SVM = ", accuracy2)