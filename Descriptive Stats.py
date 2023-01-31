import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

#Data Preprocessing

#df1 = pd.DataFrame()
#path_dir = 'C:/Users/darag/Desktop/Final Year Project/'

#all_files = os.listdir(path_dir)    
#xlsx_files = list(filter(lambda f: f.endswith('.xlsx'), all_files))

#summary_file = []
#for fil in xlsx_files:
#    print(path_dir + fil)
#    summary_file.append(path_dir + fil)

#for file in summary_file:
#    if os.path.exists(file): #if file exists:
#           df_summary = pd.read_excel(file, index_col = 0)
#           df1 = df1.append(df_summary, ignore_index=True, sort=False)
     
#file_name = df1.to_csv('All_Saves.csv', index = True)
df2 = pd.read_csv('C:/Users/darag/Desktop/Final Year Project/All_Saves.csv')
#Data Cleaning & Descriptive Statistics

#df2 = df1.dropna()
#print(df2.isnull())
y = df2['Classes']
#print(X.describe())
z = statistics.median(y)
#print("Median value :", z)
w = statistics.mode(y)
#print("Mode value :", w)

Y = df2['Classification']
X = df2.drop(['Unnamed: 0', 'Classes', 'Folder Name', 'Start Time', 'End Time', 'Classification'], axis=1)

df3 = df2.dropna()
df3 = df3[df3['Classes'] < 13]
Z = df3['Classes']
W = df3.drop(['Unnamed: 0', 'Classes', 'Folder Name', 'Start Time', 'End Time', 'Classification'], axis=1)

Saves = ['R-Low B', 'L-Low B', 'L-High B', 'High Catch F', 'R-High B', 'Block B', 'Centre B']
New_Colors = ['green','blue','purple','brown','teal', 'red', 'orange', 'yellow', 'pink']
Q = y.value_counts().plot(kind= 'bar', color=New_Colors)
#Q.set_xticklabels(Saves)
plt.ylabel('Frequency')
plt.xlabel('Type of save')
plt.title('Amount of each save made (Classified Data)')
plt.show()

#Cross Validation & Machine learning Algorithms

kf = KFold(n_splits=5)
#for train_index, test_index in kf.split(df2):
        #print("TRAIN:", train_index, "TEST:", test_index)
        
#KNN Classification Models
knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=16, p=2, weights='distance')
#knn = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=16, p=1, weights='distance')
y_pred1 = cross_val_predict(knn, X, Y, cv= kf)
print(classification_report(Y, y_pred1))

y_pred2 = cross_val_predict(knn, W, Z, cv= kf)
print(classification_report(Z, y_pred2))

#Decision Tree Models
dtree = DecisionTreeClassifier(class_weight=('balanced'))
y_pred3 = cross_val_predict(dtree, X, Y, cv= kf)
print(classification_report(Y, y_pred3))

y_pred4 = cross_val_predict(dtree, W, Z, cv= kf)
print(classification_report(Z, y_pred4))

#Random forest Models
rfc = RandomForestClassifier(n_estimators = 10000, bootstrap = True, max_depth = None, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 2, class_weight = 'balanced', random_state = 11)
y_pred5 = cross_val_predict(rfc, X, Y, cv= kf)
print(classification_report(Y, y_pred5))

y_pred6 = cross_val_predict(rfc, W, Z, cv= kf)
print(classification_report(Z, y_pred6))
