import kfold_template
import pandas
from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values

r2_scores = kfold_template.run_kfold(data, target, 4, linear_model.LinearRegression())
print(r2_scores)

# r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, linear_model.LogisticRegression(), 1, 1)

# print(r2_scores)

# print(accuracy_scores)

# for confusion_matrix in confusion_matrices:
# 	print(confusion_matrix)




