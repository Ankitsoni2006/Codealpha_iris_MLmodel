from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
iris = load_iris()
print(type(iris))
print(iris.feature_names)
print(iris.target_names)
print(iris.data[:5])
print(iris.target[:5])
print(iris.keys())
x = iris.data 
y = iris.target
x_train , x_test , y_train , y_test = train_test_split( x ,y, test_size = 0.2, random_state = 42)
print("Training features shape:", x_train.shape)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train , y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("\nModel Accuracy:" , accuracy)
print("\nEnter flower measurements to prredict its species:")
sepal_length = float(input("Sepal length (cm):"))
sepal_width = float(input("Sepal width (cm):"))
petal_length = float(input("Petal length (cm):"))
petal_width = float(input("Petal width (cm):"))
user_input = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1,-1)
prediction = knn.predict(user_input)
print("Prediction class:", prediction)
print("Predicted flower name :", iris.target_names[prediction[0]])
