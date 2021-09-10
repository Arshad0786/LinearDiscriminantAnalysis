import matplotlib.pyplot as plt
import pandas
import numpy as np

iris = pandas.read_csv("C:\\Users\\js071\\Downloads\\iris.data.csv", names=[
                       'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])


# create color dictionary
colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'g'}
# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point

for i in range(150):
    ax.scatter(iris['sepal_length'][i], iris['sepal_width']
               [i], color=colors[iris['class'][i]])


Iris_setosa_list = list()
Iris_versicolor_list = list()

for i in range(150):  # transform attributes from pandas dataframe to list
    if (i < 50):
        Iris_setosa_list.append(
            (iris['sepal_length'][i], iris['sepal_width'][i]))
    else:
        Iris_versicolor_list.append(
            (iris['sepal_length'][i], iris['sepal_width'][i]))


def isListDataCorrect():  # check if data is align with original data
    for i in range(50):
        if(Iris_setosa_list[i][0] != iris['sepal_length'][i]):
            return False
        if(Iris_setosa_list[i][1] != iris['sepal_width'][i]):
            return False
    for i in range(50, 150):
        if(Iris_versicolor_list[i-50][0] != iris['sepal_length'][i]):
            return False
        if(Iris_versicolor_list[i-50][1] != iris['sepal_width'][i]):
            return False
    return True


def mean(datalist):
    sum1 = 0
    sum2 = 0
    for i in range(len(datalist)):
        sum1 = sum1 + datalist[i][0]
        sum2 = sum2 + datalist[i][1]
    sum1 = sum1 / len(datalist)
    sum2 = sum2 / len(datalist)
    return (sum1, sum2)


Iris_setosa_mean = mean(Iris_setosa_list)
Iris_versicolor_mean = mean(Iris_versicolor_list)

# find B
u1 = np.array([Iris_setosa_mean])
u2 = np.array([Iris_versicolor_mean])
B = np.multiply(np.transpose(u1-u2), u1-u2)
# calculate scatter
S1 = np.array([[0, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 0]])

for i in range(len(Iris_setosa_list)):
    S1 = S1 + (np.array(Iris_setosa_list[i]) - u1) * \
        np.transpose((np.array(Iris_setosa_list[i]) - u2))

for i in range(len(Iris_versicolor_list)):
    S2 = S2 + (np.array(Iris_versicolor_list[i]) - u2) * \
        np.transpose((np.array(Iris_versicolor_list[i]) - u1))

S = S1 + S2

objecitveMatrix = np.matmul((np.linalg.inv(S)), B)  # find (S^-1)*B
eigval, eigvec = np.linalg.eig(objecitveMatrix)  # then we find it's eigenvalue
# find eigenvector correspond to the maximal eigenvalue
MaxEigvec = eigvec[:,np.argmax(eigval)]
# make it unit vector so it's easier to find projection matrix
MaxEigvec = MaxEigvec / np.linalg.norm(MaxEigvec)
# make it 1x2 cause so numpy can transpose it
MaxEigvec = np.array([MaxEigvec])
# find projection matrix of that eigen vector
Projection_matrix = np.transpose(MaxEigvec)*MaxEigvec


projected_Iris_setosa = list()
projected_Iris_versicolor = list()

for i in range(len(Iris_setosa_list)):  # find all points
    projected_Iris_setosa.append(
        np.matmul(Projection_matrix, np.transpose(np.array([Iris_setosa_list[i]]))))
for i in range(len(Iris_versicolor_list)):
    projected_Iris_versicolor.append(
        np.matmul(Projection_matrix, np.transpose(np.array([Iris_versicolor_list[i]]))))

JoinedList = projected_Iris_setosa + projected_Iris_versicolor
result = np.matmul(Projection_matrix, np.transpose(
    np.array([Iris_setosa_list[0]])))

Pmean1 = mean(projected_Iris_setosa)
Pmean2 = mean(projected_Iris_versicolor)

for i in range(150):
    #print(i, " : ", JoinedList[i])
    if (i < 50):
        ax.scatter(JoinedList[i][0], JoinedList[i][1], color='r')
    else:
        ax.scatter(JoinedList[i][0], JoinedList[i][1], color='g')
ax.scatter(Pmean1[0], Pmean1[1], color='b')
ax.scatter(Pmean2[0], Pmean2[1], color='k')


# set a title and labels
ax.set_title('Iris Dataset')

plt.show()
