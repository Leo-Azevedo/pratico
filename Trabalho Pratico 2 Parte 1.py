import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm, neighbors, ensemble

digits = load_digits()
expected = digits.target[1::2]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[::2], digits.target[::2])

results1 = clf.predict(digits.data[1::2])
score3 = clf.score(digits.data[1::2], digits.target[1::2])


clf2 = neighbors.KNeighborsClassifier()
clf2.fit(digits.data[::2], digits.target[::2])

results2 = clf2.predict(digits.data[1::2])
score2 = clf2.score(digits.data[1::2], digits.target[1::2])


clf3 = ensemble.RandomForestClassifier()
clf3.fit(digits.data[::2], digits.target[::2])

results3 = clf3.predict(digits.data[1::2])
score1 = clf3.score(digits.data[1::2], digits.target[1::2])


names = ["Ensemble", "KNeighbors", "SVC"]
values = [score1 * 100 , score2 * 100 , score3 * 100]

print(values)

plt.bar(names, values, color = (0.88, 0.88, 0.88))

def addLabelOnTop(barras):
    k = -1.25
    for barra in barras:
        k = k + 1
        plt.annotate("{0:.2f}".format(barra) + "%",
            xy=(0.1+k, 100), xycoords='data', textcoords='data')
        

plt.title("Comparação")

addLabelOnTop(values)

plt.show()
