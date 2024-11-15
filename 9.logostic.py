from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

classifier = LogisticRegression(max_iter=200)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)
mean_accuracy = cv_scores.mean()

print("Mean Accuracy (Cross-validation):", mean_accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
