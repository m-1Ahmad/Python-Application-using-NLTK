import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


data = pd.read_csv('data.csv')  
data['Review Text'] = data['Review Text'].str.lower().replace('[^\w\s]', '').replace('\d+', '').strip()

tfidf = TfidfVectorizer() 
tfidf_matrix = tfidf.fit_transform(data['Review Text'])
tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())

data['Label'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0 if x == 3 else -1)

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, data['Label'], test_size=0.1, random_state=42)

naive_bayes = MultinomialNB() 
svm = SVC() 
decision_tree = DecisionTreeClassifier()  

naive_bayes.fit(X_train, y_train)  
svm.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

nb_predictions = naive_bayes.predict(X_test)
svm_predictions = svm.predict(X_test)
dt_predictions = decision_tree.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_f1_score = f1_score(y_test, nb_predictions, average='macro')
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_f1_score = f1_score(y_test, svm_predictions, average='macro')
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_f1_score = f1_score(y_test, dt_predictions, average='macro')

print("Naïve Bayes Accuracy:", nb_accuracy)
print("Naïve Bayes F1 Score:", nb_f1_score)
print("Support Vector Machine Accuracy:", svm_accuracy)
print("Support Vector Machine F1 Score:", svm_f1_score)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree F1 Score:", dt_f1_score)

if nb_accuracy > svm_accuracy and nb_accuracy > dt_accuracy:
    print("Naïve Bayes performs better.")
elif svm_accuracy > nb_accuracy and svm_accuracy > dt_accuracy:
    print("Support Vector Machine performs better.")
else:
    print("Decision Tree performs better.")
