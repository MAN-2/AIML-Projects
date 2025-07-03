import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

df=pd.read_csv('creditcard.csv') #Loading the dataset
print("Dataset Sample:\n",df.head())   #Displaying rows  using df.head

# features=X , target= y
X=df.drop('Class', axis=1) 
Y=df['Class']

# Splitting of data into training and testing set
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2 , random_state=42 , stratify=Y)

# Scaling features
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100 , random_state=42) # we are using Random Forest

model.fit(X_train , Y_train) #Model Training

y_pred =model.predict(X_test)

# Evaluation
accuracy=accuracy_score(Y_test,y_pred)
report = classification_report(Y_test , y_pred)
conf_mat= confusion_matrix(Y_test,y_pred)

print(f"\nAccuracy:{accuracy}")
print("\nClassification Report:\n",report)
print("\nConfusion Matrix:\n", conf_mat)