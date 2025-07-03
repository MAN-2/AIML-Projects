import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

df=pd.read_csv('dataset.csv') #Loading the dataset
print("Dataset Sample:\n",df.head())   #Displaying rows  using df.head

# missing values handling
df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

if 'customerID' in df.columns:
    df=df.drop('customerID',axis=1)

#one hot encoding
#df_enc= pd.get_dummies(df,columns=['gender','Contract','PaymentMethod'])
df_enc = pd.get_dummies(df, drop_first=True)


# features=X , target= y
X=df_enc.drop('Churn', axis=1) 
Y=df_enc['Churn']

# Splitting of data into training and testing set
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.3 , random_state=42 , stratify=Y)

# Scaling features
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

model = LinearRegression() # we are using Linear Regression
model.fit(X_train , Y_train) #Model Training

y_pred =model.predict(X_test)

# Evaluation
accuracy=accuracy_score(Y_test,y_pred)
report = classification_report(Y_test , y_pred , zero_division=1)
conf_mat= confusion_matrix(Y_test,y_pred)

print(f"\nAccuracy:{accuracy}")
print("\nClassification Report:\n",report)
print("\nConfusion Matrix:\n", conf_mat)