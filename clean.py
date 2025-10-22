import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"C:\Users\LENOVO\Desktop\elevate labs\titanic-Dataset.csv")
print("Data loaded!\n")


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])



df = pd.get_dummies(df, drop_first=True)

df[['Age', 'Fare']] = StandardScaler().fit_transform(df[['Age', 'Fare']])



Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[~((df['Fare'] < Q1 - 1.5*IQR) | (df['Fare'] > Q3 + 1.5*IQR))]


print("Cleaned data preview:\n")
print(df.head())
print("\nShape of cleaned dataset:", df.shape)

df.to_csv(r"C:\Users\LENOVO\Desktop\elevate labs\titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
