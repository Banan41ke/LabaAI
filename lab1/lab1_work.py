import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("symbols_valid_meta.csv")

#print(df.head(10))
a = df.isnull().sum()
print(a)

print("Заполнение пропусков")

df['Financial Status'] = df['Financial Status'].fillna(df['Financial Status'].mode()[0])
df['CQS Symbol'] = df['CQS Symbol'].fillna(df['CQS Symbol'].mode()[0])

print("Пропуски после заполнения")
print(df[['Financial Status', 'CQS Symbol']].isnull().sum())

numeric_cols = df.select_dtypes(include=['number']).columns
print(f"Нормализуемые столбцы {list(numeric_cols)}")

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

categorical_cols = ['Listing Exchange', 'Market Category']
print(f"Размер до кодирования: {df.shape}")
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(f"Размер до кодирования: {df_encoded.shape}")

print("Итоговые данные")
print(df_encoded.head(25))