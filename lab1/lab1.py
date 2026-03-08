import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#загрузка и просмотр данных
print("ЗАГРУЗКА И ПРОСМОТР ДАННЫХ")
df = pd.read_csv("titanic.csv")
print("Первые 5 строк:")
print(df.head())
print(f"\nРазмер: {df.shape[0]} строк, {df.shape[1]} колонок")

#количество пропусков
print("КОЛ-ВО ПРОПУСКОВ")
missing = df.isnull().sum()
print("Количество пропусков:")
print(missing)
print("\nПроцент пропусков:")
print((missing / len(df)) * 100)

print("ЗАПОЛНЕНИЕ ПРОПУСКОВ")
#медиана
if 'age' in df.columns:
    df['age'].fillna(df['age'].median(), inplace=True)
    print("ВОЗРАСТ ЗАПОЛНИЛИ МЕДИАНОЙ")

#мода
if 'embarked' in df.columns:
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    print("МЕСТО ПОСАДКИ ЗАПОЛНЕНО МОДОЙ")

#заполняем каюту
if 'cabin' in df.columns:
    df['cabin'].fillna('Unknown', inplace=True)
    print("Каюты обработаны")
print("\nПропуски после обработки:")
print(df.isnull().sum())

#нормализация
print("НОРМАЛИЗАЦИЯ ДАННЫХ")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(f"Нормализуем столбцы: {list(numeric_cols)}")

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("ДАННЫЕ НОРМАЛИЗИРОВАНЫ MinMaxScaler")
print(df[numeric_cols].head())

#преобразование
print("ПРЕОБРАЗОВАНИЕ")

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Категориальные столбцы: {list(categorical_cols)}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Размер после кодирования: {df_encoded.shape}")
print("Новые колонки:", list(df_encoded.columns)[:10], "...")

#сохранение
print("СОХРАНЕНИЕ")
df_encoded.to_csv("processed_titanic.csv", index=False)
print("Файл 'processed_titanic.csv' сохранен")
