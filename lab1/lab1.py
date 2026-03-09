import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
print("ЗАГРУЗКА И ПРОСМОТР ДАННЫХ")
df = pd.read_csv("titanic.csv")
print("Первые 5 строк:")
print(df.head())
print(f"\nРазмер: {df.shape[0]} строк, {df.shape[1]} колонок")

# КОЛ-ВО ПРОПУСКОВ
print("\nКОЛ-ВО ПРОПУСКОВ")
missing = df.isnull().sum()
print("Количество пропусков:")
print(missing)
print("\nПроцент пропусков:")
print((missing / len(df)) * 100)

# ЗАПОЛНЕНИЕ ПРОПУСКОВ
print("\nЗАПОЛНЕНИЕ ПРОПУСКОВ")

if 'age' in df.columns:
    df['age'].fillna(df['age'].median(), inplace=True)
    print("ВОЗРАСТ ЗАПОЛНИЛИ МЕДИАНОЙ")

if 'fare' in df.columns:
    df['fare'].fillna(df['fare'].median(), inplace=True)
    print("СТОИМОСТЬ БИЛЕТА ЗАПОЛНИЛИ МЕДИАНОЙ")

if 'embarked' in df.columns:
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    print("МЕСТО ПОСАДКИ ЗАПОЛНЕНО МОДОЙ")

if 'cabin' in df.columns:
    df['cabin'].fillna('Unknown', inplace=True)
    print("Каюты обработаны")

print("\nПропуски после обработки:")
print(df.isnull().sum())

# УДАЛЕНИЕ НЕНУЖНЫХ КОЛОНОК
print("\nУДАЛЕНИЕ НЕНУЖНЫХ КОЛОНОК")
cols_to_drop = ['name', 'ticket', 'boat', 'body', 'home.dest']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
        print(f"Удалена колонка: {col}")

print(f"Размер после удаления: {df.shape[0]} строк, {df.shape[1]} колонок")

# НОРМАЛИЗАЦИЯ ДАННЫХ
print("\nНОРМАЛИЗАЦИЯ ДАННЫХ")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(f"Нормализуем столбцы: {list(numeric_cols)}")

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("ДАННЫЕ НОРМАЛИЗИРОВАНЫ MinMaxScaler")
print(df[numeric_cols].head())

# ПРЕОБРАЗОВАНИЕ
print("\nПРЕОБРАЗОВАНИЕ")

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Категориальные столбцы: {list(categorical_cols)}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Размер после кодирования: {df_encoded.shape}")
print("Новые колонки:", list(df_encoded.columns)[:10], "...")

# РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
print("\nРАЗБИЕНИЕ НА TRAIN/TEST")

train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {train_df.shape}")
print(f"Тестовая выборка: {test_df.shape}")

# СОХРАНЕНИЕ
print("\nСОХРАНЕНИЕ")

df_encoded.to_csv("processed_titanic.csv", index=False)
print("Файл 'processed_titanic.csv' сохранен")

train_df.to_csv("train_titanic.csv", index=False)
print("Файл 'train_titanic.csv' сохранен")

test_df.to_csv("test_titanic.csv", index=False)
print("Файл 'test_titanic.csv' сохранен")