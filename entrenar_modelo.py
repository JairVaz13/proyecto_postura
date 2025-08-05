import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Cargar datos
try:
    df = pd.read_csv('dataset_posturas.csv')
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset_posturas.csv'")
    print("Ejecuta primero 'procesar_imagenes.py' para generar el dataset")
    exit()

# Verificar datos
if len(df) < 50:
    print(f"Advertencia: Dataset muy pequeño ({len(df)} muestras). Se recomiendan al menos 50 por clase.")

# Codificar clases
encoder = LabelEncoder()
df['clase_num'] = encoder.fit_transform(df['clase'])

# Dividir datos
X = df.drop(['clase', 'clase_num'], axis=1)
y = df['clase_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("\nEntrenando modelo...")
modelo = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Guardar modelo
joblib.dump(modelo, 'modelo_posturas.pkl')
joblib.dump(encoder, 'encoder.pkl')
print("\nModelo guardado como 'modelo_posturas.pkl'")