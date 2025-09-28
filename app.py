import streamlit as st
import pandas as pd
import pickle

# =====================
# 1. CARGA DE OBJETOS ENTRENADOS
# =====================
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# NO necesitamos label_encoders, este es para la variable objetivo Patologia en entrenamiento
# pero no en inferencia

with open("tc.pkl", "rb") as f:
    train_columns = pickle.load(f)

with open("sc.pkl", "rb") as f:
    scaler = pickle.load(f)

# =====================
# 2. INTERFAZ GRÁFICA
# =====================
st.title("Predicción de Patología en Muestras")
st.write("Ingresa los datos de la muestra para predecir si está Enfermo o No Enfermo.")

sexo = st.selectbox("Sexo", ["M", "H"])
profundidad = st.number_input("Profundidad (mm)", min_value=0.0, step=0.1)
pesokg = st.number_input("Peso (kg)", min_value=50.0, step=0.1)
porcentajemagro = st.number_input("Porcentaje Magro (%)", min_value=0.0, max_value=100.0, step=0.1)

departamentos = [
    "Cundinamarca", "Meta", "Antioquia", "Boyaca", "Casanare",
    "Caldas", "Tolima", "Santander", "Valle del Cauca", "Guaviare",
    "Arauca", "Risaralda", "Caqueta", "Quindio", "Huila",
    "Cauca", "Cesar"
]
Departamento = st.selectbox("Departamento", departamentos)

Mes = st.selectbox("Mes", list(range(1, 13)))

# =====================
# 3. CREACIÓN DE DATAFRAME
# =====================
datos = [[sexo, profundidad, pesokg, porcentajemagro, Departamento, Mes]]
data_input = pd.DataFrame(datos, columns=['sexo','profundidad','pesokg',
                                         'porcentajemagro','Departamento','Mes'])

st.write("📋 **Datos capturados:**")
st.dataframe(data_input)

# =====================
# 4. PREPROCESAMIENTO Y PREDICCIÓN
# =====================
if st.button("Predecir"):
    try:
        # --- GENERAR DUMMIES Y REORDENAR COLUMNAS ---
        data_preparada = pd.get_dummies(data_input)
        data_preparada = data_preparada.reindex(columns=train_columns, fill_value=0)

        # --- ESCALAR DATOS ---
        data_preparada_scaled = scaler.transform(data_preparada)

        # --- PREDICCIÓN ---
        prediccion = modelo.predict(data_preparada_scaled)

        # Mostrar el resultado y el valor de la predicción
        valor_pred = prediccion[0]  # normalmente es 0 o 1
        resultado = "🟢 Sano" if valor_pred == 0 else "🔴 Enfermo"

        st.success(f"Resultado de la predicción: **{resultado}** (Valor predicho: {valor_pred})")

    except Exception as e:
        st.error(f"Ocurrió un error al predecir: {e}")

