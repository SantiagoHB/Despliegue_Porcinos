import streamlit as st
import pandas as pd
import pickle

# =====================
# 1. CARGA DE OBJETOS ENTRENADOS
# =====================
try:
    with open("modelo.pkl", "rb") as f:
        modelo = pickle.load(f)

    with open("tc.pkl", "rb") as f:
        train_columns = pickle.load(f)

    with open("sc.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.success("✅ Modelo, columnas y scaler cargados correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar los archivos de modelo: {e}")
    st.stop()

# =====================
# 2. INTERFAZ GRÁFICA
# =====================
st.title("🔎 Predicción de Patología en Muestras")
st.write("Ingresa los datos de la muestra para predecir si está **Sano (0)** o **Enfermo (1)**.")

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
data_input = pd.DataFrame(
    datos, 
    columns=['sexo', 'profundidad', 'pesokg', 'porcentajemagro', 'Departamento', 'Mes']
)

st.subheader("📋 Datos capturados")
st.dataframe(data_input)

# =====================
# 4. PREPROCESAMIENTO Y PREDICCIÓN
# =====================
if st.button("🔮 Predecir"):
    try:
        # --- One-hot encoding ---
        data_preparada = pd.get_dummies(data_input)

        # --- Reindexar columnas para coincidir con entrenamiento ---
        data_preparada = data_preparada.reindex(columns=train_columns, fill_value=0)

        # --- Escalar datos ---
        data_preparada_scaled = scaler.transform(data_preparada)

        # --- Predicción ---
        prediccion = modelo.predict(data_preparada_scaled)
        valor_pred = int(prediccion[0])  # convertir a int para mostrar limpio

        resultado = "🟢 Sano" if valor_pred == 0 else "🔴 Enfermo"
        st.success(f"Resultado: **{resultado}** (Valor: {valor_pred})")

    except Exception as e:
        st.error(f"❌ Ocurrió un error durante la predicción: {e}")
