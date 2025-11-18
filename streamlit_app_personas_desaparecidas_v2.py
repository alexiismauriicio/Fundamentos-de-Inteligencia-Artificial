import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Predicción de Personas Desaparecidas', layout='wide')
st.title('🔍 Predicción de situación de personas desaparecidas')

st.markdown("""
Esta sección estima, a partir de características demográficas, las probabilidades de que:
- el caso se **resuelva** (la persona sea localizada),
- la persona **siga desaparecida**,
- y en caso de ser localizada, sea **encontrada con vida**.
""")

# ============ Cargar modelos y transformadores ============

# Modelo 1: caso resuelto vs desaparecido
model_res = joblib.load('best_model.pkl')            # o model_resolucion.pkl
label_encoders_res = joblib.load('label_encoders.pkl')
scaler_res = joblib.load('scaler.pkl')

# Modelo 2: vivo vs fallecido (entre resueltos)
model_vivo = joblib.load('model_vivo.pkl')
label_encoders_v = joblib.load('label_encoders_v.pkl')
scaler_v = joblib.load('scaler_v.pkl')

# ============ Entradas del usuario ============

sexo = st.selectbox('Sexo', ['MUJER', 'HOMBRE'])
provincia = st.selectbox('Provincia', [
    'AZUAY', 'BOLIVAR', 'CAÑAR', 'CARCHI', 'COTOPAXI', 'CHIMBORAZO',
    'EL ORO', 'ESMERALDAS', 'GALAPAGOS', 'GUAYAS', 'IMBABURA', 'LOJA',
    'LOS RIOS', 'MANABI', 'MORONA SANTIAGO', 'NAPO', 'ORELLANA',
    'PASTAZA', 'PICHINCHA', 'SANTA ELENA', 'SANTO DOMINGO DE LOS TSACHILAS',
    'SUCUMBIOS', 'TUNGURAHUA', 'ZAMORA CHINCHIPE'
])
nacionalidad = st.selectbox('Nacionalidad', [
    'ECUADOR', 'VENEZUELA', 'COLOMBIA', 'PERU', 'DESCONOCIDO', 'OTRA'
])
etnia = st.selectbox('Etnia', [
    'MESTIZO/A', 'INDIGENA', 'AFRO', 'BLANCO/A', 'MONTUBIO/A',
    'MULATO/A', 'OTROS', 'DESCONOCIDO'
])
edad = st.number_input('Edad aproximada', min_value=0, max_value=100, value=20, step=1)

if st.button("Calcular probabilidades"):
    # Crear DataFrame de una fila con las mismas columnas que en el entrenamiento
    cols = ['sexo', 'provincia', 'nacionalidad', 'edad_aproximada', 'etnia']
    X_input = pd.DataFrame([{
        'sexo': sexo,
        'provincia': provincia,
        'nacionalidad': nacionalidad,
        'edad_aproximada': edad,
        'etnia': etnia
    }], columns=cols)

    # ========= 1) Transformaciones para modelo de resolución =========
    X_res = X_input.copy()
    for col in ['sexo', 'provincia', 'nacionalidad', 'etnia']:
        le = label_encoders_res[col]
        X_res[col] = le.transform(X_res[col])

    X_res['edad_aproximada'] = scaler_res.transform(X_res[['edad_aproximada']])

    # Probabilidad de caso resuelto
    p_resuelto = model_res.predict_proba(X_res)[:, 1][0]
    p_desaparecido = 1 - p_resuelto

    # ========= 2) Transformaciones para modelo vivo vs fallecido =========
    X_v = X_input.copy()
    for col in ['sexo', 'provincia', 'nacionalidad', 'etnia']:
        le_v = label_encoders_v[col]
        X_v[col] = le_v.transform(X_v[col])

    X_v['edad_aproximada'] = scaler_v.transform(X_v[['edad_aproximada']])

    p_vivo_cond = model_vivo.predict_proba(X_v)[:, 1][0]  # P(vivo | resuelto)
    p_fallecido_cond = 1 - p_vivo_cond

    # ========= 3) Probabilidades combinadas =========
    p_encontrado_vivo = p_resuelto * p_vivo_cond
    p_encontrado_fallecido = p_resuelto * p_fallecido_cond

    # (Opcional) Normalizar por si se va de 1 por temas numéricos
    total = p_encontrado_vivo + p_encontrado_fallecido + p_desaparecido
    p_encontrado_vivo /= total
    p_encontrado_fallecido /= total
    p_desaparecido /= total

    st.subheader("Resultados")
    st.write(f"🔵 Probabilidad de que el caso se resuelva: **{p_resuelto:.2%}**")
    st.write(f"🟢 Probabilidad de ser encontrado **vivo**: **{p_encontrado_vivo:.2%}**")
    
    st.caption("Estas probabilidades son estimaciones estadísticas basadas en datos históricos, no determinan el resultado real de un caso individual.")
