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
model_res = joblib.load('model_resolucion.pkl')            # o model_resolucion.pkl
label_encoders_res = joblib.load('label_encoders_res.pkl')
scaler_res = joblib.load('scaler_res.pkl')

# Modelo 2: vivo vs fallecido (entre resueltos)
model_vivo = joblib.load('model_vivo.pkl')
label_encoders_v = joblib.load('label_encoders_v.pkl')
scaler_v = joblib.load('scaler_v.pkl')

# ============ Entradas del usuario ============

# --- Crear pestañas ---
tab1, tab2 = st.tabs(["🤖 Predicción de Situación", "📊 Estadísticas Generales"])



# ============================================================
# 🤖 TAB 1 — Predicción de Situación
# ============================================================
with tab1:
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
    
    st.caption("Estas probabilidades son estimaciones estadísticas basadas en datos históricos,no determinan el resultado real de un caso individual.")



# ============================================================
# 📊 TAB 2 — Estadísticas Generales
# ============================================================

with tab2:
    st.header("Resumen de Desapariciones (2017–2024)")

    st.markdown("**Rango temporal de desapariciones:** 2017-01-01 → 2024-12-31")

    st.subheader("Top 10 Provincias con más desapariciones")
    st.code("""
PICHINCHA                         16668
GUAYAS                            15815
MANABI                             3346
AZUAY                              3189
EL ORO                             2921
SANTO DOMINGO DE LOS TSACHILAS     2828
LOS RIOS                           2812
CHIMBORAZO                         2615
TUNGURAHUA                         2455
COTOPAXI                           1969
""")

    st.subheader("Años con mayor número de desapariciones")
    st.code("""
2017    10457
2018    10255
2019     9962
2020     6762
2021     7955
2022     7721
2023     7808
2024     7009
""")

    st.subheader("Motivos de desaparición más frecuentes")
    st.code("""
CAUSAS FAMILIARES                                        47313
CAUSAS SOCIALES                                           5277
EXTRAVIADO - DISCAPACIDAD / ENFERMEDADES / TRASTORNOS     3981
CAUSAS PERSONALES                                         3444
FALLECIDO                                                 2186
EXTRAVIADO - AUSENCIA TEMPORAL                            2024
DESCONOCIDO                                               1904
CAUSAS ACADÉMICAS                                         1246
CERRADO POR FISCALÍA / DELITO REFORMULADO                  245
VIOLENCIA                                                  147
""")

    st.subheader("Edades más comunes de personas desaparecidas")
    st.code("""
15    8403
16    7581
14    6789
17    5593
13    4255
18    2673
19    2007
12    1771
20    1564
21    1392
""")

    st.subheader("Distribución por rango de edad")
    st.code("""
ADOLESCENTES     34391
ADULTOS          26694
NIÑOS(AS)         4223
ADULTO MAYOR      2621
""")

    st.subheader("Distribución por nacionalidad")
    st.code("""
ECUADOR                 65615
VENEZUELA                1149
COLOMBIA                  638
DESCONOCIDO               264
PERU                       76
... (otros países con menor frecuencia)
""")

    st.subheader("Distribución por etnia")
    st.code("""
MESTIZO/A      58893
INDIGENA        3484
AFRO            1873
BLANCO/A        1140
OTROS            871
MONTUBIO/A       783
MULATO/A         683
DESCONOCIDO      170
ASIATICO/A        32
""")

    st.subheader("Distribución por sexo")
    st.code("""
MUJER     42981
HOMBRE    24948
""")
