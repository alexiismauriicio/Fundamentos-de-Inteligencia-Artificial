import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI
import json
import textwrap

# ---------------------------------------------------------
# Configuración general de la app
# ---------------------------------------------------------
st.set_page_config(page_title='Predicción de Personas Desaparecidas', layout='wide')
st.title('🔍 Predicción de situación de personas desaparecidas')

st.markdown("""
Esta sección estima, a partir de características demográficas, las probabilidades de que:
- el caso se **resuelva** (la persona sea localizada),
- la persona **siga desaparecida**,
- y en caso de ser localizada, sea **encontrada con vida** o **fallecida**.

> ⚠️ Estas probabilidades son estimaciones estadísticas basadas en datos históricos,
> no determinan el resultado real de un caso individual.
""")

# ---------------------------------------------------------
# Cliente OpenAI (usa tu API key desde .streamlit/secrets.toml)
# ---------------------------------------------------------
# En Streamlit Cloud, en "Secrets" define:
# OPENAI_API_KEY = "tu_clave_aquí"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GPT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------
# Funciones para metadatos del dataset
# ---------------------------------------------------------
def cargar_metadata(path: str = "metadata_mdi_personas_desaparecidas_pm_historico_2014_2024.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta
    except FileNotFoundError:
        return None

def construir_resumen_desde_metadata(meta):
    if meta is None:
        return (
            "No se pudo cargar el archivo de metadatos. "
            "Solo se dispone de la información estadística básica calculada en el notebook."
        )

    zona_info = meta.get("zona", {})
    provincia_info = meta.get("provincia", {})
    sexo_info = meta.get("sexo", {})
    nacionalidad_info = meta.get("nacionalidad", {})
    rango_edad_info = meta.get("rango_edad", {})
    etnia_info = meta.get("etnia", {})
    motivo_info = meta.get("motivo_desaparicion", {})
    situacion_info = meta.get("situacion_actual", {})
    fecha_desap_info = meta.get("fecha_desaparicion", {})

    total_registros = zona_info.get("non_null", "desconocido")
    provincias = provincia_info.get("unique", "desconocido")
    sexos = sexo_info.get("sample", [])
    nacionalidades = nacionalidad_info.get("sample", [])
    rangos_edad = rango_edad_info.get("sample", [])
    etnias = etnia_info.get("sample", [])
    motivos = motivo_info.get("sample", [])
    situaciones = situacion_info.get("sample", [])
    fecha_min = fecha_desap_info.get("min_date", "")
    fecha_max = fecha_desap_info.get("max_date", "")

    resumen = f"""
    Metadatos del dataset histórico de personas desaparecidas en Ecuador (2014–2024):

    - Número aproximado de registros: {total_registros}
    - Número de provincias distintas: {provincias}
    - Ejemplos de valores de sexo: {', '.join(sexos)}
    - Ejemplos de nacionalidades: {', '.join(nacionalidades)}
    - Rangos de edad registrados (ejemplos): {', '.join(rangos_edad)}
    - Etnias registradas (ejemplos): {', '.join(etnias)}
    - Motivos de desaparición (ejemplos): {', '.join(motivos)}
    - Situaciones finales posibles: {', '.join(situaciones)}
    - Rango de fechas de desaparición en los datos: {fecha_min} a {fecha_max}

    El modelo de machine learning utiliza variables como sexo, provincia,
    nacionalidad, etnia y edad aproximada para estimar probabilidades de:
    - que el caso se resuelva (encontrado o fallecido),
    - que la persona siga desaparecida,
    - que, en caso de ser localizada, sea encontrada viva o fallecida.

    Estas estimaciones se basan únicamente en patrones estadísticos del
    historial 2014–2024 y no determinan el resultado de casos individuales.
    """
    return textwrap.dedent(resumen).strip()

metadata = cargar_metadata()
resumen_metadata = construir_resumen_desde_metadata(metadata)

# ---------------------------------------------------------
# Chatbot restringido al contexto del proyecto
# ---------------------------------------------------------
def obtener_respuesta_chat(
    pregunta_usuario: str,
    resumen_metadata: str,
    contexto_prediccion: str | None = None,
) -> str:
    """
    Chatbot que SOLO responde sobre:
    - estadísticas del dataset,
    - funcionamiento del modelo,
    - interpretación de las probabilidades calculadas.

    Si la pregunta está fuera de contexto, lo indica explícitamente.
    """

    system_prompt = """
    Eres un asistente que explica un proyecto académico de machine learning
    sobre personas desaparecidas en Ecuador.

    SOLO puedes hablar sobre:
    - estadísticas del dataset usado (metadatos, distribución general de casos),
    - cómo funciona el modelo de predicción a alto nivel,
    - interpretación de las probabilidades que el modelo calcula
      (probabilidad de que el caso se resuelva, que siga desaparecido,
       probabilidad de ser encontrado vivo o fallecido),
    - limitaciones y consideraciones éticas de usar modelos estadísticos
      con este tipo de información sensible.

    NO puedes:
    - dar consejos operativos para casos reales,
    - opinar o especular sobre un caso real concreto,
    - ofrecer ayuda legal, policial, psicológica o médica,
    - responder preguntas de temas no relacionados con este proyecto.

    Si la pregunta está fuera de contexto, responde de forma breve algo como:
    "Solo puedo responder preguntas relacionadas con las estadísticas del dataset,
    el funcionamiento del modelo y la interpretación de las probabilidades
    de este proyecto académico."

    Si el usuario necesita información oficial o actualizada sobre personas desaparecidas
    en Ecuador, recomiéndale visitar:
    http://www.desaparecidosecuador.gob.ec/presentacion

    Responde SIEMPRE en español, con tono claro y respetuoso, reconociendo
    que el tema es sensible. No des falsas certezas ni promesas.
    """

    contexto = f"Metadatos del dataset:\n{resumen_metadata}\n\n"
    if contexto_prediccion:
        contexto += f"Contexto de la predicción actual:\n{contexto_prediccion}\n\n"

    mensajes = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": contexto + "\n\nPregunta del usuario: " + pregunta_usuario,
        },
    ]

    respuesta = client.chat.completions.create(
        model=GPT_MODEL,
        messages=mensajes,
        temperature=0.2,
    )

    return respuesta.choices[0].message.content

# ---------------------------------------------------------
# Cargar modelos y transformadores
# ---------------------------------------------------------

# Modelo 1: caso resuelto vs desaparecido
model_res = joblib.load('model_resolucion.pkl')
label_encoders_res = joblib.load('label_encoders_res.pkl')
scaler_res = joblib.load('scaler_res.pkl')

# Modelo 2: vivo vs fallecido (entre resueltos)
model_vivo = joblib.load('model_vivo.pkl')
label_encoders_v = joblib.load('label_encoders_v.pkl')
scaler_v = joblib.load('scaler_v.pkl')

# ---------------------------------------------------------
# Pestañas
# ---------------------------------------------------------
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
        p_resuelto = float(model_res.predict_proba(X_res)[:, 1][0])
        p_desaparecido = 1 - p_resuelto

        # ========= 2) Transformaciones para modelo vivo vs fallecido =========
        X_v = X_input.copy()
        for col in ['sexo', 'provincia', 'nacionalidad', 'etnia']:
            le_v = label_encoders_v[col]
            X_v[col] = le_v.transform(X_v[col])

        X_v['edad_aproximada'] = scaler_v.transform(X_v[['edad_aproximada']])

        p_vivo_cond = float(model_vivo.predict_proba(X_v)[:, 1][0])  # P(vivo | resuelto)
        p_fallecido_cond = 1 - p_vivo_cond

        # ========= 3) Probabilidades combinadas =========
        p_encontrado_vivo = p_resuelto * p_vivo_cond
        p_encontrado_fallecido = p_resuelto * p_fallecido_cond

        # Normalizar por seguridad numérica
        total = p_encontrado_vivo + p_encontrado_fallecido + p_desaparecido
        if total > 0:
            p_encontrado_vivo /= total
            p_encontrado_fallecido /= total
            p_desaparecido /= total

        # Guardar en session_state para que el chatbot pueda usar la última predicción
        st.session_state["ultima_prediccion"] = {
            "sexo": sexo,
            "provincia": provincia,
            "nacionalidad": nacionalidad,
            "etnia": etnia,
            "edad": edad,
            "p_resuelto": p_resuelto,
            "p_desaparecido": p_desaparecido,
            "p_encontrado_vivo": p_encontrado_vivo,
            "p_encontrado_fallecido": p_encontrado_fallecido,
        }

        st.subheader("Resultados")
        st.write(f"🔵 Probabilidad de que el caso se **resuelva**: **{p_resuelto:.2%}**")
        st.write(f"🟢 Probabilidad de ser encontrado **vivo**: **{p_encontrado_vivo:.2%}**")
        st.write(f"🟠 Probabilidad de ser encontrado **fallecido**: **{p_encontrado_fallecido:.2%}**")
        st.write(f"🔴 Probabilidad de que la persona **siga desaparecida**: **{p_desaparecido:.2%}**")

        st.caption(
            "Estas probabilidades son estimaciones estadísticas basadas en datos históricos (2014–2024); "
            "no determinan el resultado real de un caso individual."
        )

        st.bar_chart({
            "Prob. sigue desaparecida": [p_desaparecido],
            "Prob. encontrada viva": [p_encontrado_vivo],
            "Prob. encontrada fallecida": [p_encontrado_fallecido],
        })

    # ---------------- Chatbot sobre el modelo y las estadísticas ----------------
    st.markdown("---")
    st.subheader("🗨️ Asistente sobre el modelo y las estadísticas")

    st.caption(
        "Este asistente solo responde preguntas sobre las estadísticas del dataset, "
        "el funcionamiento del modelo y la interpretación de las probabilidades. "
        "No ofrece asesoría para casos reales."
    )

    if "ultima_prediccion" not in st.session_state:
        st.info("Primero ingresa los datos y pulsa **Calcular probabilidades** para habilitar el asistente.")
    else:
        pregunta_chat = st.text_area(
            "Escribe una pregunta relacionada con este proyecto o estos resultados:"
        )

        if st.button("Preguntar al asistente") and pregunta_chat.strip():
            pred = st.session_state["ultima_prediccion"]
            contexto_prediccion = f"""
            Caso ingresado por el usuario:
            - Sexo: {pred['sexo']}
            - Provincia: {pred['provincia']}
            - Nacionalidad: {pred['nacionalidad']}
            - Etnia: {pred['etnia']}
            - Edad aproximada: {pred['edad']}

            Probabilidades estimadas por el modelo (aprox.):
            - Probabilidad de que el caso se resuelva (encontrado o fallecido): {pred['p_resuelto']:.2%}
            - Probabilidad de que la persona siga desaparecida: {pred['p_desaparecido']:.2%}
            - Probabilidad de que sea encontrada viva: {pred['p_encontrado_vivo']:.2%}
            - Probabilidad de que sea encontrada fallecida: {pred['p_encontrado_fallecido']:.2%}
            """

            with st.spinner("El asistente está analizando tu pregunta..."):
                respuesta_chat = obtener_respuesta_chat(
                    pregunta_usuario=pregunta_chat,
                    resumen_metadata=resumen_metadata,
                    contexto_prediccion=contexto_prediccion,
                )

            st.markdown("**Respuesta del asistente:**")
            st.write(respuesta_chat)

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
