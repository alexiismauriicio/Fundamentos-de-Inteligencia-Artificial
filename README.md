# Fundamentos-de-Inteligencia-Artificial
Proyecto Final FIA

# Predicción de situación de personas desaparecidas en Ecuador

Este proyecto utiliza **modelos de Machine Learning** para estimar, a partir de características demográficas, la probabilidad de que un caso de persona desaparecida en Ecuador:

- 🔵 Se **resuelva** (la persona sea localizada, viva o fallecida)  
- 🟢 Sea **encontrada con vida**  
- 🟠 Sea **encontrada fallecida**  
- 🔴 **Siga desaparecida**  

Además, incluye un **asistente de IA (chatbot)** que explica las estadísticas del dataset y la interpretación de las probabilidades de manera controlada y ética.

---

## 🚀 Demo en Streamlit

La aplicación final está desplegada en **Streamlit Community Cloud**:

👉 **[Abrir la app](https://fundamentos-de-inteligencia-artificial-7frgrz76mh3bvqdde2ati3.streamlit.app/)**  

No requiere instalación local: basta con abrir el enlace en el navegador.

---

## 🧩 ¿Cómo usar la aplicación?

### 1. Pestaña: 🤖 *Predicción de Situación*

En la primera pestaña podrás:

1. **Ingresar las características demográficas** de un caso hipotético:
   - Sexo  
   - Provincia  
   - Nacionalidad  
   - Etnia  
   - Edad aproximada  

2. Pulsar el botón **"Calcular probabilidades"**.  

3. Ver los resultados:
   - Probabilidad de que el caso se **resuelva**.  
   - Probabilidad de que la persona **siga desaparecida**.  
   - Probabilidad de ser **encontrada viva**.  
   - Probabilidad de ser **encontrada fallecida**.  
   - Un **gráfico de barras** con estas probabilidades.

4. Opcionalmente, escribir una pregunta al **asistente de IA**:
   - El chatbot solo responde sobre:
     - las estadísticas del dataset,  
     - cómo funciona el modelo,  
     - cómo interpretar las probabilidades.  
   - Si se le pregunta algo fuera de contexto, lo indica y no responde sobre otros temas.  
   - Si se requiere información oficial, remite a la página de:
     - http://www.desaparecidosecuador.gob.ec/presentacion

---

### 2. Pestaña: 📊 *Estadísticas Generales*

En la segunda pestaña se muestran, a modo de resumen:

- Top 10 provincias con más desapariciones.  
- Años con mayor número de casos.  
- Motivos de desaparición más frecuentes.  
- Distribución por edad, rango de edad, nacionalidad, etnia y sexo.  

Estas estadísticas ayudan a contextualizar los resultados y entender el comportamiento histórico de los datos.

---

## 🧠 Modelos utilizados

- **Modelo 1:** Clasificador binario (XGBoost) para predecir si un caso se **resuelve** o si la persona **sigue desaparecida**.  
- **Modelo 2:** Clasificador binario (XGBoost) entrenado solo con casos resueltos, para distinguir entre:
  - **Encontrado vivo**  
  - **Encontrado fallecido**

Las probabilidades finales mostradas en la app son una combinación de ambos modelos.

---

## ⚠️ Descargo de responsabilidad (muy importante)

- Este proyecto es **exclusivamente académico**.  
- Las probabilidades mostradas son **estimaciones estadísticas basadas en datos históricos (2017–2024)**.  
- **No deben utilizarse** para:
  - Tomar decisiones sobre casos reales.  
  - Comunicar resultados a familiares, autoridades u otras personas involucradas.  
- El tema de personas desaparecidas es sensible, por lo que la app prioriza un uso responsable y explicativo.

Para información oficial, consultar:
👉 http://www.desaparecidosecuador.gob.ec/presentacion

---
