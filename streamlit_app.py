# --- Librerías necesarias ---
import streamlit as st
import fitz  # PyMuPDF para leer PDFs
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Descripción del puesto objetivo ---
DESCRIPCION_PUESTO = """
Se requiere un Ingeniero en Sistemas Informáticos con conocimientos en:
- Desarrollo de software backend y frontend
- Manejo de bases de datos SQL y NoSQL
- Programación en Python, Java o C#
- Uso de servicios en la nube como AWS o Azure
- Buenas prácticas de seguridad informática
- Trabajo en equipo y metodologías ágiles como Scrum
"""

# --- Función para extraer texto de archivos PDF ---
def extraer_texto_pdf(archivo_pdf):
    texto = ""
    with fitz.open(stream=archivo_pdf.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto

# --- Función IA: Procesamiento de textos y cálculo de similitud ---
def calcular_similitud(textos_cv, texto_referencia):
    vectorizer = TfidfVectorizer(stop_words='english')  # IA: vectoriza los textos
    vectores = vectorizer.fit_transform([texto_referencia] + textos_cv)
    similitud = cosine_similarity(vectores[0:1], vectores[1:]).flatten()  # IA: calcula similitud coseno
    return similitud

# --- Interfaz de usuario con Streamlit ---
st.set_page_config(page_title="Reclutamiento IA", layout="centered")
st.title("IDEA ESPOCH 2025")
st.title("🤖 Prototipo Sistema de Selección y Reclutamiento con Inteligencia Artificial")
st.markdown("Este sistema compara CVs con el perfil de un Ingeniero en Sistemas Informáticos.")

# Mostrar descripción del puesto
st.subheader("📋 Descripción del Cargo")
st.text_area("Puesto: Ingeniero en Sistemas Informáticos", value=DESCRIPCION_PUESTO, height=200, disabled=True)

# Cargar CVs en PDF
archivos = st.file_uploader("📎 Sube uno o más CVs en formato PDF", type="pdf", accept_multiple_files=True)

# Botón para analizar los candidatos
if st.button("🔍 Analizar Candidatos"):
    if not archivos:
        st.warning("⚠️ Debes subir al menos un archivo PDF.")
    else:
        textos_cv = []
        nombres_cv = []

        # Leer todos los CVs
        for archivo in archivos:
            texto_cv = extraer_texto_pdf(archivo)
            textos_cv.append(texto_cv)
            nombres_cv.append(archivo.name)

        # IA: calcular similitud con el perfil
        similitudes = calcular_similitud(textos_cv, DESCRIPCION_PUESTO)

        # Crear un DataFrame con los resultados
        resultados = pd.DataFrame({
            "Candidato": nombres_cv,
            "Coincidencia con el perfil (%)": (similitudes * 100).round(2)
        }).sort_values(by="Coincidencia con el perfil (%)", ascending=False)

        # Mostrar resultados
        st.success("✅ Análisis completado. Aquí está el ranking de los candidatos:")
        st.dataframe(resultados)