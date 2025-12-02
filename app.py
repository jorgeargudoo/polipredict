import joblib
import pandas as pd
import streamlit as st

# ======================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ======================================================================
st.set_page_config(
    page_title="PoliPredict - UPV",
    page_icon="🎓",
    layout="wide"
)

UPV_RED = "#d84315"
UPV_GREY = "#f5f5f5"

custom_css = f"""
<style>
    .block-container {{
        padding-top: 3.5rem !important;
        padding-left: 6rem !important;
        padding-right: 3rem !important;
    }}

    .polipredict-title {{
        font-family: Arial, sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }}

    .predict-grey {{
        color: #777;
    }}
    .predict-red {{
        color: {UPV_RED};
    }}

    .polipredict-subtitle {{
        color: #555;
        font-size: 1.15rem;
        margin-bottom: 1.3rem;
        text-align: center;
    }}

    .metric-card {{
        background-color: {UPV_GREY};
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        border-left: 6px solid {UPV_RED};
        margin-bottom: 0.7rem;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ======================================================================
# CARGAR MODELO Y DATOS
# ======================================================================
@st.cache_resource
def load_model():
    return joblib.load("polipredict_gb_model.pkl")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("indicadores_doctorado_grupos.csv")
    except:
        return None

model = load_model()
df_raw = load_data()

# ======================================================================
# CABECERA ESTILO POLIFORMAT
# ======================================================================
st.markdown(
    """
    <div class="polipredict-title">
        <span class="predict-grey">Poli</span><span class="predict-red">[PredicT]</span>
    </div>
    <div class="polipredict-subtitle">
        Predicción de tesis matriculadas y estimación de recursos para programas de doctorado.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.title("Mi Polipredict")
st.sidebar.markdown("### Parámetros de entrada")

# Extraer categorías reales
if df_raw is not None:
    grupos = sorted(df_raw["GRUPO_TITULACION"].dropna().unique().tolist())
    cursos = sorted(df_raw["CURSO"].dropna().astype(str).unique().tolist())
else:
    grupos = ["Grupo 1", "Grupo 2"]
    cursos = ["2020-21", "2021-22"]

grupo_titulacion = st.sidebar.selectbox("Programa de doctorado", grupos)
curso = st.sidebar.selectbox("Curso académico", cursos)

tesis_lag = st.sidebar.number_input("Tesis año anterior", min_value=0.0, value=10.0)
satis_pdi_lag = st.sidebar.slider("Satisfacción PDI (0–10)", 0.0, 10.0, 7.0)
satis_alum_lag = st.sidebar.slider("Satisfacción alumnado (0–10)", 0.0, 10.0, 7.5)
abandono_lag = st.sidebar.slider("Tasa de abandono (%)", 0.0, 100.0, 10.0)

btn_pred = st.sidebar.button("Calcular predicción y recursos", type="primary")

# ======================================================================
# FUNCIÓN RECURSOS
# ======================================================================
def estimate_resources(n):
    return {
        "profesores": n / 3,
        "horas": n * 20,
        "puestos": n * 0.6,
        "lab": n * 0.7,
        "coste": n * 1500,
        "expedientes": n * 1.5
    }

# ======================================================================
# CUERPO PRINCIPAL
# ======================================================================
if btn_pred:

    input_df = pd.DataFrame({
        "TESIS_LAG": [tesis_lag],
        "SATIS_PDI_LAG": [satis_pdi_lag],
        "SATIS_ALUM_LAG": [satis_alum_lag],
        "ABANDONO_LAG": [abandono_lag],
        "GRUPO_TITULACION": [grupo_titulacion],
        "CURSO": [curso]
    })

    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error("⚠️ El modelo no pudo transformar las categorías seleccionadas. "
                 "Reentrena el modelo con handle_unknown='ignore'.")
        st.code(str(e))
        st.stop()

    pred_r = round(pred)

    st.subheader("📊 Predicción")
    st.markdown(
        f"<div class='metric-card'>Tesis previstas: <b>{pred_r}</b></div>",
        unsafe_allow_html=True
    )

    r = estimate_resources(pred_r)

    st.subheader("📌 Recursos estimados")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='metric-card'><b>Profesores:</b> {r['profesores']:.1f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Puestos:</b> {r['puestos']:.0f}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='metric-card'><b>Horas tutoría:</b> {r['horas']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Laboratorio:</b> {r['lab']:.0f}</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='metric-card'><b>Coste:</b> {r['coste']} €</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Expedientes:</b> {r['expedientes']}</div>", unsafe_allow_html=True)

else:
    st.info("Ajusta los parámetros y pulsa **Calcular predicción y recursos**")
