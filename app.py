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

UPV_RED = "#c0392b"
UPV_GREY = "#f4f4f4"

custom_css = f"""
<style>
    .block-container {{
        padding-top: 4rem !important;
        padding-bottom: 2rem;
    }}
    .polipredict-title {{
        color: {UPV_RED};
        font-weight: 700;
        font-size: 2.4rem;
        margin-top: 1rem;
        margin-bottom: 0.2rem;
    }}
    .polipredict-subtitle {{
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background-color: {UPV_GREY};
        padding: 0.9rem 1.2rem;
        border-radius: 0.75rem;
        border-left: 5px solid {UPV_RED};
        margin-bottom: 0.7rem;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ======================================================================
# CARGA MODELO Y DATOS
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
# CABECERA
# ======================================================================
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.write("🎓")
with col_title:
    st.markdown('<div class="polipredict-title">PoliPredict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="polipredict-subtitle">Predicción de tesis matriculadas y estimación de recursos para programas de doctorado.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ======================================================================
# PANEL LATERAL
# ======================================================================
st.sidebar.title("Mi Polipredict")
st.sidebar.markdown("### Parámetros de entrada")

# Extraemos valores reales del dataset
if df_raw is not None:
    grupos = sorted(df_raw["GRUPO_TITULACION"].dropna().unique().tolist())
    cursos = sorted(df_raw["CURSO"].dropna().unique().astype(str).tolist())
else:
    grupos = ["Grupo 1", "Grupo 2"]
    cursos = ["2020-21", "2021-22"]

grupo_titulacion = st.sidebar.selectbox(
    "Grupo / Programa de doctorado",
    options=grupos
)

curso = st.sidebar.selectbox("Curso académico (texto)", options=cursos)

st.sidebar.markdown("##### Indicadores del año anterior")

tesis_lag = st.sidebar.number_input(
    "Tesis matriculadas el curso anterior",
    min_value=0.0, value=10.0, step=1.0
)

satis_pdi_lag = st.sidebar.slider("Satisfacción PDI (0–10)", 0.0, 10.0, 7.0)
satis_alum_lag = st.sidebar.slider("Satisfacción alumnado (0–10)", 0.0, 10.0, 7.5)
abandono_lag = st.sidebar.slider("Tasa de abandono (%)", 0.0, 100.0, 10.0)

btn_pred = st.sidebar.button("Calcular predicción y recursos", type="primary")

# ======================================================================
# FUNCIÓN DE RECURSOS
# ======================================================================
def estimate_resources(n_theses):
    ratio_tesis_prof = 3.0
    horas_tutoria_tesis = 20.0
    duracion_media = 4.0
    porc_puesto_trabajo = 0.6
    coste_medio_tesis = 1500.0
    factor_admin = 1.5
    porc_lab = 0.7

    return {
        "profesores_equiv": n_theses / ratio_tesis_prof,
        "horas_totales": n_theses * horas_tutoria_tesis,
        "comites": n_theses,
        "defensas": n_theses / duracion_media,
        "puestos": n_theses * porc_puesto_trabajo,
        "coste": n_theses * coste_medio_tesis,
        "expedientes": n_theses * factor_admin,
        "usuarios_lab": n_theses * porc_lab
    }

# ======================================================================
# ZONA PRINCIPAL DE RESULTADOS
# ======================================================================
if btn_pred:

    input_df = pd.DataFrame({
        "TESIS_LAG": [tesis_lag],
        "SATIS_PDI_LAG": [satis_pdi_lag],
        "SATIS_ALUM_LAG": [satis_alum_lag],
        "ABANDONO_LAG": [abandono_lag],
        "GRUPO_TITULACION": [grupo_titulacion],
        "CURSO": [str(curso)]
    })

    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error("⚠️ El modelo no pudo transformar las categorías seleccionadas. "
                 "Seguramente ‘CURSO’ o ‘GRUPO_TITULACION’ contienen una categoría "
                 "que no existía en el conjunto de entrenamiento.")
        st.code(str(e))
        st.stop()

    pred_rounded = max(0, round(pred))

    st.subheader("📊 Predicción de tesis matriculadas")
    st.markdown(
        f"""
        <div class="metric-card">
        Tesis previstas: <b>{pred_rounded}</b>  
        <br>Valor del modelo: {pred:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Recursos
    r = estimate_resources(pred_rounded)

    st.subheader("📌 Estimación de recursos")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='metric-card'><b>Profesores equivalentes:</b> {r['profesores_equiv']:.1f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Horas de tutoría:</b> {r['horas_totales']:.0f}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='metric-card'><b>Puestos de trabajo:</b> {r['puestos']:.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Usuarios laboratorio:</b> {r['usuarios_lab']:.0f}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='metric-card'><b>Coste anual estimado:</b> {r['coste']:.0f} €</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><b>Expedientes administrativos:</b> {r['expedientes']:.0f}</div>", unsafe_allow_html=True)

else:
    st.info("Configura los parámetros en la barra lateral y pulsa «Calcular predicción y recursos».")
