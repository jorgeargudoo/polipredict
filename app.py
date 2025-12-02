import math
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Polipredict - UPV",
    page_icon="🎓",
    layout="wide"
)

UPV_RED = "#c0392b"
UPV_GREY = "#f4f4f4"

custom_css = f"""
<style>
    .main {{
        background-color: white;
    }}
    .sidebar .sidebar-content {{
        background-color: {UPV_GREY};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}
    .polipredict-title {{
        color: {UPV_RED};
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }}
    .polipredict-subtitle {{
        color: #555555;
        font-size: 1rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background-color: {UPV_GREY};
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border-left: 5px solid {UPV_RED};
        margin-bottom: 0.5rem;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("polipredict_gb_model.pkl")
    return model

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("indicadores_doctorado_grupos.csv")
        return df
    except FileNotFoundError:
        return None

model = load_model()
df_raw = load_data()

col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.write("🎓")
with col_title:
    st.markdown('<div class="polipredict-title">PoliPredict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="polipredict-subtitle">'
        'Predicción de tesis matriculadas y estimación de recursos para programas de doctorado.'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

st.sidebar.title("Mi Polipredict")

st.sidebar.markdown("### Parámetros de entrada")

if df_raw is not None and "GRUPO_TITULACION" in df_raw.columns:
    grupos = sorted(df_raw["GRUPO_TITULACION"].dropna().unique().tolist())
else:
    grupos = ["Grupo 1", "Grupo 2", "Grupo 3"]

grupo_titulacion = st.sidebar.selectbox(
    "Grupo / Programa de doctorado",
    options=grupos
)

if df_raw is not None and "CURSO" in df_raw.columns:
    cursos = sorted(df_raw["CURSO"].dropna().unique().tolist())
    curso = st.sidebar.selectbox("Curso académico (código numérico)", options=cursos)
else:
    curso = st.sidebar.number_input("Curso académico (ej. 2022)", value=2022, step=1)

st.sidebar.markdown("##### Indicadores del año anterior")

tesis_lag = st.sidebar.number_input(
    "Tesis matriculadas el curso anterior (mismo programa)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

satis_pdi_lag = st.sidebar.slider(
    "Satisfacción PDI (año anterior, 0–10)",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

satis_alum_lag = st.sidebar.slider(
    "Satisfacción alumnado (año anterior, 0–10)",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1
)

abandono_lag = st.sidebar.slider(
    "Tasa de abandono (%) año anterior",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

btn_pred = st.sidebar.button("Calcular predicción y recursos", type="primary")


def estimate_resources(n_theses):
    ratio_tesis_prof = 3.0
    horas_tutoria_tesis = 20.0
    duracion_media = 4.0  # años
    porc_puesto_trabajo = 0.6
    coste_medio_tesis = 1500.0  # €/año
    factor_admin = 1.5
    porc_usa_lab = 0.7

    profesores_equiv = n_theses / ratio_tesis_prof
    horas_totales = n_theses * horas_tutoria_tesis
    comites_seguimiento = n_theses
    defensas_est = n_theses / duracion_media
    puestos_trabajo = n_theses * porc_puesto_trabajo
    coste_total = n_theses * coste_medio_tesis
    expedientes = n_theses * factor_admin
    usuarios_lab = n_theses * porc_usa_lab

    return {
        "profesores_equiv": profesores_equiv,
        "horas_totales": horas_totales,
        "comites_seguimiento": comites_seguimiento,
        "defensas_est": defensas_est,
        "puestos_trabajo": puestos_trabajo,
        "coste_total": coste_total,
        "expedientes": expedientes,
        "usuarios_lab": usuarios_lab
    }

if btn_pred:
    # Crear DataFrame con un único caso
    input_df = pd.DataFrame({
        "TESIS_LAG": [tesis_lag],
        "SATIS_PDI_LAG": [satis_pdi_lag],
        "SATIS_ALUM_LAG": [satis_alum_lag],
        "ABANDONO_LAG": [abandono_lag],
        "GRUPO_TITULACION": [grupo_titulacion],
        "CURSO": [curso]
    })

    pred = model.predict(input_df)[0]
    pred_rounded = max(0, round(pred))

    st.subheader("📊 Predicción de tesis matriculadas")

    st.markdown(
        f"""
        <div class="metric-card">
            <b>Programa:</b> {grupo_titulacion} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Curso:</b> {curso}<br>
            <span style="font-size: 1.2rem;">
            Tesis previstas: <b>{pred_rounded}</b> (valor modelo: {pred:.2f})
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    res = estimate_resources(pred_rounded)

    st.subheader("📌 Estimación general de recursos")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Profesores equivalentes</b><br>
                ≈ {res['profesores_equiv']:.1f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Horas de tutoría/año</b><br>
                ≈ {res['horas_totales']:.0f} h
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Comités de seguimiento</b><br>
                ≈ {res['comites_seguimiento']:.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Defensas de tesis/año (aprox.)</b><br>
                ≈ {res['defensas_est']:.1f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Puestos de trabajo para doctorandos</b><br>
                ≈ {res['puestos_trabajo']:.0f}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Usuarios de laboratorio adicionales</b><br>
                ≈ {res['usuarios_lab']:.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Coste total anual estimado</b><br>
                ≈ {res['coste_total']:,.0f} € 
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Expedientes/gestiones administrativas</b><br>
                ≈ {res['expedientes']:.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("Configura los parámetros en la barra lateral y pulsa **«Calcular predicción y recursos»**.")

