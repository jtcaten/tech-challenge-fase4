import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa (Completo)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ESTILO
# =========================
st.markdown(
    """
    <style>
        .main-header {
            text-align: center;
            color: #38bdf8;
            font-size: 3rem;
            font-weight: bold;
            margin: 2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-header">📊 Ibovespa Dashboard Completo</h1>',
    unsafe_allow_html=True,
)
st.write("Aplicação desenvolvida para o Tech Challenge – Fase 4")

# =========================
# CAMINHOS (MESMA ESTRUTURA ATUAL)
# =========================
DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")
MODEL_PATH = Path("model/modelo_ibov.pkl")

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)

    # Normalizar colunas
    df.columns = df.columns.str.strip()

    # Converter data
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    # Criar Fechamento
    if "Último" not in df.columns:
        st.error("Coluna 'Último' não encontrada no CSV.")
        st.stop()

    df["Fechamento"] = (
        df["Último"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Fechamento"] = pd.to_numeric(df["Fechamento"], errors="coerce")

    df = df.dropna(subset=["Data", "Fechamento"])
    df = df.sort_values("Data")

    # Renomear para padrão usado nas visualizações avançadas
    df = df.rename(columns={"Data": "date", "Fechamento": "close"})

    return df


# =========================
# CARREGAMENTO DO MODELO
# =========================
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


# =========================
# EXECUÇÃO – CARGA
# =========================
df = carregar_dados()
modelo = carregar_modelo()

# =========================
# FEATURE ENGINEERING
# =========================
df["log_return"] = np.log(df["close"]).diff()
df_lr = df.dropna(subset=["log_return"])


def create_features(df_temp: pd.DataFrame) -> pd.DataFrame:
    """Features simples para visualização, inspiradas no painel completo."""
    df_feat = df_temp.copy()

    # Retornos simples
    df_feat["returns"] = df_feat["close"].pct_change()

    # Médias móveis
    df_feat["ma5"] = df_feat["close"].rolling(5).mean()
    df_feat["ma20"] = df_feat["close"].rolling(20).mean()
    df_feat["ma50"] = df_feat["close"].rolling(50).mean()

    # Volatilidade (desvio padrão móvel dos retornos)
    df_feat["volatility"] = df_feat["returns"].rolling(20).std()

    return df_feat


df_feat = create_features(df).dropna()

# =========================
# SIDEBAR – CONTROLES
# =========================
with st.sidebar:
    st.header("⚙️ Configurações")
    
    n_points = st.slider(
        "Número de dias para mostrar",
        min_value=30,
        max_value=len(df),
        value=365,
        step=30
    )
    
    df_filtered = df.tail(n_points)

# =========================
# MÉTRICAS SUPERIORES
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    if len(df) >= 2:
        variacao = (
            (df["close"].iloc[-1] - df["close"].iloc[-2])
            / df["close"].iloc[-2]
            * 100
        )
        st.metric(
            "Última Cotação",
            f"{df['close'].iloc[-1]:,.0f}",
            f"{variacao:.2f}%",
        )
    else:
        st.metric("Última Cotação", "N/A")

with col2:
    if len(df_lr) < 1:
        st.metric("Log-return previsto", "N/A")
    else:
        ultimo_valor = df_lr["log_return"].iloc[-1]
        X_input = np.array([[ultimo_valor]])
        previsao = modelo.predict(X_input)[0]
        st.metric("Log-return previsto", f"{previsao:.6f}")

with col3:
    st.metric("Data", df["date"].iloc[-1].strftime("%d/%m/%Y"))

st.divider()

# =========================
# TABS PRINCIPAIS
# =========================
tab1, tab2, tab3 = st.tabs(
    ["📈 Série Histórica", "📊 Indicadores", "📋 Dados"]
)

# -------------------------
# TAB 1 – SÉRIE HISTÓRICA
# -------------------------
with tab1:
    st.subheader("Série Histórica com Médias Móveis")

    fig = go.Figure()

    # Série de preços filtrada
    fig.add_trace(
        go.Scatter(
            x=df_filtered["date"],
            y=df_filtered["close"],
            name="Ibovespa",
            line=dict(color="#38bdf8", width=2),
        )
    )

    # Filtrar features para mesmo período
    if not df_filtered.empty:
        df_feat_filtered = df_feat[
            df_feat["date"].between(
                df_filtered["date"].min(),
                df_filtered["date"].max(),
            )
        ]
    else:
        df_feat_filtered = df_feat

    # Médias móveis
    fig.add_trace(
        go.Scatter(
            x=df_feat_filtered["date"],
            y=df_feat_filtered["ma5"],
            name="MA5",
            line=dict(color="#fbbf24", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_feat_filtered["date"],
            y=df_feat_filtered["ma20"],
            name="MA20",
            line=dict(color="#f87171"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_feat_filtered["date"],
            y=df_feat_filtered["ma50"],
            name="MA50",
            line=dict(color="#10b981"),
        )
    )

    fig.update_layout(
        title="Ibovespa - Série Histórica com Médias Móveis",
        xaxis_title="Data",
        yaxis_title="Índice",
        template="plotly_dark",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 2 – INDICADORES
# -------------------------
with tab2:
    st.subheader("Indicadores Técnicos - Últimos 100 dias")

    df_recent = df_feat.tail(100)

    if df_recent.empty:
        st.warning("Ainda não há dados suficientes para indicadores.")
    else:
        fig2 = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Preço", "Retornos", "Volatilidade"),
            vertical_spacing=0.1,
        )

        # Preço
        fig2.add_trace(
            go.Scatter(
                x=df_recent["date"],
                y=df_recent["close"],
                name="Ibov",
                line=dict(color="#38bdf8"),
            ),
            row=1,
            col=1,
        )

        # Retornos
        fig2.add_trace(
            go.Scatter(
                x=df_recent["date"],
                y=df_recent["returns"],
                name="Retornos",
                line=dict(color="#fbbf24"),
            ),
            row=2,
            col=1,
        )

        # Volatilidade
        fig2.add_trace(
            go.Bar(
                x=df_recent["date"],
                y=df_recent["volatility"],
                name="Volatilidade",
                marker_color="#f87171",
            ),
            row=3,
            col=1,
        )

        fig2.update_layout(
            title="Indicadores Técnicos",
            template="plotly_dark",
            height=800,
            showlegend=True,
        )

        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# TAB 3 – DADOS
# -------------------------
with tab3:
    st.subheader("Dados Históricos")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"**Total de dados:** {len(df)} dias")
        st.write(
            f"**Período:** {df['date'].min().date()} "
            f"até {df['date'].max().date()}"
        )
        st.write(f"**Preço mínimo:** {df['close'].min():,.0f}")
        st.write(f"**Preço máximo:** {df['close'].max():,.0f}")

    with col_b:
        st.write("**Modelo:** Modelo Ibov Fase 2")
        st.write("**Feature de previsão:** log_return (t-1)")
        st.write("**Fonte dos dados:** CSV Ibovespa 2005-2025")

    st.write("**Últimas 10 linhas:**")
    st.dataframe(df.tail(10), use_container_width=True)

# =========================
# RODAPÉ
# =========================
st.caption(
    "Versão completa baseada no app funcional original, "
    "com visualização aprimorada."
)
