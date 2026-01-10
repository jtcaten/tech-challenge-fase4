import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa",
    layout="wide"
)

st.title("📈 Previsão do Ibovespa")
st.write("Aplicação desenvolvida para o Tech Challenge – Fase 4")

# =========================
# CAMINHOS
# =========================
DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")
MODEL_PATH = Path("model/modelo_ibov.pkl")

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)

    # Normalizar nomes das colunas
    df.columns = df.columns.str.strip()

    # Converter coluna de data (formato brasileiro)
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Criar coluna Fechamento a partir de "Último"
    if "Último" in df.columns:
        df["Fechamento"] = (
            df["Último"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    else:
        st.error("Coluna 'Último' não encontrada no CSV.")
        st.stop()

    df = df.dropna(subset=["Data", "Fechamento"])
    df = df.sort_values("Data")

    return df


# =========================
# CARREGAR MODELO
# =========================
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


# =========================
# EXECUÇÃO
# =========================
df = carregar_dados()
modelo = carregar_modelo()

# =========================
# FEATURE ENGINEERING
# =========================
df["log_return"] = np.log(df["Fechamento"]).diff()
df = df.dropna()

# =========================
# VISUALIZAÇÃO DOS DADOS
# =========================
st.subheader("📊 Série Histórica do Ibovespa")

fig, ax = plt.subplots()
ax.plot(df["Data"], df["Fechamento"])
ax.set_xlabel("Data")
ax.set_ylabel("Ibovespa")
ax.grid(True)

st.pyplot(fig)

# =========================
# PREVISÃO
# =========================
st.subheader("🔮 Previsão do Próximo Log-Return")

ultimo_valor = df["log_return"].iloc[-1]
X_input = np.array([[ultimo_valor]])

previsao = modelo.predict(X_input)[0]

st.metric(
    label="Log-return previsto",
    value=f"{previsao:.6f}"
)

st.caption("Modelo treinado na Fase 2 e aplicado em ambiente Streamlit Cloud.")
