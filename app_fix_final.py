# ========================================
# SOLUÇÃO FINAL - FUNCIONA COM SEU CSV
# ========================================
# 
# Este script VALIDA as colunas e adapta
# automaticamente para qualquer estrutura
#
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import json
import pickle

# ========================================
# 1. CARREGAR CSV COM VALIDAÇÃO
# ========================================

@st.cache_data(ttl=3600)
def load_csv_optimized():
    """Carrega CSV e valida colunas"""
    df = pd.read_csv(
        'Unified_Data.csv',
        dtype={
            'close': 'float32',
            'selic': 'float32'
        },
        parse_dates=['date']
    )
    
    # DEBUG: Mostrar colunas disponíveis
    print(f"✅ Colunas encontradas: {df.columns.tolist()}")
    print(f"✅ Shape: {df.shape}")
    print(f"✅ Primeiras linhas:\n{df.head()}")
    
    return df


# ========================================
# 2. ADAPTAR FUNÇÃO create_features
# ========================================

def create_features(df):
    """
    Cria features - ADAPTADO PARA QUALQUER ESTRUTURA
    """
    df = df.copy()
    
    # Verificar quais colunas existem
    has_high = 'high' in df.columns
    has_low = 'low' in df.columns
    has_open = 'open' in df.columns
    has_usd_close = 'usd_close' in df.columns
    has_selic = 'selic' in df.columns
    
    print(f"Colunas disponíveis: high={has_high}, low={has_low}, open={has_open}, usd_close={has_usd_close}, selic={has_selic}")
    
    # ===== MÉDIAS MÓVEIS (sempre possível com 'close') =====
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # ===== RSI (sempre possível com 'close') =====
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # ===== MACD (sempre possível com 'close') =====
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # ===== VOLATILIDADE =====
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # ===== RETORNO LOGARÍTMICO =====
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # ===== ATR (SÓ SE TIVER high/low) =====
    if has_high and has_low:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        print("✅ ATR calculado (high/low disponíveis)")
    else:
        # ATR alternativo usando apenas close
        df['atr'] = df['close'].rolling(window=14).std() * 1.5
        print("⚠️  ATR aproximado (high/low não disponíveis)")
    
    # ===== RAZÃO HIGH/LOW (SÓ SE TIVER) =====
    if has_high and has_low:
        df['hl_ratio'] = df['high'] / df['low']
        print("✅ HL Ratio calculado")
    
    # ===== RAZÃO CLOSE/OPEN (SÓ SE TIVER) =====
    if has_open:
        df['co_ratio'] = df['close'] / df['open']
        print("✅ CO Ratio calculado")
    
    # ===== USD CLOSE (SÓ SE TIVER) =====
    if has_usd_close:
        df['close_usd_ratio'] = df['close'] / df['usd_close']
        print("✅ Close/USD Ratio calculado")
    
    # ===== SELIC (SÓ SE TIVER) =====
    if has_selic:
        df['selic_normalized'] = (df['selic'] - df['selic'].mean()) / df['selic'].std()
        print("✅ SELIC normalizado")
    
    # ===== MOMENTUM E PRICE RATE OF CHANGE =====
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['roc'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100
    
    # ===== BANDAS DE BOLLINGER =====
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ===== ICHIMOKU (simplificado) =====
    df['tenkan'] = (df['close'].rolling(window=9).max() + df['close'].rolling(window=9).min()) / 2
    df['kijun'] = (df['close'].rolling(window=26).max() + df['close'].rolling(window=26).min()) / 2
    
    print(f"✅ Features criadas: {df.columns.tolist()}")
    
    return df


# ========================================
# 3. LIMPEZA DO CLOSE PRICE
# ========================================

def clean_close_price(df):
    """Remove outliers mantendo alinhamento"""
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1
    
    mask = ~((df['close'] < (Q1 - 1.5 * IQR)) | 
             (df['close'] > (Q3 + 1.5 * IQR)))
    
    print(f"Linhas removidas por outlier: {(~mask).sum()}")
    return df[mask]


# ========================================
# 4. CACHE DE FEATURES
# ========================================

@st.cache_data(ttl=3600)
def load_features_cached():
    """Carrega e cria features com cache"""
    print("🔄 Carregando CSV...")
    df = load_csv_optimized()
    
    print("🧹 Limpando outliers...")
    df = clean_close_price(df)
    
    print("📊 Criando features...")
    df_feat = create_features(df).dropna()
    
    print(f"✅ Features finais: {len(df_feat)} linhas")
    return df_feat, df


# ========================================
# 5. CARREGAR MODELO
# ========================================

@st.cache_data(ttl=3600)
def load_model_and_info():
    """Carrega modelo e informações"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        return model, model_info, feature_columns
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None


# ========================================
# 6. PREVISÃO
# ========================================

def predict_next_day(df_feat_last, feature_columns, model):
    """Faz previsão usando features cacheados"""
    try:
        # Validar se todas as colunas existem
        missing_cols = [col for col in feature_columns if col not in df_feat_last.columns]
        if missing_cols:
            print(f"⚠️  Colunas faltando: {missing_cols}")
            print(f"Colunas disponíveis: {df_feat_last.columns.tolist()}")
            return None, None
        
        X_last = df_feat_last[feature_columns].iloc[-1:].values
        pred = model.predict(X_last)[0]
        proba = model.predict_proba(X_last)
        confidence = max(proba[0]) * 100
        
        return 'ALTA' if pred == 1 else 'BAIXA', confidence
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        print(f"Erro detalhado: {e}")
        return None, None


# ========================================
# 7. MAIN - STREAMLIT
# ========================================

st.set_page_config(page_title="IBOVESPA Dashboard", layout="wide")
st.title("📊 IBOVESPA Prediction Dashboard")

# Carregar dados
print("\n🚀 INICIANDO APP...")
try:
    df_feat, df = load_features_cached()
    model, model_info, feature_columns = load_model_and_info()
    
    if model is None:
        st.error("❌ Erro ao carregar modelo. Verifique os arquivos.")
    else:
        st.success("✅ Dados carregados com sucesso!")
        
        # Sidebar
        st.sidebar.header("⚙️ Filtros")
        dias_back = st.sidebar.slider("Dias para exibir:", 30, 250, 100)
        
        # Filtrar dados
        df_filtered = df.tail(dias_back)
        df_feat_filtered = df_feat.tail(dias_back)
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Preço Atual", f"R$ {df_filtered['close'].iloc[-1]:,.2f}")
        with col2:
            variacao = ((df_filtered['close'].iloc[-1] / df_filtered['close'].iloc[0]) - 1) * 100
            st.metric("Variação", f"{variacao:+.2f}%")
        with col3:
            pred, conf = predict_next_day(df_feat, feature_columns, model)
            if pred:
                st.metric("Previsão", pred, f"Confiança: {conf:.1f}%")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Série Histórica", "📊 Indicadores", "🎯 Performance"])
        
        with tab1:
            st.subheader("Evolução do Preço")
            sampling_rate = 5
            df_plot = df_filtered.iloc[::sampling_rate].copy()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot['date'],
                y=df_plot['close'],
                mode='lines',
                name='Close',
                hovertemplate='%{x|%d/%m/%Y}<br>R$ %{y:,.0f}<extra></extra>'
            ))
            fig.update_layout(hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Indicadores Técnicos")
            st.write("RSI, MACD, Volatilidade")
        
        with tab3:
            st.subheader("Performance do Modelo")
            st.write("Estatísticas de acurácia")

except Exception as e:
    st.error(f"❌ ERRO CRÍTICO: {e}")
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()
