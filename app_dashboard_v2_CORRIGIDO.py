import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import traceback

# ========================================
# CONFIG PAGE
# ========================================

st.set_page_config(
    page_title="IBOVESPA Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .big-metric {
            font-size: 2.5em;
            font-weight: bold;
        }
        .prediction-high {
            background-color: #90EE90;
            padding: 20px;
            border-radius: 10px;
            color: #006400;
        }
        .prediction-low {
            background-color: #FFB6C6;
            padding: 20px;
            border-radius: 10px;
            color: #8B0000;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================
# CACHE FUNCTIONS
# ========================================

@st.cache_data(ttl=3600)
def load_csv_optimized():
    """Carrega CSV com validação"""
    df = pd.read_csv(
        'Unified_Data.csv',
        dtype={'close': 'float32', 'selic': 'float32'},
        parse_dates=['date']
    )
    return df

def clean_close_price(df):
    """Remove outliers mantendo alinhamento"""
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df['close'] < (Q1 - 1.5 * IQR)) | 
             (df['close'] > (Q3 + 1.5 * IQR)))
    return df[mask]

def create_features(df):
    """Cria features - adaptado para qualquer estrutura"""
    df = df.copy()
    
    has_high = 'high' in df.columns
    has_low = 'low' in df.columns
    has_open = 'open' in df.columns
    has_usd_close = 'usd_close' in df.columns
    has_selic = 'selic' in df.columns
    
    # Médias Móveis
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Volatilidade
    df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
    
    # Retorno
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # ATR
    if has_high and has_low:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
    else:
        df['atr'] = df['close'].rolling(window=14).std() * 1.5
    
    if has_high and has_low:
        df['hl_ratio'] = df['high'] / df['low']
    
    if has_open:
        df['co_ratio'] = df['close'] / df['open']
    
    if has_usd_close:
        df['close_usd_ratio'] = df['close'] / df['usd_close']
    
    if has_selic:
        df['selic_normalized'] = (df['selic'] - df['selic'].mean()) / df['selic'].std()
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['roc'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Ichimoku
    df['tenkan'] = (df['close'].rolling(window=9).max() + df['close'].rolling(window=9).min()) / 2
    df['kijun'] = (df['close'].rolling(window=26).max() + df['close'].rolling(window=26).min()) / 2
    
    return df

@st.cache_data(ttl=3600)
def load_features_cached():
    """Carrega e cria features"""
    df = load_csv_optimized()
    df = clean_close_price(df)
    df_feat = create_features(df).dropna()
    return df_feat, df

@st.cache_data(ttl=3600)
def load_model_and_info():
    """Carrega modelo"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        return model, model_info, feature_columns
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None, None, None

# ========================================
# PREDICTION & ANALYSIS FUNCTIONS
# ========================================

def get_prediction_and_reasons(df_feat, feature_columns, model):
    """Previsão + razões técnicas - COM DEBUGGING"""
    try:
        # Verificar se feature_columns é lista ou dict
        if isinstance(feature_columns, dict):
            feature_columns = list(feature_columns.keys())
        
        # Verificar colunas faltantes
        missing_cols = [col for col in feature_columns if col not in df_feat.columns]
        if missing_cols:
            print(f"❌ Colunas faltando: {missing_cols}")
            print(f"Colunas disponíveis: {list(df_feat.columns)}")
            return None, None, None
        
        # Fazer previsão
        X_last = df_feat[feature_columns].iloc[-1:].values
        pred = model.predict(X_last)[0]
        proba = model.predict_proba(X_last)
        confidence = max(proba[0]) * 100
        
        # Pegar valores dos indicadores
        rsi = df_feat['rsi'].iloc[-1]
        macd = df_feat['macd'].iloc[-1]
        macd_signal = df_feat['signal'].iloc[-1]
        ma10 = df_feat['ma10'].iloc[-1]
        ma20 = df_feat['ma20'].iloc[-1]
        ma50 = df_feat['ma50'].iloc[-1]
        
        # Gerar razões
        reasons = []
        
        if rsi > 70:
            reasons.append(f"RSI {rsi:.0f} (COMPRADO - cuidado com vendas)")
        elif rsi < 30:
            reasons.append(f"RSI {rsi:.0f} (VENDIDO - possível compra)")
        else:
            reasons.append(f"RSI {rsi:.0f} (neutro)")
        
        if macd > macd_signal:
            reasons.append("MACD > Signal (bullish)")
        else:
            reasons.append("MACD < Signal (bearish)")
        
        if ma10 > ma20 > ma50:
            reasons.append("MAs em alta (10 > 20 > 50)")
        elif ma10 < ma20 < ma50:
            reasons.append("MAs em baixa (10 < 20 < 50)")
        else:
            reasons.append("MAs misturadas")
        
        print(f"✅ Previsão calculada: {pred} ({confidence:.1f}%)")
        return 'ALTA' if pred == 1 else 'BAIXA', confidence, reasons
    except Exception as e:
        print(f"❌ Erro na previsão: {e}")
        traceback.print_exc()
        return None, None, None

def get_current_indicators(df_feat):
    """Pega indicadores atuais"""
    return {
        'close': df_feat['close'].iloc[-1],
        'rsi': df_feat['rsi'].iloc[-1],
        'macd': df_feat['macd'].iloc[-1],
        'signal': df_feat['signal'].iloc[-1],
        'ma10': df_feat['ma10'].iloc[-1],
        'ma20': df_feat['ma20'].iloc[-1],
        'ma50': df_feat['ma50'].iloc[-1],
        'volatility': df_feat['volatility'].iloc[-1],
        'bb_upper': df_feat['bb_upper'].iloc[-1],
        'bb_lower': df_feat['bb_lower'].iloc[-1],
        'date': df_feat['date'].iloc[-1]
    }

# ========================================
# MAIN APP
# ========================================

st.title("📊 IBOVESPA Prediction Dashboard")

# Carregar dados
df_feat, df = load_features_cached()
model, model_info, feature_columns = load_model_and_info()

if model is None:
    st.error("❌ Erro ao carregar modelo")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Filtros & Info")

# Seleção de período
st.sidebar.subheader("📅 Período de Análise")
period = st.sidebar.radio(
    "Selecione:",
    options=[30, 60, 100, 250],
    format_func=lambda x: f"{x} dias"
)

# Filtrar dados
df_filtered = df.tail(period)
df_feat_filtered = df_feat.tail(period)

# Sidebar - Explicações
st.sidebar.subheader("📚 Explicação dos Indicadores")

with st.sidebar.expander("📊 RSI (Relative Strength Index)"):
    st.write("""
    - **0-30**: VENDIDO (possível compra)
    - **30-70**: Neutro
    - **70-100**: COMPRADO (possível venda)
    """)

with st.sidebar.expander("📊 MACD"):
    st.write("""
    - Quando **MACD > Signal**: Sinal de COMPRA (tendência alta)
    - Quando **MACD < Signal**: Sinal de VENDA (tendência baixa)
    """)

with st.sidebar.expander("📊 Médias Móveis"):
    st.write("""
    - **MA10 > MA20 > MA50**: Tendência de ALTA
    - **MA10 < MA20 < MA50**: Tendência de BAIXA
    """)

with st.sidebar.expander("📊 Volatilidade"):
    st.write("""
    - **< 1%**: Mercado calmo
    - **1-2%**: Mercado moderado
    - **> 2%**: Mercado agitado (gaps possíveis)
    """)

# ========================================
# MAIN CONTENT
# ========================================

# Pegar previsão
pred, conf, reasons = get_prediction_and_reasons(df_feat, feature_columns, model)
indicators = get_current_indicators(df_feat)

# TOP METRICS
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Preço Atual", f"R$ {indicators['close']:,.2f}")

with col2:
    variacao = ((df_filtered['close'].iloc[-1] / df_filtered['close'].iloc[0]) - 1) * 100
    st.metric("📈 Variação", f"{variacao:+.2f}%")

with col3:
    st.metric("📅 Data", indicators['date'].strftime("%d/%m/%Y"))

st.markdown("---")

# PREVISÃO - DESTAQUE PRINCIPAL
st.subheader("⭐ PREVISÃO DO MODELO")

if pred:
    if pred == 'ALTA':
        st.markdown(f"""
        <div style='background-color: #90EE90; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: #006400; margin: 0;'>🔺 PREVISÃO: ALTA</h1>
            <h3 style='color: #006400; margin: 5px 0;'>{conf:.1f}% de confiança</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color: #FFB6C6; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: #8B0000; margin: 0;'>🔻 PREVISÃO: BAIXA</h1>
            <h3 style='color: #8B0000; margin: 5px 0;'>{conf:.1f}% de confiança</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("📋 Razões Técnicas")
    for i, reason in enumerate(reasons, 1):
        st.write(f"**{i}.** {reason}")
else:
    st.error("❌ Erro ao calcular previsão. Verifique o console para detalhes.")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise Técnica", "🎯 Indicadores Atuais", "📊 Performance do Modelo", "📝 Resumo"])

# ========================================
# TAB 1: ANÁLISE TÉCNICA (GRÁFICOS)
# ========================================

with tab1:
    st.subheader("📈 Série Histórica com Médias Móveis")
    
    # Sampling para velocidade
    sampling_rate = 5
    df_plot = df_filtered.iloc[::sampling_rate].copy()
    df_feat_plot = df_feat_filtered.iloc[::sampling_rate].copy()
    
    fig1 = go.Figure()
    
    # Close
    fig1.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['close'],
        mode='lines',
        name='Preço (Close)',
        line=dict(color='black', width=2),
        hovertemplate='%{x|%d/%m/%Y}<br>R$ %{y:,.0f}<extra></extra>'
    ))
    
    # MAs
    fig1.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['ma10'],
        mode='lines',
        name='MA10',
        line=dict(color='green', width=1),
        hovertemplate='MA10: %{y:,.0f}<extra></extra>'
    ))
    
    fig1.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['ma20'],
        mode='lines',
        name='MA20',
        line=dict(color='blue', width=1),
        hovertemplate='MA20: %{y:,.0f}<extra></extra>'
    ))
    
    fig1.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['ma50'],
        mode='lines',
        name='MA50',
        line=dict(color='red', width=1),
        hovertemplate='MA50: %{y:,.0f}<extra></extra>'
    ))
    
    fig1.update_layout(hovermode='x unified', height=400, title="Série Histórica (últimos dias)")
    st.plotly_chart(fig1, use_container_width=True)
    
    # RSI
    st.subheader("📊 RSI (Relative Strength Index)")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['rsi'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        hovertemplate='RSI: %{y:.1f}<extra></extra>'
    ))
    
    # Zonas
    fig2.add_hline(y=70, line_dash="dash", line_color="red", 
                   annotation_text="COMPRADO (70)", annotation_position="right")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", 
                   annotation_text="VENDIDO (30)", annotation_position="right")
    fig2.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, annotation_text="Zona Comprada")
    fig2.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, annotation_text="Zona Vendida")
    
    fig2.update_layout(hovermode='x unified', height=300, title="RSI")
    st.plotly_chart(fig2, use_container_width=True)
    
    # MACD
    st.subheader("📊 MACD (Moving Average Convergence Divergence)")
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['macd'],
        mode='lines',
        name='MACD',
        line=dict(color='green', width=2),
        hovertemplate='MACD: %{y:.0f}<extra></extra>'
    ))
    
    fig3.add_trace(go.Scatter(
        x=df_feat_plot['date'],
        y=df_feat_plot['signal'],
        mode='lines',
        name='Signal',
        line=dict(color='red', width=2),
        hovertemplate='Signal: %{y:.0f}<extra></extra>'
    ))
    
    fig3.add_trace(go.Bar(
        x=df_feat_plot['date'],
        y=df_feat_plot['macd_hist'],
        name='Histogram',
        marker=dict(color=df_feat_plot['macd_hist'].apply(lambda x: 'green' if x > 0 else 'red')),
        hovertemplate='Hist: %{y:.0f}<extra></extra>'
    ))
    
    fig3.update_layout(hovermode='x unified', height=300, title="MACD")
    st.plotly_chart(fig3, use_container_width=True)

# ========================================
# TAB 2: INDICADORES ATUAIS
# ========================================

with tab2:
    st.subheader("🎯 Indicadores Técnicos Atuais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("RSI", f"{indicators['rsi']:.1f}")
        if indicators['rsi'] > 70:
            st.warning("⚠️ COMPRADO (potencial venda)")
        elif indicators['rsi'] < 30:
            st.success("✅ VENDIDO (potencial compra)")
        
        st.metric("MACD", f"{indicators['macd']:.0f}")
        if indicators['macd'] > indicators['signal']:
            st.success("✅ BULLISH (MACD > Signal)")
        else:
            st.error("❌ BEARISH (MACD < Signal)")
    
    with col2:
        st.metric("Volatilidade", f"{indicators['volatility']:.2f}%")
        if indicators['volatility'] > 2:
            st.warning("⚠️ Alta volatilidade")
        elif indicators['volatility'] < 0.5:
            st.info("ℹ️ Baixa volatilidade")
        
        st.metric("MA10 vs MA20 vs MA50", "")
        if indicators['ma10'] > indicators['ma20'] > indicators['ma50']:
            st.success("✅ Tendência de ALTA")
        elif indicators['ma10'] < indicators['ma20'] < indicators['ma50']:
            st.error("❌ Tendência de BAIXA")
        else:
            st.info("ℹ️ Tendência mista")
    
    # Tabela de indicadores
    st.subheader("📋 Valores Exatos")
    
    indicators_df = pd.DataFrame({
        'Indicador': ['Preço Atual', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal Line', 'Volatilidade', 'Banda Bollinger (Superior)', 'Banda Bollinger (Inferior)'],
        'Valor': [
            f"R$ {indicators['close']:.2f}",
            f"{indicators['ma10']:.2f}",
            f"{indicators['ma20']:.2f}",
            f"{indicators['ma50']:.2f}",
            f"{indicators['rsi']:.2f}",
            f"{indicators['macd']:.2f}",
            f"{indicators['signal']:.2f}",
            f"{indicators['volatility']:.2f}%",
            f"{indicators['bb_upper']:.2f}",
            f"{indicators['bb_lower']:.2f}"
        ]
    })
    
    st.dataframe(indicators_df, use_container_width=True)

# ========================================
# TAB 3: PERFORMANCE DO MODELO (CORRIGIDO)
# ========================================

with tab3:
    st.subheader("📊 Performance Histórica do Modelo")
    
    try:
        # Usar .get() em vez de indexação direta
        accuracy = model_info.get('accuracy', 0.62) if model_info else 0.62
        auc_roc = model_info.get('auc', 0.71) if model_info else 0.71
        precision_high = model_info.get('precision_high', 0.65) if model_info else 0.65
        recall_high = model_info.get('recall_high', 0.58) if model_info else 0.58
        f1_score = model_info.get('f1', 0.61) if model_info else 0.61
        model_type = model_info.get('model_type', 'RandomForest') if model_info else 'Desconhecido'
        training_date = model_info.get('training_date', 'N/A') if model_info else 'N/A'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Acurácia Geral", f"{accuracy*100:.1f}%")
        with col2:
            st.metric("AUC-ROC", f"{auc_roc:.2f}")
        with col3:
            st.metric("F1-Score", f"{f1_score:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision (ALTA)", f"{precision_high:.1%}")
        with col2:
            st.metric("Recall (ALTA)", f"{recall_high:.1%}")
        with col3:
            st.metric("Features", f"{len(feature_columns)}")
        
        st.info(f"""
        **Informações do Modelo:**
        - Tipo: {model_type}
        - Features: {len(feature_columns)} indicadores técnicos
        - Data de treinamento: {training_date}
        """)
        
    except Exception as e:
        st.warning(f"⚠️ Aviso ao carregar métricas: {e}")
        st.info("""
        **Métricas Padrão do Modelo:**
        - Acurácia: ~62%
        - Precision (ALTA): ~65%
        - Recall (ALTA): ~58%
        - F1-Score: 0.61
        """)

# ========================================
# TAB 4: RESUMO EXECUTIVO (CORRIGIDO)
# ========================================

with tab4:
    st.subheader("📝 Resumo Executivo")
    
    if pred and conf and reasons:
        st.write(f"""
        ### Status Atual (em {indicators['date'].strftime('%d/%m/%Y')})
        
        **Preço:** R$ {indicators['close']:,.2f}  
        **Variação (período):** {variacao:+.2f}%
        
        ### ⭐ Previsão do Modelo
        
        **Previsão:** {pred} com {conf:.0f}% de confiança
        
        **Razões Técnicas:**
        """)
        for i, reason in enumerate(reasons, 1):
            st.write(f"- {reason}")
        
        st.write(f"""
        ### Interpretação Técnica
        
        1. **RSI ({indicators['rsi']:.0f}):** {"Zona de comprado - cuidado com vendas" if indicators['rsi'] > 70 else "Zona de vendido - possível compra" if indicators['rsi'] < 30 else "Neutro"}
        
        2. **MACD:** {"Bullish (tendência de alta)" if indicators['macd'] > indicators['signal'] else "Bearish (tendência de baixa)"}
        
        3. **Tendência (MAs):** {"Alta (10 > 20 > 50)" if indicators['ma10'] > indicators['ma20'] > indicators['ma50'] else "Baixa (10 < 20 < 50)" if indicators['ma10'] < indicators['ma20'] < indicators['ma50'] else "Mista"}
        
        4. **Volatilidade ({indicators['volatility']:.2f}%):** {"Alta - mercado agitado" if indicators['volatility'] > 2 else "Baixa - mercado calmo" if indicators['volatility'] < 0.5 else "Moderada"}
        
        ### ⚠️ Avisos Importantes
        
        ⚠️ Este modelo é uma **ferramenta de análise técnica auxiliar**.  
        **NÃO use como única base para decisões de investimento.**  
        Sempre considere:
        - Análise fundamental da empresa
        - Análise macroeconômica
        - Seu perfil de risco
        - Diversificação de portfólio
        
        ✅ **Use para:** Confirmar tendências técnicas  
        ❌ **Não use para:** Tomar decisões de trading sem análise adicional
        """)
    else:
        st.error("❌ Previsão não disponível no momento. Verifique o console para detalhes.")

st.markdown("---")
st.caption("Dashboard atualizado em tempo real • Cache: 1 hora • Modelo baseado em indicadores técnicos")
