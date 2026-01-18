import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════════════════
# 🔧 FUNÇÕES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    """Carrega dados do CSV com tratamento de erros"""
    try:
        df = pd.read_csv('Unified_Data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ CSV carregado: {len(df)} linhas")
        print(f"Colunas originais: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        traceback.print_exc()
        return None

def create_features(df):
    """Cria features técnicas a partir do close"""
    try:
        df = df.copy()
        
        # ✅ FEATURES QUE SEMPRE FUNCIONAM (sem dependências externas)
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # RSI (14 períodos)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Retorno (label para classificação)
        df['return'] = df['close'].pct_change() * 100
        df['target'] = (df['return'].shift(-1) > 0).astype(int)  # 1=ALTA, 0=BAIXA
        
        # Volatilidade
        df['volatility'] = df['return'].rolling(window=20, min_periods=1).std()
        
        # Volume (se existir, se não, cria dummy)
        if 'volume' not in df.columns:
            df['volume'] = 1000 + np.random.randint(0, 500, len(df))
        
        # Banda de Bollinger
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['bb_std'] = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Preencher NaN
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✅ Features criadas: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Erro ao criar features: {e}")
        traceback.print_exc()
        return None

def train_model(df):
    """Treina modelo de classificação"""
    try:
        # Features para treino
        feature_cols = ['sma_5', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 
                       'volatility', 'bb_upper', 'bb_lower']
        
        X = df[feature_cols].fillna(0)
        y = df['target'].fillna(0)
        
        # Remover últimas 5 linhas para teste (sem NaN target)
        train_size = len(df) - 5
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # Padronizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Treinar
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        print(f"✅ Modelo treinado com {len(X_train)} amostras")
        
        return model, scaler, feature_cols
        
    except Exception as e:
        print(f"❌ Erro ao treinar modelo: {e}")
        traceback.print_exc()
        return None, None, None

def get_prediction_and_reasons(df, model, scaler, feature_cols):
    """Obtém previsão e motivos técnicos para o próximo dia"""
    try:
        if df is None or model is None or scaler is None:
            print("❌ Dados ou modelo ausentes")
            return None, None, None
        
        # Última linha (hoje)
        last_row = df.iloc[-1]
        
        # Preparar features
        X_latest = df[feature_cols].iloc[-1:].fillna(0)
        
        print(f"✅ Features da última linha: {X_latest.values}")
        
        # Prever
        X_scaled = scaler.transform(X_latest)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        confidence = max(probability) * 100
        
        # Razões técnicas
        reasons = []
        
        rsi = last_row['rsi']
        if rsi < 30:
            reasons.append("RSI abaixo de 30 (sobrevenda)")
        elif rsi > 70:
            reasons.append("RSI acima de 70 (sobrecompra)")
        
        macd = last_row['macd']
        macd_signal = last_row['macd_signal']
        if macd > macd_signal:
            reasons.append("MACD acima da linha de sinal (momentum positivo)")
        else:
            reasons.append("MACD abaixo da linha de sinal (momentum negativo)")
        
        close = last_row['close']
        bb_upper = last_row['bb_upper']
        bb_lower = last_row['bb_lower']
        
        if close > bb_upper:
            reasons.append("Preço acima da banda de Bollinger superior (resistência)")
        elif close < bb_lower:
            reasons.append("Preço abaixo da banda de Bollinger inferior (suporte)")
        
        print(f"✅ Previsão calculada: {prediction} ({confidence:.1f}%)")
        print(f"   Razões: {reasons}")
        
        return prediction, confidence, reasons
        
    except Exception as e:
        print(f"❌ Erro ao calcular previsão: {e}")
        traceback.print_exc()
        return None, None, None

# ═════════════════════════════════════════════════════════════════════════════
# 📊 STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Dashboard Previsão", layout="wide")

# Carregar dados
df = load_data()

if df is None:
    st.error("❌ Erro ao carregar dados. Verifique o arquivo CSV.")
    st.stop()

# Criar features
df = create_features(df)

if df is None:
    st.error("❌ Erro ao criar features.")
    st.stop()

# Treinar modelo
model, scaler, feature_cols = train_model(df)

if model is None:
    st.error("❌ Erro ao treinar modelo.")
    st.stop()

# Obter previsão
pred, conf, reasons = get_prediction_and_reasons(df, model, scaler, feature_cols)

# ═════════════════════════════════════════════════════════════════════════════
# 📈 LAYOUT PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

st.title("📊 Dashboard de Previsão de Mercado")

# Card de previsão (TOPO)
if pred is not None and conf is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if pred == 1:
            st.success(f"🟢 PREVISÃO: ALTA (Confiança: {conf:.1f}%)")
        else:
            st.error(f"🔴 PREVISÃO: BAIXA (Confiança: {conf:.1f}%)")
    
    with col2:
        st.metric("Última Cotação", f"R$ {df.iloc[-1]['close']:,.0f}")
else:
    st.error("❌ Erro ao calcular previsão. Verifique o console para detalhes.")

st.divider()

# Abas
tab1, tab2, tab3, tab4 = st.tabs(["Análise Técnica", "Indicadores Atuais", "Performance", "Resumo"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: ANÁLISE TÉCNICA
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📈 Série Histórica com SMA")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        name='Close',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_5'],
        name='SMA 5',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_20'],
        name='SMA 20',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(height=400, title="Preço e Médias Móveis")
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI
    st.subheader("📊 RSI (14)")
    
    fig_rsi = go.Figure()
    
    fig_rsi.add_trace(go.Scatter(
        x=df['date'],
        y=df['rsi'],
        name='RSI',
        line=dict(color='purple', width=2),
        fill='tozeroy'
    ))
    
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Sobrevenda (30)")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Sobrecompra (70)")
    
    fig_rsi.update_layout(height=300, title="Relative Strength Index")
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD
    st.subheader("📉 MACD")
    
    fig_macd = go.Figure()
    
    fig_macd.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    fig_macd.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd_signal'],
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    fig_macd.update_layout(height=300, title="MACD e Linha de Sinal")
    st.plotly_chart(fig_macd, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: INDICADORES ATUAIS
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("📊 Indicadores Atuais")
    
    last = df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Close", f"R$ {last['close']:,.0f}")
        st.metric("SMA 5", f"R$ {last['sma_5']:,.0f}")
        st.metric("SMA 20", f"R$ {last['sma_20']:,.0f}")
    
    with col2:
        st.metric("RSI", f"{last['rsi']:.2f}")
        st.metric("MACD", f"{last['macd']:.4f}")
        st.metric("Volatilidade", f"{last['volatility']:.4f}")
    
    with col3:
        st.metric("BB Upper", f"R$ {last['bb_upper']:,.0f}")
        st.metric("BB Lower", f"R$ {last['bb_lower']:,.0f}")
        st.metric("Retorno (%)", f"{last['return']:.2f}%")
    
    # Tabela completa
    st.subheader("📋 Últimos 10 Dias")
    
    display_cols = ['date', 'close', 'sma_5', 'rsi', 'macd', 'volatility', 'return']
    display_df = df[display_cols].tail(10).copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: PERFORMANCE DO MODELO
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🤖 Performance do Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tipo de Modelo", "Random Forest")
        st.metric("Árvores", "100")
    
    with col2:
        st.metric("Features", len(feature_cols))
        st.metric("Amostras Treino", len(df) - 5)
    
    with col3:
        st.metric("Data Treino", df.iloc[-1]['date'].strftime('%Y-%m-%d'))
        st.metric("Status", "✅ OK")
    
    st.subheader("📊 Importância das Features")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_feat = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Importância das Features no Modelo'
    )
    
    st.plotly_chart(fig_feat, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: RESUMO EXECUTIVO
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📋 Resumo Executivo")
    
    if pred is not None and conf is not None and reasons:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if pred == 1:
                st.info(f"🟢 **PREVISÃO: ALTA**")
                st.write(f"A análise técnica aponta movimento de ALTA para o próximo dia.")
            else:
                st.warning(f"🔴 **PREVISÃO: BAIXA**")
                st.write(f"A análise técnica aponta movimento de BAIXA para o próximo dia.")
        
        with col2:
            st.metric("Confiança", f"{conf:.1f}%")
        
        st.subheader("🔍 Razões Técnicas")
        
        for i, reason in enumerate(reasons, 1):
            st.write(f"{i}. {reason}")
        
        st.subheader("📊 Dados da Última Linha")
        
        last_data = {
            'Métrica': ['Data', 'Close', 'RSI', 'MACD', 'Volatilidade', 'Retorno (%)'],
            'Valor': [
                df.iloc[-1]['date'].strftime('%Y-%m-%d'),
                f"R$ {df.iloc[-1]['close']:,.0f}",
                f"{df.iloc[-1]['rsi']:.2f}",
                f"{df.iloc[-1]['macd']:.4f}",
                f"{df.iloc[-1]['volatility']:.4f}",
                f"{df.iloc[-1]['return']:.2f}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(last_data), use_container_width=True)
    
    else:
        st.error("❌ Erro ao calcular previsão. Verifique o console para detalhes.")
        st.write("Possíveis causas:")
        st.write("1. Features não foram criadas corretamente")
        st.write("2. Modelo não foi treinado com sucesso")
        st.write("3. Dados insuficientes para previsão")

st.divider()

# Footer
st.caption("🚀 Dashboard de Previsão v1")
