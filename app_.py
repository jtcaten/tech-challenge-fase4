import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title='IBOVESPA Prediction Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS customizado
st.markdown("""<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-size: 3rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 IBOVESPA Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('---')

# Carregar modelo e dados
@st.cache_resource
def load_model_and_data():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model_info.json', 'r') as f:
        model_info = json.load(f)

    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)['feature_columns']

    df = pd.read_csv('Unified_Data.csv')
    df['date'] = pd.to_datetime(df['date'])

    return model, model_info, feature_columns, df

try:
    best_model, model_info, feature_columns, df = load_model_and_data()
    df = df.sort_values('date').reset_index(drop=True)
except FileNotFoundError:
    st.error('❌ Arquivos necessários não encontrados: best_model.pkl, model_info.json, feature_columns.json, Unified_Data.csv')
    st.stop()

# Função de limpeza
def clean_close_price(close_price):
    close_array = close_price.copy()
    outlier_mask = close_array < 10000

    for idx in np.where(outlier_mask)[0]:
        valid_before = close_array[:idx][close_array[:idx] >= 10000]
        valid_after = close_array[idx+1:][close_array[idx+1:] >= 10000]

        if len(valid_before) > 0 and len(valid_after) > 0:
            close_array[idx] = (valid_before.iloc[-1] + valid_after.iloc[0]) / 2
        elif len(valid_before) > 0:
            close_array[idx] = valid_before.iloc[-1]
        elif len(valid_after) > 0:
            close_array[idx] = valid_after.iloc[0]

    return close_array

# Função de features
def create_features(df_temp):
    df_feat = df_temp.copy()
    df_feat['returns'] = df_feat['close'].pct_change()

    for lag in range(1, 6):
        df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
    df_feat['close_lag_10'] = df_feat['close'].shift(10)

    df_feat['sinal_t1'] = (df_feat['close'] > df_feat['close'].shift(1)).astype(int)
    df_feat['sinal_t2'] = (df_feat['close'] > df_feat['close'].shift(2)).astype(int)
    df_feat['sinal_t3'] = (df_feat['close'] > df_feat['close'].shift(3)).astype(int)
    df_feat['sinal_t5'] = (df_feat['close'] > df_feat['close'].shift(5)).astype(int)
    df_feat['sinal_t10'] = (df_feat['close'] > df_feat['close'].shift(10)).astype(int)

    for lag in range(1, 11):
        df_feat[f'returns_lag_{lag}'] = df_feat['returns'].shift(lag)

    df_feat['ma5'] = df_feat['close'].rolling(5).mean()
    df_feat['ma20'] = df_feat['close'].rolling(20).mean()
    df_feat['ma50'] = df_feat['close'].rolling(50).mean()

    df_feat['sinal_ma5_ma20'] = (df_feat['ma5'] > df_feat['ma20']).astype(int)
    df_feat['close_acima_ma5'] = (df_feat['close'] > df_feat['ma5']).astype(int)
    df_feat['close_acima_ma20'] = (df_feat['close'] > df_feat['ma20']).astype(int)

    df_feat['volatility'] = df_feat['returns'].rolling(20).std()

    delta = df_feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))

    df_feat['sinal_usd_up'] = (df_feat['usd_close'] > df_feat['usd_close'].shift(1)).astype(int)
    df_feat['usd_change'] = df_feat['usd_close'].pct_change()

    df_feat['selic_subindo'] = (df_feat['selic'] > df_feat['selic'].shift(1)).astype(int)
    df_feat['selic_change'] = df_feat['selic'].diff()

    return df_feat

# Função de previsão
def predict_next_day(df_last):
    df_feat = create_features(df_last.tail(100))

    if len(df_feat) < len(feature_columns):
        return None, None

    X_last = df_feat[feature_columns].iloc[-1:]

    if X_last.isnull().any().any():
        X_last = X_last.fillna(0)

    pred = best_model.predict(X_last)[0]
    proba = best_model.predict_proba(X_last)[0]

    prediction = 'ALTA' if pred == 1 else 'BAIXA'
    confidence = max(proba) * 100

    return prediction, confidence

# Limpar dados
df['close'] = clean_close_price(df['close'])
df_feat = create_features(df).dropna()

# Fazer previsão
pred, conf = predict_next_day(df)

# ==== SIDEBAR ====
with st.sidebar:
    st.header('⚙️ Configurações')

    # Data range slider
    date_range = st.slider(
        'Período para análise',
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date(),
        value=(df['date'].max().date() - pd.Timedelta(days=365), df['date'].max().date()),
        format='YYYY-MM-DD'
    )

    df_filtered = df[(df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))]

    st.divider()

    st.subheader('📊 Modelo Info')
    st.text(f"Modelo: {model_info['model_name']}")
    st.text(f"Accuracy: {model_info['accuracy']:.2%}")
    st.text(f"Features: {model_info['feature_count']}")

# ==== MAIN CONTENT ====
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        'Última Cotação',
        f'{df["close"].iloc[-1]:,.0f}',
        f"{(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100:.2f}%"
    )

with col2:
    st.metric('Previsão', f'{pred}', f'{conf:.1f}% confiança')

with col3:
    st.metric('Data', df['date'].iloc[-1].strftime('%d/%m/%Y'))

st.divider()

# Tabs para diferentes visualizações
tab1, tab2, tab3, tab4 = st.tabs(['📈 Série Histórica', '📊 Indicadores', '🎯 Performance', '📋 Dados'])

with tab1:
    st.subheader('IBOVESPA com Médias Móveis')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['close'],
        name='IBOVESPA',
        line=dict(color='#38bdf8', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_feat['date'],
        y=df_feat['ma5'],
        name='MA5',
        line=dict(color='#fbbf24', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df_feat['date'],
        y=df_feat['ma20'],
        name='MA20',
        line=dict(color='#f87171')
    ))

    fig.add_trace(go.Scatter(
        x=df_feat['date'],
        y=df_feat['ma50'],
        name='MA50',
        line=dict(color='#10b981')
    ))

    fig.update_layout(
        title='IBOVESPA - Série Histórica com Médias Móveis',
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('Indicadores Técnicos - Últimos 100 dias')

    df_recent = df_feat.tail(100)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Preço', 'RSI', 'Volatilidade'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['close'], name='IBOV', line=dict(color='#38bdf8')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['rsi'], name='RSI', line=dict(color='#fbbf24')), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    fig.add_trace(go.Bar(x=df_recent['date'], y=df_recent['volatility'], name='Volatility', marker_color='#f87171'), row=3, col=1)

    fig.update_layout(title='Indicadores Técnicos', template='plotly_dark', height=800, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader('Performance do Modelo')

    col1, col2 = st.columns(2)

    with col1:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        values = [
            model_info['accuracy'],
            model_info['precision'],
            model_info['recall'],
            model_info['f1'],
            model_info['roc_auc']
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=['#38bdf8', '#fbbf24', '#10b981', '#f87171', '#a78bfa'],
            text=[f'{v:.1%}' for v in values],
            textposition='auto'
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray')
        fig.update_layout(title='Métricas do Modelo', template='plotly_dark', height=400, showlegend=False)
        fig.update_yaxes(range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write('**Modelos Comparados:**')
        for model_name, metrics in model_info['all_models'].items():
            st.write(f"
**{model_name}**")
            st.write(f"Accuracy: {metrics['accuracy']:.2%}")
            st.write(f"F1-Score: {metrics['f1']:.2%}")

with tab4:
    st.subheader('Dados Históricos')

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total de dados:** {len(df)} dias")
        st.write(f"**Período:** {df['date'].min().date()} até {df['date'].max().date()}")
        st.write(f"**Preço mínimo:** {df['close'].min():,.0f}")
        st.write(f"**Preço máximo:** {df['close'].max():,.0f}")

    with col2:
        st.write(f"**Modelos treinados:** 3")
        st.write(f"**Melhor modelo:** {model_info['model_name']}")
        st.write(f"**Features utilizadas:** {model_info['feature_count']}")
        st.write(f"**Data treino:** {model_info['training_date']}")

    st.write('
**Últimas 10 linhas:**')
    st.dataframe(df[['date', 'close', 'usd_close', 'selic']].tail(10), use_container_width=True)
