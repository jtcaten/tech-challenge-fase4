# 📈 Tech Challenge – Fase 4 | Previsão do IBOVESPA

Este projeto corresponde à **Fase 4 do Tech Challenge (FIAP / POSTECH)** e tem
como objetivo realizar o **deploy de um modelo de séries temporais**
desenvolvido na Fase 2, disponibilizando uma aplicação interativa utilizando
**Streamlit**.

---

## 🎯 Objetivo

Disponibilizar um modelo preditivo do **IBOVESPA**, permitindo que o usuário:

- Visualize dados históricos do índice
- Escolha o horizonte de previsão
- Acompanhe métricas de desempenho do modelo
- Interaja com previsões de forma simples e visual

---

## 🧠 Modelo Utilizado

- **Modelo:** ARIMA(1,0,0)
- **Variável modelada:** Retorno logarítmico do IBOVESPA
- **Validação:** Walk-forward
- **Avaliação:** Previsão da direção do mercado (alta ou baixa)
- **Deploy:** Conversão do retorno previsto para nível de preço

As métricas apresentadas no dashboard foram obtidas durante a validação
realizada na **Fase 2 do Tech Challenge**.

---

## 🖥️ Aplicação Streamlit

A aplicação desenvolvida com Streamlit oferece:

- Gráfico com histórico do IBOVESPA
- Previsão futura baseada no modelo treinado
- Painel de métricas do modelo (Acurácia, Precisão, Recall e F1-Score)
- Registro das interações do usuário para simular monitoramento do modelo

---

## 📁 Estrutura do Projeto

```
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── modelo_ibov.pkl
│
├── data/
│   ├── Dados Históricos - Ibovespa 2005-2025.csv
│   └── logs_previsoes.csv
│
└── notebook/
    └── Tech_challenge_fase_2_grupo_8.ipynb
```

---

## 🚀 Como Executar Localmente

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute a aplicação:
   ```bash
   streamlit run app__.py
   ```

---

## 🌐 Deploy

O deploy da aplicação foi realizado utilizando o **Streamlit Cloud**, com
integração direta ao repositório do GitHub.

---

## 📹 Vídeo Demonstrativo

Foi produzido um vídeo de até **5 minutos**, apresentando:

- O contexto do problema
- O modelo desenvolvido na Fase 2
- A aplicação Streamlit em funcionamento
- O painel de métricas e monitoramento

---

## 👨‍🎓 Projeto Acadêmico

Projeto desenvolvido para fins acadêmicos no curso **POSTECH – FIAP**,
como parte do **Tech Challenge – Fase 4**.
