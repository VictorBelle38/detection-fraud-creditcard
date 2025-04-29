import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e scaler
model = joblib.load('models/xgb_fraud_model.joblib')
scaler = joblib.load('models/scaler.joblib')

st.set_page_config(page_title="Detector de Fraudes", layout="wide")
st.title("💳 Detecção de Fraudes em Cartões de Crédito")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Faça upload do arquivo CSV com as transações", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # Verificar se o formato está correto
    if 'Class' in df_input.columns:
        df_input = df_input.drop('Class', axis=1)

    # Escalonar os dados
    df_scaled = scaler.transform(df_input)

    # Fazer a previsão
    predictions = model.predict(df_scaled)

    # Adicionar os resultados ao DataFrame original
    df_result = df_input.copy()
    df_result['Previsao'] = predictions
    df_result['Fraude'] = df_result['Previsao'].apply(lambda x: '⚠️ Sim' if x == 1 else '✅ Não')

    # Exibir métricas
    total = len(df_result)
    fraudes = df_result['Previsao'].sum()
    st.metric("Total de transações analisadas", total)
    st.metric("Fraudes detectadas", fraudes)

    # Mostrar a tabela com resultados
    st.subheader("Transações com Detecção de Fraude")
    st.dataframe(df_result[df_result['Previsao'] == 1])

    st.subheader("Todas as Transações")
    st.dataframe(df_result)

else:
    st.info("Por favor, envie um arquivo CSV com as transações a serem analisadas.")
