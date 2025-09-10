#============================================================
# PROJETO DE ANÁLISE DE DADOS E DASHBOARD - TRAINEE SCITEC JR
# ALUNO: Leandro Pereira da Silva Filho
# Arquivo: app.py (Dashboard Interativo) - Versão com ML implementado
#============================================================

# Importação das Bibliotecas 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import zipfile
import os
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@st.cache_data
def carregar_e_preparar_dados():
    # Baixa e retorna o caminho da pasta do dataset
    path = kagglehub.dataset_download("cyberevil545/youtube-videos-data-for-ml-and-trend-analysis")
    
    # Monta o caminho completo do CSV (ajuste o nome exato se for diferente)
    csv_path = os.path.join(path, "youtube_data.csv")
    df = pd.read_csv(csv_path)

    # Tratamento das colunas
    df.dropna(subset=['views', 'likes', 'category'], inplace=True)
    # Conversão em tipo númerico para realizar os cálculos
    df['views'] = pd.to_numeric(df["views"],errors='coerce')
    df['likes'] = pd.to_numeric(df["likes"], errors= 'coerce')
    df.dropna(subset=['views', 'likes'], inplace= True)
    return df

# Execução do carregamento 
try:
    df = carregar_e_preparar_dados()
except Exception as e:
    st.error("Erro ao carregar os dados. Verifique sua chave API do Kaggle e a conexão.")
    st.exception(e) # Mostra o erro detalhado
    st.stop()

# Construção do Dashboard

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Análise de Tendências do YouTube")

# Filtros na Barra Lateral
st.sidebar.header("Filtros")
# CORREÇÃO: Usado o nome de coluna correto 'category' (minúsculo)
categorias_unicas = sorted(df['category'].unique())
categoria_selecionada = st.sidebar.multiselect( 
    "Filtre por categoria:", 
    options=categorias_unicas,
    default=categorias_unicas[:5]
)

# Aplicação do filtro
if categoria_selecionada:
    df_filtrado = df[df['category'].isin(categoria_selecionada)]
else:
    df_filtrado = df.copy()

# Exibição das Métricas
st.header("Métricas Gerais (para a seleção atual)")
# CORREÇÃO: Sintaxe das colunas (vírgula)
col1, col2, col3 = st.columns(3)
col1.metric("Total de Vídeos", f"{len(df_filtrado):,}")
# CORREÇÃO: Nomes de colunas em minúsculo ('views', 'likes')
col2.metric("Total de Visualizações", f"{int(df_filtrado['views'].sum()):,}")
col3.metric("Total de Likes", f"{int(df_filtrado['likes'].sum()):,}")

st.markdown("---")

# Visualizações do Dashboard 
st.header("Análises Visuais Interativas")

# Gráfico 1: Top 10 Vídeos por visualizações (conforme solicitado)
st.subheader("🏆 Top 10 Vídeos por Visualizações")
# CORREÇÃO: Agrupado por 'title' e usando a coluna 'views'
top_videos = df_filtrado.groupby('title')['views'].sum().nlargest(10)
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_videos.values, y=top_videos.index, ax=ax1, palette='viridis')
ax1.set_xlabel("Total de Visualizações")
ax1.set_ylabel("Título do Vídeo")
st.pyplot(fig1)

# GRÁFICO 2: Contagem de Vídeos por Categori
st.subheader("📊 Contagem de Vídeos por Categoria")
# CORREÇÃO: Corrigido 'subplots' e 'countplot' e 'category'
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(y='category', data=df_filtrado, order=df_filtrado['category'].value_counts().index, ax=ax2, palette='plasma')
ax2.set_xlabel("Número de Vídeos")
ax2.set_ylabel("Categoria")
st.pyplot(fig2)

#==========================================================
#SEÇÃO ADICIONAL COM MACHINE LEARNING ( ML )
#==========================================================

st.markdown("---")
st.header("🤖 Prevendo o Sucesso de um vídeo ( Machine Learning)")

#Verifica se há dados suficientes para treinar o modelo
if len(df_filtrado) > 10:
    st.write(""" Nesta seção, iremos usar o modelo de **Regressão Linear** para prever quantos **Likes** um vídeo pode receber com base no número de ** Visualizações**. O modelo é treinado dinamicamente com os dados filtrados por você na barra lateral. """)
    # Preparação dos dados para o modelo
    X = df_filtrado[['views']] # Feature( Variável de entrada)
    Y = df_filtrado['likes'] # target ( o que queremos prever)

    # 2. Divisão dos dados em Treino e Teste
    #Usamos 70% para treinar e 30% para testar, como mais ou menos o exemplo em sala
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #3. Criação e modelo sendo treinado
    modelo_regressao = LinearRegression()
    modelo_regressao.fit( X_train, y_train)

    #4.  Verificação do modelo
    previsoes = modelo_regressao.predict(X_test)
    mse = mean_squared_error(y_test, previsoes)
    rmse = np.sqrt(mse)
    st.subheader("Avaliação do Modelo de Regressão")
    col_aval1, col_aval2 = st.columns(2)
    col_aval1.metric("Erro Quadrático Médio(RMSE)", f"{rmse:,.2f}")
    col_aval2.write("O RMSE indica, em média, qual a margem de erro da previsão do modelo(em n de likes). QUanto menor, melhor.")

    #5. Implementação: Interface para Previsão
    st.subheader("faça uma Previsão!")

    #Campo para o usuário inserir o número de visualizações
    input_views = st.number_input('Digite o número de visualizações para prever os likes:', min_value=0,
         value=1000000,#Valor padrão)
         step=10000
    )
    if st.button("Prever Likes"):
        #Prepara o input do usuário para o modelo
        #(Precisa ser um array 2D, por isso os colchetes duplos)
        input_data = np.array([[input_views]])

        #Previsão
        previsao_likes = modelo_regressao.predict(input_data)

        #Exibi o resultado
        st.success(f" Previsão: Um vídeo com **{input_views:,}** visualizações teria aproximadamente **{int(previsao_likes[0]):,}** likes.")
    else:
        st.warning("Selecione mais dados ou uma categoria com mais vídeos para ativar a funcionalidade  com base em Machine learning ( ML ).")