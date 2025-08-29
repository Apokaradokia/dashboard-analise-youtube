#============================================================
# PROJETO DE ANÁLISE DE DADOS E DASHBOARD - TRAINEE SCITEC JR
# ALUNO: Leandro Pereira da Silva Filho
# Arquivo: app.py (Dashboard Interativo) - Versão Simplificada
#============================================================

# Importação das Bibliotecas 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import zipfile
import os

@st.cache_data
def carregar_e_preparar_dados():
    # Baixa e retorna o caminho da pasta do dataset
    path = kagglehub.dataset_download("cyberevil545/youtube-videos-data-for-ml-and-trend-analysis")
    
    # Monta o caminho completo do CSV (ajuste o nome exato se for diferente)
    csv_path = os.path.join(path, "youtube_data.csv")  
    
    df = pd.read_csv(csv_path)

    # Tratamento das colunas
    df.dropna(subset=['views', 'likes', 'category'], inplace=True)
    return df

# Execução do carregamento 
try:
    df = carregar_e_preparar_dados()
except Exception as e:
    st.error("Erro ao carregar os dados. Verifique sua chave API do Kaggle e a conexão.")
    st.exception(e) # Mostra o erro detalhado
    st.stop()

# Construção do Dashboard
# CORREÇÃO: "wide" em minúsculo
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

# GRÁFICO 2: Contagem de Vídeos por Categoria
st.subheader("📊 Contagem de Vídeos por Categoria")
# CORREÇÃO: Corrigido 'subplots' e 'countplot' e 'category'
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(y='category', data=df_filtrado, order=df_filtrado['category'].value_counts().index, ax=ax2, palette='plasma')
ax2.set_xlabel("Número de Vídeos")
ax2.set_ylabel("Categoria")
st.pyplot(fig2)