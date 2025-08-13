import streamlit as st
import os
import pandas as pd 
import plotly.express as px 
import numpy as np 
import datetime
from wordcloud import WordCloud
from sklearn.cluster import KMeans

st.set_page_config(page_title="Painel de Tendências", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

def ajeita_data(t):
    try:
        return pd.to_datetime(t)
    except:
        return np.nan

def len_texto(x):
    return len(x.split())

@st.cache_data
def carregar_dados_brasil():
    df = pd.read_csv('data/TSNE_BR_COM_SENTIMENTO.csv', sep=';')
    df['DT_PUBLICACAO'] = df['DT_PUBLICACAO'].apply(ajeita_data)
    df.dropna(subset=['DT_PUBLICACAO'], inplace=True)
    df.sort_values('DT_PUBLICACAO', inplace=True)
    df.rename(columns={'N_CURTIDAS_NEW': 'CURTIDAS', 'N_COMENTARIOS_NEW': "COMENTARIOS"}, inplace=True)
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df['LEN_TEXTO'] = df['TEXTO'].apply(len_texto)
    return df

@st.cache_data
def carregar_dados_suecia():
    df = pd.read_csv('data/TSNE_SE_COM_SENTIMENTO.csv', sep=';')
    df['DT_PUBLICACAO'] = df['DT_PUBLICACAO'].apply(ajeita_data)
    df.dropna(subset=['DT_PUBLICACAO'], inplace=True)
    df.sort_values('DT_PUBLICACAO', inplace=True)
    df.rename(columns={'N_CURTIDAS_NEW': 'CURTIDAS', 'N_COMENTARIOS_NEW': "COMENTARIOS"}, inplace=True)
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df['LEN_TEXTO'] = df['TEXTO'].apply(len_texto)
    return df

@st.cache_data
def carregar_stop_words_br():
    stop_df = pd.read_csv('data/STOP_WORDS_PORTUGUES.csv', sep=';')
    return list(stop_df.STOP_WORDS.values)

@st.cache_data
def carregar_dados_reclame_aqui():
    # Carrega os dados pulando a primeira linha se ela for cabeçalho
    df = pd.read_excel('data/reclameaqui.xlsx', sheet_name='Página1', header=None)
    
    # Verifica se a primeira linha é cabeçalho (se contém 'MARCA' ou similar)
    if 'MARCA' in str(df.iloc[0, 0]).upper():
        # Se for cabeçalho, usamos a primeira linha como nomes de colunas
        df.columns = df.iloc[0]
        df = df[1:]
    else:
        # Se não for cabeçalho, definimos os nomes das colunas manualmente
        df.columns = ['MARCA', 'CATEGORIA', 'ITEM1', 'PORCENTAGEM1', 'QUANTIDADE1', 
                     'ITEM2', 'QUANTIDADE2', 'ITEM3', 'QUANTIDADE3', 
                     'ITEM4', 'QUANTIDADE4', 'ITEM5', 'QUANTIDADE5']
    
    # Remove linhas vazias
    df = df.dropna(how='all')
    
    # Converte porcentagens para string se não forem
    if 'PORCENTAGEM1' in df.columns:
        df['PORCENTAGEM1'] = df['PORCENTAGEM1'].astype(str)
    
    return df

brasil = carregar_dados_brasil()
stop_words = carregar_stop_words_br()

st.markdown('<h1 style="text-align:center; color:#000000;">Painel de Tendências</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:grey;">Explore as tendências e métricas de montadoras automobilísticas</p>', unsafe_allow_html=True)

# --- Barra Lateral para Filtros ---
with st.sidebar:
    st.header("Filtros de Análise")

    todas_marcas = sorted(list(brasil["MARCA"].unique()))
    marcas_selecionadas = st.multiselect(
        "Comparar Marcas", 
        todas_marcas, 
        default=[], 
        placeholder="+ Adicionar marcas"
    )

    if not marcas_selecionadas:
        st.warning("Selecione pelo menos uma marca para iniciar a análise.")
        st.stop()

    paises = st.multiselect(
        "País", 
        ["BR", "SE"], 
        placeholder="Selecionar"
    )
    
    anos_disponiveis = [2020, 2021, 2022, 2023, 2024]
    periodo_selecionado = st.multiselect(
        "Período", 
        anos_disponiveis, 
        placeholder="Selecionar"
    )

    agrupamento = st.radio(
        "Agrupar por", 
        options=["mês", "ano"], 
        index=0, 
        horizontal=True,
        help="Escolha a granularidade do agrupamento temporal."
    )

    if not paises or not periodo_selecionado:
        st.error("Por favor, selecione pelo menos um país e um período.")
        st.stop()

dados_lista = []
if "BR" in paises:
    df_br = carregar_dados_brasil()
    df_br["PAIS"] = "BR"
    dados_lista.append(df_br)
if "SE" in paises:
    df_se = carregar_dados_suecia()
    df_se["PAIS"] = "SE"
    dados_lista.append(df_se)

dados = pd.concat(dados_lista, ignore_index=True)

dados["ANO"] = dados["DT_PUBLICACAO"].dt.year
dados = dados[dados["ANO"].isin(periodo_selecionado)]
dados = dados[dados["MARCA"].isin(marcas_selecionadas)]
dados["MARCA_PAIS"] = dados["MARCA"] + " - " + dados["PAIS"]

if agrupamento == "Anual":
    dados['PERIODO'] = dados['DT_PUBLICACAO'].dt.to_period("Y").dt.to_timestamp()
else:
    dados['PERIODO'] = dados['DT_PUBLICACAO'].dt.to_period("M").dt.to_timestamp()

# --- Métricas Chave (Cards) ---
st.markdown("---")
st.subheader("Métricas Chave")
col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

total_publicacoes = dados.shape[0]
total_curtidas = dados['CURTIDAS'].sum()
total_comentarios = dados['COMENTARIOS'].sum()

with col_metrics1:
    st.metric(label="Total de Publicações", value=f"{total_publicacoes:,}".replace(",", "."))
with col_metrics2:
    st.metric(label="Total de Curtidas", value=f"{total_curtidas:,}".replace(",", "."))
with col_metrics3:
    st.metric(label="Total de Comentários", value=f"{total_comentarios:,}".replace(",", "."))

st.markdown("---")

# --- Seção de Métricas ao Longo do Tempo ---
with st.expander("Métricas ao Longo do Tempo (Instagram)", expanded=True):
    st.subheader(f"Evolução de Publicações, Curtidas e Comentários por {agrupamento}")

    df_pub = dados.groupby(['PERIODO', 'MARCA_PAIS']).size().reset_index(name='PUBLICAÇÕES')
    fig_pub = px.line(df_pub, 
                     x='PERIODO', 
                     y='PUBLICAÇÕES', 
                     color='MARCA_PAIS', 
                     title=f'Publicações por {agrupamento.lower()}',
                     markers=True)
    fig_pub.update_layout(
        xaxis_title='Período', 
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified'
    )
    fig_pub.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Publicações: %{y:,}'
    )
    st.plotly_chart(fig_pub, use_container_width=True)

    df_curtidas = dados.groupby(['PERIODO', 'MARCA_PAIS'])['CURTIDAS'].sum().reset_index()
    fig_curtidas = px.bar(df_curtidas, 
                         x='PERIODO', 
                         y='CURTIDAS', 
                         color='MARCA_PAIS',
                         barmode='group',
                         title=f'Curtidas por {agrupamento.lower()}',
                         text_auto=False)
    fig_curtidas.update_layout(
        xaxis_title='Período', 
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified'
    )
    fig_curtidas.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Curtidas: %{y:,}'
    )
    st.plotly_chart(fig_curtidas, use_container_width=True)

    df_comentarios = dados.groupby(['PERIODO', 'MARCA_PAIS'])['COMENTARIOS'].sum().reset_index()
    fig_comentarios = px.bar(df_comentarios, 
                            x='PERIODO', 
                            y='COMENTARIOS', 
                            color='MARCA_PAIS',
                            barmode='group',
                            title=f'Comentários por {agrupamento.lower()}',
                            text_auto=False)
    fig_comentarios.update_layout(
        xaxis_title='Período', 
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified'
    )
    fig_comentarios.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Comentários: %{y:,}'
    )
    st.plotly_chart(fig_comentarios, use_container_width=True)

# --- Seção de Análise de Sentimentos ---
with st.expander("Análise de Sentimentos (Instagram)", expanded=True): 
    if 'SENTIMENTO' in dados.columns:
        st.subheader("Distribuição Geral dos Sentimentos")
        dist_sent = dados['SENTIMENTO'].value_counts().reset_index()
        dist_sent.columns = ['Sentimento', 'Quantidade']
        fig1 = px.bar(dist_sent, 
                     x='Sentimento', 
                     y='Quantidade', 
                     color='Sentimento',
                     title="Frequência dos sentimentos",
                     category_orders={"Sentimento": ["alegria", "tristeza", "raiva", "surpresa", "medo"]})
        # Gráfico de percentual por sentimento
        st.subheader("Percentual de Sentimentos em Relação ao Total de Publicações")

        # Calcula percentuais
        dist_sent['Percentual'] = (dist_sent['Quantidade'] / dist_sent['Quantidade'].sum()) * 100
        dist_sent['Percentual'] = dist_sent['Percentual'].round(2)

        # Gráfico de pizza
        fig_pizza = px.pie(dist_sent, 
                        names='Sentimento', 
                        values='Percentual',
                        title='Percentual de Publicações por Sentimento',
                        hole=0.4)

        fig_pizza.update_traces(textinfo='percent+label')

        st.plotly_chart(fig_pizza, use_container_width=True)

        fig1.update_traces(
            hovertemplate='<br>Quantidade: %{y:,}'
        )
        fig1.update_layout(
            xaxis_title='Sentimento', 
            yaxis_title='Quantidade',
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Distribuição de Sentimentos por Marca")
        df_marca_sent = dados.groupby(['MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='Quantidade')

        fig2 = px.bar(df_marca_sent, 
                     x='MARCA_PAIS', 
                     y='Quantidade', 
                     color='SENTIMENTO', 
                     barmode='group',
                     title="Distribuição de sentimentos por Marca e País")
        fig2.update_traces(
            hovertemplate='<br>Marca: %{x}<br>Quantidade: %{y:,}'
        )
        fig2.update_layout(
            xaxis_title='Marca - País', 
            yaxis_title='Quantidade', 
            xaxis_tickangle=-45,
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Evolução Temporal dos Sentimentos por Marca")
        
        filtro_col1, filtro_col2 = st.columns(2)
        
        with filtro_col1:
            sentimentos_disponiveis = dados['SENTIMENTO'].unique()
            sentimentos_selecionados = st.multiselect(
                "Selecione os sentimentos:",
                options=sentimentos_disponiveis,
                default=["alegria"] if "alegria" in sentimentos_disponiveis else list(sentimentos_disponiveis),
                key="filtro_sentimentos"
            )
        
        with filtro_col2:
            paises_disponiveis = dados['PAIS'].unique()
            paises_selecionados = st.multiselect(
                "Selecione os países:",
                options=paises_disponiveis,
                default=["BR", "SE"] if set(["BR", "SE"]).issubset(paises_disponiveis) else list(paises_disponiveis),
                key="filtro_paises_evolucao"
            )
        
        dados['MES_ANO'] = dados['DT_PUBLICACAO'].dt.to_period('M').dt.to_timestamp()
        evolucao = dados[
            (dados['SENTIMENTO'].isin(sentimentos_selecionados)) & 
            (dados['PAIS'].isin(paises_selecionados))
        ]
        
        evolucao = evolucao.groupby(['MES_ANO', 'MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='COUNT')
        
        fig3 = px.line(
            evolucao,
            x='MES_ANO',
            y='COUNT',
            color='MARCA_PAIS',
            line_dash='SENTIMENTO',
            markers=True,
            title='Evolução temporal dos sentimentos por Marca e País',
            labels={'MES_ANO': 'Date', 'COUNT': 'Quantity'},
            category_orders={"SENTIMENTO": sentimentos_selecionados}
        )
        
        fig3.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b/%Y}<br>Quantity: %{y}',
            line=dict(width=2)
        )
        
        fig3.update_layout(
            xaxis_title='Período',
            yaxis_title='Quantidade',
            legend_title='Marca - País, Sentimento',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Evolução de Sentimentos por Marca ao Longo do Tempo (Anual)")
        
        anos_disponiveis = sorted(dados['ANO'].unique())
        ano_selecionado = st.selectbox(
            "Selecione o ano para detalhe:",
            options=anos_disponiveis,
            index=len(anos_disponiveis)-1,
            key="filtro_ano_sentimento"
        )
        
        dados_filtrados = dados[dados['ANO'] == ano_selecionado]
        dados_filtrados['MES_ANO_FORMATADO'] = dados_filtrados['DT_PUBLICACAO'].dt.strftime('%b %Y')
        dados_filtrados = dados_filtrados.sort_values('DT_PUBLICACAO')
        
        ordem_cronologica = dados_filtrados.sort_values('DT_PUBLICACAO')['MES_ANO_FORMATADO'].unique()
        
        evolucao_sentimento = dados_filtrados.groupby(
            ['MES_ANO_FORMATADO', 'DT_PUBLICACAO', 'MARCA_PAIS', 'SENTIMENTO']
        ).size().reset_index(name='COUNT')
        
        fig4 = px.bar(
            evolucao_sentimento,
            x='MES_ANO_FORMATADO',
            y='COUNT',
            color='SENTIMENTO',
            facet_col='MARCA_PAIS',
            barmode='group',
            title=f'Sentimento por Marca e País em {ano_selecionado}',
            labels={'MES_ANO_FORMATADO': 'Mês/Ano', 'COUNT': 'Quantidade'},
            category_orders={"MES_ANO_FORMATADO": list(ordem_cronologica)}
        )
        
        fig4.update_xaxes(type='category', tickangle=45)
        fig4.update_layout(
            yaxis_title='Quantidade',
            hovermode='x unified',
            showlegend=True,
            xaxis_title='Mês/Ano'
        )
        fig4.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Mês/Ano: %{x}<br>Quantidade: %{y:,}'
        )
        
        fig4.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig4.update_layout(
            margin=dict(l=50, r=50, t=80, b=150),
            height=500
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Dados de sentimento não disponíveis para as marcas e período selecionados.")

# --- Seção de Visualização Bidimensional ---
with st.expander("Visualização Bidimensional (T-SNE)", expanded=False):
    if len(marcas_selecionadas) > 1:
        marca_visualizada = st.selectbox(
            "Escolha uma marca para visualizações específicas:", 
            marcas_selecionadas, 
            key="select_marca_tsne"
        )
        dados_tsne = dados[dados["MARCA"] == marca_visualizada]
    else:
        dados_tsne = dados.copy()

    st.subheader("Distribuição de Publicações no Espaço 2D (T-SNE)")
    fig = px.scatter(
        dados_tsne,
        x="X",
        y="Y",
        color="MARCA",
        hover_data=["CURTIDAS", "COMENTARIOS", "DT_PUBLICACAO", "TEXTO"],
        title=f"Visualização T-SNE das publicações - {marca_visualizada if len(marcas_selecionadas) > 1 else 'Todas as Marcas'}"
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='black'), size=10),
        hovertemplate='<b>Marca: %{customdata[3]}</b><br>X: %{x}<br>Y: %{y}<br>Curtidas: %{customdata[0]:,}<br>Comentários: %{customdata[1]:,}<br>Data: %{customdata[2]}<br>Texto: %{customdata[4]}'
    )
    fig.update_layout(
        template="plotly_white",
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Seção de Nuvem de Palavras ---
with st.expander("Nuvem de Palavras", expanded=False):
    if len(marcas_selecionadas) > 1:
        marca_visualizada = st.selectbox(
            "Escolha uma marca para visualizações específicas:", 
            marcas_selecionadas, 
            key="select_marca_wordcloud"
        )
        dados_wc = dados[dados["MARCA"] == marca_visualizada]
    else:
        dados_wc = dados.copy()

    st.subheader("Nuvem de Palavras das Publicações Selecionadas")
    textos = ' '.join(dados_wc['TEXTO'].dropna().str.lower().tolist())
    if textos:
        wc = WordCloud(max_words=200, colormap='viridis', stopwords=stop_words, width=800, height=400, background_color='white').generate(textos)
        st.image(wc.to_array(), caption=f"Nuvem de Palavras - {marca_visualizada if len(marcas_selecionadas) > 1 else 'Todas as Marcas'}")
    else:
        st.info("Não há texto disponível para gerar a nuvem de palavras com os filtros atuais.")

# --- Seção de Clusterização KMeans ---
with st.expander("Clusterização KMeans", expanded=False):
    if len(marcas_selecionadas) > 1:
        marca_visualizada = st.selectbox(
            "Escolha uma marca para visualizações específicas:", 
            marcas_selecionadas, 
            key="select_marca_kmeans"
        )
        dados_cluster = dados[dados["MARCA"] == marca_visualizada]
    else:
        dados_cluster = dados.copy()

    st.subheader("Análise de Clusters de Publicações")
    ncluster = st.number_input(
        "Número de clusters (k)", value=6, placeholder="Quantos clusters você acha que existem?",
        min_value=1, max_value=30, step=1, key="num_clusters_kmeans"
    )

    if dados_cluster.empty or dados_cluster[['X', 'Y']].isnull().values.any():
        st.warning("Dados insuficientes ou inválidos para clusterização. Verifique os filtros.")
    else:
        matriz = dados_cluster[['X', 'Y']].values
        try:
            kmeans = KMeans(n_clusters=ncluster, n_init=30, random_state=42).fit(matriz)
            dados_cluster['CLUSTER'] = kmeans.labels_.astype(str)

            st.subheader("Visualização 2D dos Clusters")
            fig_cluster_2d = px.scatter(
                dados_cluster,
                x="X",
                y="Y",
                color="CLUSTER",
                hover_data=["CURTIDAS", "COMENTARIOS", "DT_PUBLICACAO", "MARCA", "TEXTO"],
                title=f"Clusters de publicações (k={ncluster}) - {marca_visualizada if len(marcas_selecionadas) > 1 else 'Todas as Marcas'}"
            )
            fig_cluster_2d.update_traces(
                marker=dict(line=dict(width=1, color='black'), size=10),
                hovertemplate='<b>Cluster %{marker.color}</b><br>X: %{x}<br>Y: %{y}<br>Curtidas: %{customdata[0]:,}<br>Comentários: %{customdata[1]:,}<br>Data: %{customdata[2]}<br>Marca: %{customdata[3]}<br>Texto: %{customdata[4]}'
            )
            fig_cluster_2d.update_layout(
                template="plotly_white",
                hovermode='closest'
            )
            st.plotly_chart(fig_cluster_2d, use_container_width=True)

            st.subheader("Distribuição de Publicações por Cluster")
            st.write("A tabela abaixo mostra a quantidade de marcas distintas e publicações em cada cluster.")
            st.dataframe(dados_cluster.groupby('CLUSTER').agg({'MARCA': 'nunique', 'LINK_PUBLICACAO': 'nunique'}).rename(columns={'MARCA': 'Marcas Distintas', 'LINK_PUBLICACAO': 'Total de Publicações'}))

            st.subheader("Nuvem de Palavras por Cluster")
            lista_cluster = sorted(dados_cluster['CLUSTER'].unique())
            cluster_escolhido = st.selectbox("Selecione um CLUSTER para ver sua Nuvem de Palavras:", lista_cluster, key="select_cluster_wc")
            textos_cluster = ' '.join(dados_cluster[dados_cluster.CLUSTER == cluster_escolhido]['TEXTO'].dropna().str.lower())
            if textos_cluster:
                wc_cluster = WordCloud(max_words=200, colormap='viridis', stopwords=stop_words, width=800, height=400, background_color='white').generate(textos_cluster)
                st.image(wc_cluster.to_array(), caption=f"Nuvem de Palavras para o Cluster {cluster_escolhido}")
            else:
                st.info(f"Não há texto disponível para gerar a nuvem de palavras para o Cluster {cluster_escolhido}.")

            st.subheader(f"Detalhes das Publicações no CLUSTER = {cluster_escolhido}")
            if 'EMBEDDING' in dados_cluster.columns:
                st.dataframe(dados_cluster[dados_cluster.CLUSTER == cluster_escolhido].drop("EMBEDDING", axis=1))
            else:
                st.dataframe(dados_cluster[dados_cluster.CLUSTER == cluster_escolhido])
        except Exception as e:
            st.error(f"Erro ao realizar a clusterização: {e}. Tente ajustar o número de clusters ou os filtros.")

# --- Seção de Análise de Reclamações (Reclame Aqui) ---
with st.expander("Análise de Reclamações (Reclame Aqui)", expanded=True):
    st.subheader("Análise de Reclamações por Marca")
    
    # Carrega os dados
    df_reclame = carregar_dados_reclame_aqui()
    
    # Filtro por marca
    marcas_reclame = sorted(df_reclame['MARCA'].unique())
    marca_selecionada = st.selectbox(
        "Selecione a marca para análise:",
        options=marcas_reclame,
        index=0,
        key="select_marca_reclame"
    )
    
    # Filtra os dados pela marca selecionada
    dados_marca = df_reclame[df_reclame['MARCA'] == marca_selecionada]
    
    # Cria três colunas para os cards de métricas
    col1, col2, col3 = st.columns(3)
    
    def criar_card(categoria, titulo):
        dados_categoria = dados_marca[dados_marca['CATEGORIA'].str.strip().str.lower() == categoria.lower().strip()]
        
        if len(dados_categoria) == 0:
            return f"""
            <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
                <h4 style="color:#333;margin-top:0;">{titulo}</h4>
                <p style="color:#666;">Dados não disponíveis para esta marca</p>
            </div>
            """
        
        row = dados_categoria.iloc[0]
        itens = []
        
        # Verifica cada item (1-5)
        for i in range(1, 6):
            item_col = f'ITEM{i}'
            qtd_col = f'QUANTIDADE{i}'
            
            if item_col in row and qtd_col in row:
                item = str(row[item_col]).strip()
                quantidade = str(row[qtd_col]).strip()
                
                if item and item.lower() != 'nan' and quantidade and quantidade.lower() != 'nan':
                    itens.append(f"<p style='margin-bottom:5px;'>{item} ({quantidade})</p>")
        
        porcentagem = str(row['PORCENTAGEM1']).strip() if 'PORCENTAGEM1' in row and pd.notna(row['PORCENTAGEM1']) else 'N/A'
        
        return f"""
        <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
            <h4 style="color:#333;margin-top:0;">{titulo}</h4>
            <p style="font-size:24px;font-weight:bold;color:#2c3e50;margin-bottom:5px;">
                {porcentagem}
            </p>
            {''.join(itens)}
        </div>
        """
    
    with col1:
        st.markdown(criar_card('Tipos de problemas', 'Tipos de problemas'), unsafe_allow_html=True)
    
    with col2:
        st.markdown(criar_card('Produtos e Serviços', 'Produtos e Serviços'), unsafe_allow_html=True)
    
    with col3:
        st.markdown(criar_card('Categorias', 'Categorias'), unsafe_allow_html=True)
    
    # Gráficos adicionais para análise visual
    st.subheader("Visualização Gráfica das Reclamações")
    
    def criar_grafico(categoria, titulo):
        dados_categoria = dados_marca[dados_marca['CATEGORIA'] == categoria]
        if len(dados_categoria) == 0:
            st.warning(f"Não há dados disponíveis para {titulo.lower()}")
            return None
        
        row = dados_categoria.iloc[0]
        itens = []
        quantidades = []
        
        for i in range(1, 6):
            item = row[f'ITEM{i}']
            quantidade = row[f'QUANTIDADE{i}']
            if pd.notna(item) and pd.notna(quantidade):
                itens.append(item)
                quantidades.append(quantidade)
        
        if not itens:
            st.warning(f"Não há dados válidos para {titulo.lower()}")
            return None
        
        df = pd.DataFrame({'Item': itens, 'Quantidade': quantidades})
        fig = px.bar(df, x='Item', y='Quantidade', 
                    title=f'{titulo} - {marca_selecionada}',
                    color='Item')
        fig.update_layout(showlegend=False)
        return fig
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = criar_grafico('Tipos de problemas', 'Tipos de Problemas')
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = criar_grafico('Produtos e Serviços', 'Produtos e Serviços')
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = criar_grafico('Categorias', 'Categorias')
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)