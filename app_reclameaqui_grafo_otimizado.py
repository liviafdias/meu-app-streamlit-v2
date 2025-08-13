# app_reclameaqui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
from sklearn.cluster import KMeans

# --- Imports p/ grafo ---
import re
import itertools
from collections import Counter
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
# ------------------------

st.set_page_config(page_title="Painel de Tendências", layout="wide", initial_sidebar_state="expanded")

# (opcional) ícones
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# ------------------ Funções utilitárias ------------------
def ajeita_data(t):
    try:
        return pd.to_datetime(t)
    except:
        return pd.NaT

def len_texto(x):
    try:
        return len(str(x).split())
    except:
        return 0

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
    return set(stop_df.STOP_WORDS.values)

@st.cache_data
def carregar_dados_reclame_aqui():
    # Lê o CSV considerando que não há cabeçalho
    try:
        df = pd.read_csv('data/reclameaqui.csv', sep=',', header=None)  # ajuste sep se necessário
    except Exception as e:
        st.error(f"Erro ao carregar 'reclameaqui.csv': {e}")
        return pd.DataFrame()

    # Define nomes das colunas
    df.columns = [
        'MARCA', 'CATEGORIA',
        'ITEM1', 'PORCENTAGEM1', 'QUANTIDADE1',
        'ITEM2', 'QUANTIDADE2',
        'ITEM3', 'QUANTIDADE3',
        'ITEM4', 'QUANTIDADE4',
        'ITEM5', 'QUANTIDADE5'
    ]

    # Garantir que a coluna PORCENTAGEM1 é string
    df['PORCENTAGEM1'] = df['PORCENTAGEM1'].astype(str)

    return df

# --------- Funções do grafo (baseadas no seu grafo.py) ---------
def find_topics_column(df: pd.DataFrame, hint: str = "TOPICOS") -> str:
    cols = list(df.columns)
    cands = [c for c in cols if str(c).strip().lower() == hint.lower()]
    if cands: return cands[0]
    cands = [c for c in cols if "topico" in str(c).strip().lower()]
    if cands: return cands[0]
    cands = [c for c in cols if str(c).strip().lower() in {"tópicos", "tópico", "topics", "topic"}]
    if cands: return cands[0]
    raise ValueError(f'Não encontrei a coluna "{hint}". Colunas: {cols}')

def parse_topics(cell) -> list[str]:
    if pd.isna(cell): return []
    s = str(cell)
    s = re.sub(r"[|;/]+", ",", s)  # normaliza delimitadores
    parts = [p.strip() for p in s.split(",") if str(p).strip()]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

@st.cache_data
def build_cooccurrence(topic_lists: pd.Series):
    all_topics = [t for sub in topic_lists for t in sub]
    node_freq = Counter(all_topics)
    edge_weights = Counter()
    for topics in topic_lists:
        if len(topics) >= 2:
            for a, b in itertools.combinations(sorted(topics), 2):
                edge_weights[(a, b)] += 1
    nodes_df = (pd.DataFrame([{"node": n, "freq": int(f)} for n, f in node_freq.items()])
                .sort_values("freq", ascending=False).reset_index(drop=True))
    edges_df = (pd.DataFrame([{"ORIGEM": s, "DESTINO": t, "W": int(w)} for (s, t), w in edge_weights.items()])
                .sort_values("W", ascending=False).reset_index(drop=True))
    return nodes_df, edges_df

def compute_layout(G: nx.Graph, seed=42):
    return nx.spring_layout(G, seed=seed)

def figure_size_from_pos(pos: dict, base_height=14, dpi=300):
    xs = [p[0] for p in pos.values()] or [0, 1]
    ys = [p[1] for p in pos.values()] or [0, 1]
    width = (max(xs) - min(xs)) or 1e-9
    height = (max(ys) - min(ys)) or 1e-9
    aspect = width / height
    fig_h = base_height
    fig_w = max(fig_h * aspect, fig_h * 0.8)
    return (fig_w, fig_h, dpi)
# ---------------------------------------------------------------

# ------------------ Dados base e filtros ------------------
st.markdown('<h1 style="text-align:center;">Painel de Tendências</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:grey;">Explore tendências no Instagram e dados do Reclame Aqui</p>', unsafe_allow_html=True)

# Carregamentos
stop_words = carregar_stop_words_br()
try:
    brasil = carregar_dados_brasil()
except Exception as e:
    st.warning(f"Não foi possível carregar dados do Brasil: {e}")
    brasil = pd.DataFrame()
try:
    suecia = carregar_dados_suecia()
except Exception as e:
    st.warning(f"Não foi possível carregar dados da Suécia: {e}")
    suecia = pd.DataFrame()

# Sidebar (filtros do Instagram) — INICIAM VAZIOS
with st.sidebar:
    st.header("Filtros (Instagram)")

    # Constrói lista de marcas a partir do BR (se vazio, tenta SE)
    base_marcas = brasil if not brasil.empty else suecia
    todas_marcas = sorted(list(base_marcas["MARCA"].unique())) if "MARCA" in base_marcas.columns else []

    marcas_selecionadas = st.multiselect(
        "Comparar Marcas",
        options=todas_marcas,
        default=[],                     # inicia vazio
        placeholder="+ Adicionar marcas",
        key="sb_marcas"
    )

    # Países disponíveis a partir dos dados carregados
    paises_opts = []
    if not brasil.empty: paises_opts.append("BR")
    if not suecia.empty: paises_opts.append("SE")

    paises = st.multiselect(
        "País",
        options=paises_opts or ["BR", "SE"],
        default=[],                     # inicia vazio
        placeholder="Selecionar país(es)",
        key="sb_paises"
    )

    # Anos dinâmicos a partir dos dados disponíveis
    anos_br = brasil['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in brasil.columns else []
    anos_se = suecia['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in suecia.columns else []
    anos_disponiveis = sorted(set(anos_br) | set(anos_se)) or [2020, 2021, 2022, 2023, 2024]

    periodo_selecionado = st.multiselect(
        "Período (anos)",
        options=anos_disponiveis,
        default=[],                     # inicia vazio
        placeholder="Selecionar ano(s)",
        key="sb_anos"
    )

    agrupamento = st.radio(
        "Agrupar por",
        options=["mês", "ano"],
        index=0,
        horizontal=True,
        key="sb_agrupar"
    )

# Constrói dataframe filtrado do Instagram (sem st.stop)
def filtra_instagram():
    df_list = []
    if "BR" in paises and not brasil.empty:
        df_br = brasil.copy(); df_br["PAIS"] = "BR"; df_list.append(df_br)
    if "SE" in paises and not suecia.empty:
        df_se = suecia.copy(); df_se["PAIS"] = "SE"; df_list.append(df_se)
    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    if periodo_selecionado:
        df["ANO"] = df["DT_PUBLICACAO"].dt.year
        df = df[df["ANO"].isin(periodo_selecionado)]
    if marcas_selecionadas:
        df = df[df["MARCA"].isin(marcas_selecionadas)]
    df["MARCA_PAIS"] = df["MARCA"] + " - " + df["PAIS"]
    if agrupamento == "ano":
        df['PERIODO'] = df['DT_PUBLICACAO'].dt.to_period("Y").dt.to_timestamp()
    else:
        df['PERIODO'] = df['DT_PUBLICACAO'].dt.to_period("M").dt.to_timestamp()
    return df

dados = filtra_instagram()

# ------------------ Abas principais ------------------
tab_instagram, tab_reclameaqui = st.tabs(["Instagram", "Reclame Aqui"])

# ===========================
# TAB: Instagram
# ===========================
with tab_instagram:
    ig_tab1, ig_tab2, ig_tab3, ig_tab4, ig_tab5 = st.tabs([
        "Métricas ao Longo do Tempo",
        "Análise de Sentimentos",
        "Visualização Bidimensional (T-SNE)",
        "Nuvem de Palavras",
        "Clusterização KMeans"
    ])

    # --- Métricas ao longo do tempo ---
    with ig_tab1:
        st.subheader("Evolução de Publicações, Curtidas e Comentários")
        if dados.empty:
            st.info("Ajuste os filtros na barra lateral para ver as métricas do Instagram.")
        else:
            df_pub = dados.groupby(['PERIODO', 'MARCA_PAIS']).size().reset_index(name='PUBLICAÇÕES')
            fig_pub = px.line(df_pub, x='PERIODO', y='PUBLICAÇÕES', color='MARCA_PAIS',
                              title=f'Publicações por {agrupamento}', markers=True)
            fig_pub.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                                  legend_title='Marca - País', hovermode='x unified')
            fig_pub.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Publicações: %{y:,}')
            st.plotly_chart(fig_pub, use_container_width=True)

            df_curtidas = dados.groupby(['PERIODO', 'MARCA_PAIS'])['CURTIDAS'].sum().reset_index()
            fig_curtidas = px.bar(df_curtidas, x='PERIODO', y='CURTIDAS', color='MARCA_PAIS',
                                  barmode='group', title=f'Curtidas por {agrupamento}')
            fig_curtidas.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                                       legend_title='Marca - País', hovermode='x unified')
            fig_curtidas.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Curtidas: %{y:,}')
            st.plotly_chart(fig_curtidas, use_container_width=True)

            df_comentarios = dados.groupby(['PERIODO', 'MARCA_PAIS'])['COMENTARIOS'].sum().reset_index()
            fig_comentarios = px.bar(df_comentarios, x='PERIODO', y='COMENTARIOS', color='MARCA_PAIS',
                                     barmode='group', title=f'Comentários por {agrupamento}')
            fig_comentarios.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                                          legend_title='Marca - País', hovermode='x unified')
            fig_comentarios.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Comentários: %{y:,}')
            st.plotly_chart(fig_comentarios, use_container_width=True)

            # Cartas rápidas
            colm1, colm2, colm3 = st.columns(3)
            total_publicacoes = len(dados)
            total_curtidas = int(dados['CURTIDAS'].sum())
            total_comentarios = int(dados['COMENTARIOS'].sum())
            colm1.metric("Total de Publicações", f"{total_publicacoes:,}".replace(",", "."))
            colm2.metric("Total de Curtidas", f"{total_curtidas:,}".replace(",", "."))
            colm3.metric("Total de Comentários", f"{total_comentarios:,}".replace(",", "."))

    # --- Análise de sentimentos ---
    with ig_tab2:
        st.subheader("Distribuição e Evolução dos Sentimentos")
        if dados.empty:
            st.info("Ajuste os filtros para visualizar os sentimentos.")
        elif 'SENTIMENTO' not in dados.columns:
            st.warning("Coluna 'SENTIMENTO' não encontrada nos dados.")
        else:
            dist_sent = dados['SENTIMENTO'].value_counts().reset_index()
            dist_sent.columns = ['Sentimento', 'Quantidade']
            fig1 = px.bar(dist_sent, x='Sentimento', y='Quantidade', color='Sentimento',
                          title="Frequência dos sentimentos")
            st.plotly_chart(fig1, use_container_width=True)

            dist_sent['Percentual'] = (dist_sent['Quantidade'] / dist_sent['Quantidade'].sum()) * 100
            dist_sent['Percentual'] = dist_sent['Percentual'].round(2)
            fig_pizza = px.pie(dist_sent, names='Sentimento', values='Percentual',
                               title='Percentual de Publicações por Sentimento', hole=0.4)
            fig_pizza.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pizza, use_container_width=True)

            st.subheader("Distribuição de Sentimentos por Marca")
            df_marca_sent = dados.groupby(['MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='Quantidade')
            fig2 = px.bar(df_marca_sent, x='MARCA_PAIS', y='Quantidade', color='SENTIMENTO',
                          barmode='group', title="Distribuição de sentimentos por Marca e País")
            fig2.update_layout(xaxis_tickangle=-45, hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Evolução temporal por Sentimento")
            sentimentos_disponiveis = sorted(dados['SENTIMENTO'].dropna().unique().tolist())
            sentimentos_selecionados = st.multiselect(
                "Selecione sentimentos:", options=sentimentos_disponiveis,
                default=sentimentos_disponiveis[:2], key="ig_sent_sel")
            paises_disponiveis = sorted(dados['PAIS'].dropna().unique().tolist())
            paises_selecionados = st.multiselect(
                "Selecione países:", options=paises_disponiveis,
                default=paises_disponiveis, key="ig_paises_sel"
            )
            dados['MES_ANO'] = dados['DT_PUBLICACAO'].dt.to_period('M').dt.to_timestamp()
            evolucao = dados[(dados['SENTIMENTO'].isin(sentimentos_selecionados)) &
                             (dados['PAIS'].isin(paises_selecionados))]
            evolucao = evolucao.groupby(['MES_ANO', 'MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='COUNT')
            if evolucao.empty:
                st.info("Sem dados para os filtros atuais.")
            else:
                fig3 = px.line(evolucao, x='MES_ANO', y='COUNT', color='MARCA_PAIS',
                               line_dash='SENTIMENTO', markers=True,
                               title='Evolução temporal dos sentimentos por Marca e País')
                fig3.update_layout(hovermode='x unified')
                st.plotly_chart(fig3, use_container_width=True)

    # --- T-SNE ---
    with ig_tab3:
        st.subheader("Visualização Bidimensional (T-SNE)")
        if dados.empty:
            st.info("Ajuste os filtros para visualizar o T-SNE.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_tsne_marca")
                dados_tsne = dados[dados["MARCA"] == marca_visualizada]
            else:
                dados_tsne = dados.copy()
            fig = px.scatter(
                dados_tsne, x="X", y="Y", color="MARCA",
                hover_data=["CURTIDAS", "COMENTARIOS", "DT_PUBLICACAO", "TEXTO"],
                title=f"Visualização T-SNE - {marca_visualizada if len(marcas_selecionadas)>1 else 'marcas selecionadas'}"
            )
            fig.update_traces(marker=dict(line=dict(width=1, color='black'), size=10))
            st.plotly_chart(fig, use_container_width=True)

    # --- Nuvem de palavras ---
    with ig_tab4:
        st.subheader("Nuvem de Palavras")
        if dados.empty:
            st.info("Ajuste os filtros para gerar a nuvem.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_wc_marca")
                dados_wc = dados[dados["MARCA"] == marca_visualizada]
            else:
                dados_wc = dados.copy()
            textos = ' '.join(dados_wc['TEXTO'].dropna().astype(str).str.lower().tolist())
            if textos.strip():
                wc = WordCloud(max_words=200, colormap='viridis',
                               stopwords=stop_words, width=900, height=420,
                               background_color='white').generate(textos)
                st.image(wc.to_array(), caption="Nuvem de Palavras")
            else:
                st.info("Não há texto disponível com os filtros atuais.")

    # --- Clusterização KMeans ---
    with ig_tab5:
        st.subheader("Clusterização KMeans")
        if dados.empty or dados[['X','Y']].isnull().values.any():
            st.info("Dados insuficientes ou inválidos para clusterização.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_km_marca")
                dados_cluster = dados[dados["MARCA"] == marca_visualizada].copy()
            else:
                dados_cluster = dados.copy()

            ncluster = st.number_input("Número de clusters (k)", value=6, min_value=1, max_value=30, step=1, key="ig_km_k")
            try:
                matriz = dados_cluster[['X','Y']].values
                kmeans = KMeans(n_clusters=ncluster, n_init=30, random_state=42).fit(matriz)
                dados_cluster['CLUSTER'] = kmeans.labels_.astype(str)

                fig_cluster_2d = px.scatter(
                    dados_cluster, x="X", y="Y", color="CLUSTER",
                    hover_data=["CURTIDAS","COMENTARIOS","DT_PUBLICACAO","MARCA","TEXTO"],
                    title=f"Clusters (k={ncluster})"
                )
                fig_cluster_2d.update_traces(marker=dict(line=dict(width=1, color='black'), size=10))
                st.plotly_chart(fig_cluster_2d, use_container_width=True)

                st.subheader("Distribuição por Cluster")
                if 'LINK_PUBLICACAO' in dados_cluster.columns:
                    st.dataframe(dados_cluster.groupby('CLUSTER')
                                 .agg({'MARCA':'nunique','LINK_PUBLICACAO':'nunique'})
                                 .rename(columns={'MARCA':'Marcas Distintas','LINK_PUBLICACAO':'Total de Publicações'}))
                else:
                    st.dataframe(dados_cluster.groupby('CLUSTER')
                                 .agg({'MARCA':'nunique','X':'count'})
                                 .rename(columns={'MARCA':'Marcas Distintas','X':'Publicações (contagem)'}))

                st.subheader("Nuvem de Palavras por Cluster")
                lista_cluster = sorted(dados_cluster['CLUSTER'].unique())
                cluster_escolhido = st.selectbox("Selecione um cluster:", lista_cluster, key="ig_km_cluster_sel")
                textos_cluster = ' '.join(dados_cluster[dados_cluster.CLUSTER == cluster_escolhido]['TEXTO'].dropna().astype(str).str.lower())
                if textos_cluster.strip():
                    wc_cluster = WordCloud(max_words=200, colormap='viridis',
                                           stopwords=stop_words, width=900, height=420,
                                           background_color='white').generate(textos_cluster)
                    st.image(wc_cluster.to_array(), caption=f"Nuvem - Cluster {cluster_escolhido}")
                else:
                    st.info("Sem texto para a nuvem deste cluster.")
            except Exception as e:
                st.error(f"Erro ao clusterizar: {e}")

# ===========================
# TAB: Reclame Aqui
# ===========================
with tab_reclameaqui:
    rq_tab1, rq_tab2 = st.tabs([
        "Análise de Reclamações",
        "Grafo de Coocorrência (PEUGEOT/TOYOTA)"
    ])

    # --- Análise de Reclamações ---
    with rq_tab1:
        st.subheader("Análise de Reclamações por Marca")
        try:
            df_reclame = carregar_dados_reclame_aqui()
        except Exception as e:
            st.error(f"Erro ao carregar 'data/reclameaqui.xlsx': {e}")
            df_reclame = pd.DataFrame()

        if df_reclame.empty:
            st.info("Adicione o arquivo 'data/reclameaqui.xlsx' para visualizar esta aba.")
        else:
            marcas_reclame = sorted(df_reclame['MARCA'].dropna().unique().tolist())
            marca_selecionada = st.selectbox("Selecione a marca:", options=marcas_reclame, index=0, key="rq_marca")

            dados_marca = df_reclame[df_reclame['MARCA'] == marca_selecionada]

            def criar_card(categoria, titulo):
                dados_categoria = dados_marca[dados_marca['CATEGORIA'].astype(str).str.strip().str.lower() == categoria.lower().strip()]
                if len(dados_categoria) == 0:
                    return f"""
                    <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
                        <h4 style="color:#333;margin-top:0;">{titulo}</h4>
                        <p style="color:#666;">Dados não disponíveis para esta marca</p>
                    </div>
                    """
                row = dados_categoria.iloc[0]
                itens = []
                for i in range(1, 6):
                    item_col = f'ITEM{i}'; qtd_col = f'QUANTIDADE{i}'
                    item = str(row.get(item_col, "")).strip()
                    quantidade = str(row.get(qtd_col, "")).strip()
                    if item and item.lower() != 'nan' and quantidade and quantidade.lower() != 'nan':
                        itens.append(f"<p style='margin-bottom:5px;'>{item} ({quantidade})</p>")
                porcentagem = str(row.get('PORCENTAGEM1', 'N/A')).strip()
                return f"""
                <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
                    <h4 style="color:#333;margin-top:0;">{titulo}</h4>
                    <p style="font-size:24px;font-weight:bold;color:#2c3e50;margin-bottom:5px;">{porcentagem}</p>
                    {''.join(itens)}
                </div>
                """

            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(criar_card('Tipos de problemas', 'Tipos de problemas'), unsafe_allow_html=True)
            with c2: st.markdown(criar_card('Produtos e Serviços', 'Produtos e Serviços'), unsafe_allow_html=True)
            with c3: st.markdown(criar_card('Categorias', 'Categorias'), unsafe_allow_html=True)

            st.subheader("Visualização Gráfica")
            def criar_grafico(categoria, titulo):
                dados_categoria = dados_marca[dados_marca['CATEGORIA'] == categoria]
                if len(dados_categoria) == 0: return None
                row = dados_categoria.iloc[0]
                itens, quantidades = [], []
                for i in range(1, 5+1):
                    item = row.get(f'ITEM{i}')
                    quantidade = row.get(f'QUANTIDADE{i}')
                    if pd.notna(item) and pd.notna(quantidade):
                        itens.append(item); quantidades.append(quantidade)
                if not itens: return None
                df = pd.DataFrame({'Item': itens, 'Quantidade': quantidades})
                fig = px.bar(df, x='Item', y='Quantidade', title=f'{titulo} - {marca_selecionada}', color='Item')
                fig.update_layout(showlegend=False)
                return fig

            g1, g2, g3 = st.columns(3)
            with g1:
                fig1 = criar_grafico('Tipos de problemas', 'Tipos de Problemas')
                if fig1: st.plotly_chart(fig1, use_container_width=True)
            with g2:
                fig2 = criar_grafico('Produtos e Serviços', 'Produtos e Serviços')
                if fig2: st.plotly_chart(fig2, use_container_width=True)
            with g3:
                fig3 = criar_grafico('Categorias', 'Categorias')
                if fig3: st.plotly_chart(fig3, use_container_width=True)

    # --- Grafo PEUGEOT/TOYOTA ---
    with rq_tab2:
        st.subheader("Coocorrência de tópicos")
        marca_escolhida = st.selectbox(
            "Selecione a base de dados do grafo:",
            options=["PEUGEOT", "TOYOTA"],
            index=0,
            key="rq_grafo_base"
        )
        arquivo_map = {"PEUGEOT": "data/peugeot.csv", "TOYOTA": "data/toyota.csv"}
        arquivo_escolhido = arquivo_map[marca_escolhida]
        candidatos = [Path(arquivo_escolhido), Path("data") / arquivo_escolhido]
        xlsx_path = next((p for p in candidatos if p.exists()), None)

        if xlsx_path is None:
            st.error(f"Arquivo '{arquivo_escolhido}' não encontrado na raiz nem em 'data/'.")
            st.info("Coloque o arquivo .xlsx no mesmo diretório do app ou dentro da pasta 'data/'.")
        else:
            try:
                df_topics = pd.read_csv(xlsx_path)
                col = find_topics_column(df_topics, "TOPICOS")
                df_topics = df_topics.rename(columns={col: "TOPICOS"})
                topic_lists = df_topics["TOPICOS"].apply(parse_topics)

                nodes_df, edges_df = build_cooccurrence(topic_lists)
                if edges_df.empty or nodes_df.empty:
                    st.warning("Não foram encontradas coocorrências suficientes para montar o grafo.")
                else:
                    G = nx.from_pandas_edgelist(edges_df, "ORIGEM", "DESTINO", edge_attr="W", create_using=nx.Graph)
                    posx = compute_layout(G, seed=42)

                    LABEL_SOME_NODES = True
                    TOP_N_BY_FREQ = 20
                    LABEL_ALL_IF_SMALLER = 25
                    if (LABEL_SOME_NODES and G.number_of_nodes() > LABEL_ALL_IF_SMALLER):
                        listap = set(nodes_df.sort_values("freq", ascending=False).head(TOP_N_BY_FREQ)["node"])
                    else:
                        listap = set(G.nodes())

                    fig_w, fig_h, dpi = figure_size_from_pos(posx, base_height=14, dpi=300)
                    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                    ax = plt.gca()
                    ax.set_xlabel(" "); ax.set_ylabel(" ")
                    ax.axis("off"); ax.set_aspect("equal")

                    dgrapg = edges_df.sort_values("W", ascending=False).reset_index(drop=True)
                    z = dgrapg["W"].values
                    vmin, vmax = float(np.min(z)), float(np.max(z))
                    if np.isclose(vmin, vmax): vmin, vmax = vmin - 0.5, vmax + 0.5
                    normal = plt.Normalize(vmin, vmax)

                    for i in range(len(dgrapg)):
                        Gi = nx.from_pandas_edgelist(dgrapg.iloc[i:i+1], "ORIGEM", "DESTINO", "W")
                        caux = plt.cm.Oranges(normal(dgrapg.iloc[i]["W"]))
                        nx.draw_networkx_edges(Gi, posx, edge_color=caux, alpha=1.0,
                                               edge_cmap=plt.cm.Oranges, width=4.0, ax=ax)

                    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=normal)
                    cb = plt.colorbar(sm, ax=ax)
                    cb.set_label(r"$F_{ij}$", size=24)
                    for t in cb.ax.get_yticklabels(): t.set_fontsize(6)

                    for n, (x, y) in posx.items():
                        ax.scatter(x, y, s=100, zorder=2, edgecolor="black", lw=1.5, c="#556C8E")
                        if n in listap:
                            ax.annotate(n, xy=(x, y), fontsize=6, ha="center", va="center",
                                        xytext=(0, 10), textcoords="offset points")

                    st.caption(f"Base selecionada: **{marca_escolhida}** ({xlsx_path.as_posix()})")
                    st.pyplot(fig, clear_figure=True)

            except Exception as e:
                st.error(f"Erro ao gerar o grafo para '{marca_escolhida}': {e}")
