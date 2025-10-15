import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="Análise de Presença Digital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAPEAMENTO DE SEÇÕES CNAE
# ==============================================================================

def get_secoes_cnae():
    """Retorna estrutura completa de seções CNAE"""
    return {
        'A': {'divisoes': ['01', '02', '03'], 'nome': 'AGRICULTURA, PECUÁRIA, PRODUÇÃO FLORESTAL, PESCA E AQUICULTURA'},
        'B': {'divisoes': ['05', '06', '07', '08', '09'], 'nome': 'INDÚSTRIAS EXTRATIVAS'},
        'C': {'divisoes': [str(i).zfill(2) for i in range(10, 34)], 'nome': 'INDÚSTRIAS DE TRANSFORMAÇÃO'},
        'D': {'divisoes': ['35'], 'nome': 'ELETRICIDADE E GÁS'},
        'E': {'divisoes': ['36', '37', '38', '39'], 'nome': 'ÁGUA, ESGOTO, ATIVIDADES DE GESTÃO DE RESÍDUOS E DESCONTAMINAÇÃO'},
        'F': {'divisoes': ['41', '42', '43'], 'nome': 'CONSTRUÇÃO'},
        'G': {'divisoes': ['45', '46', '47'], 'nome': 'COMÉRCIO; REPARAÇÃO DE VEÍCULOS AUTOMOTORES E MOTOCICLETAS'},
        'H': {'divisoes': ['49', '50', '51', '52', '53'], 'nome': 'TRANSPORTE, ARMAZENAGEM E CORREIO'},
        'I': {'divisoes': ['55', '56'], 'nome': 'ALOJAMENTO E ALIMENTAÇÃO'},
        'J': {'divisoes': ['58', '59', '60', '61', '62', '63'], 'nome': 'INFORMAÇÃO E COMUNICAÇÃO'},
        'K': {'divisoes': ['64', '65', '66'], 'nome': 'ATIVIDADES FINANCEIRAS, DE SEGUROS E SERVIÇOS RELACIONADOS'},
        'L': {'divisoes': ['68'], 'nome': 'ATIVIDADES IMOBILIÁRIAS'},
        'M': {'divisoes': ['69', '70', '71', '72', '73', '74', '75'], 'nome': 'ATIVIDADES PROFISSIONAIS, CIENTÍFICAS E TÉCNICAS'},
        'N': {'divisoes': ['77', '78', '79', '80', '81', '82'], 'nome': 'ATIVIDADES ADMINISTRATIVAS E SERVIÇOS COMPLEMENTARES'},
        'O': {'divisoes': ['84'], 'nome': 'ADMINISTRAÇÃO PÚBLICA, DEFESA E SEGURIDADE SOCIAL'},
        'P': {'divisoes': ['85'], 'nome': 'EDUCAÇÃO'},
        'Q': {'divisoes': ['86', '87', '88'], 'nome': 'SAÚDE HUMANA E SERVIÇOS SOCIAIS'},
        'R': {'divisoes': ['90', '91', '92', '93'], 'nome': 'ARTES, CULTURA, ESPORTE E RECREAÇÃO'},
        'S': {'divisoes': ['94', '95', '96'], 'nome': 'OUTRAS ATIVIDADES DE SERVIÇOS'},
        'T': {'divisoes': ['97'], 'nome': 'SERVIÇOS DOMÉSTICOS'},
        'U': {'divisoes': ['99'], 'nome': 'ORGANISMOS INTERNACIONAIS E OUTRAS INSTITUIÇÕES EXTRATERRITORIAIS'}
    }

def get_divisao_para_secao():
    """Cria mapeamento reverso: divisão -> seção"""
    secoes = get_secoes_cnae()
    divisao_para_secao = {}
    for secao, dados in secoes.items():
        for divisao in dados['divisoes']:
            divisao_para_secao[divisao] = secao
    return divisao_para_secao

def formatar_cnae(cnae_int):
    """Formata CNAE como string sem pontos decimais (ex: 111301)"""
    if pd.isna(cnae_int) or cnae_int == 0:
        return 'N/A'
    return str(int(cnae_int))

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

@st.cache_data
def load_data():
    """Carrega e processa os dados"""
    df = pd.read_csv('dados/tabela_final_empresas_reabertas_completo.csv', sep=';')

    # Carregar descrições CNAE e REMOVER DUPLICATAS
    df_cnae_desc = pd.read_csv('dados/codigos_cnae_2.csv', sep=';', encoding='utf-8')
    df_cnae_desc = df_cnae_desc.dropna(subset=['CNAE'])
    df_cnae_desc['CNAE'] = df_cnae_desc['CNAE'].astype(int)

    # IMPORTANTE: Remover duplicatas mantendo a descrição sem traço no início
    # Prioriza descrições sem "-" no início
    df_cnae_desc['tem_traco'] = df_cnae_desc['DESCRIÇÃO'].str.startswith('-')
    df_cnae_desc = df_cnae_desc.sort_values('tem_traco')  # False vem primeiro
    df_cnae_desc = df_cnae_desc.drop_duplicates(subset='CNAE', keep='first')
    df_cnae_desc = df_cnae_desc.drop('tem_traco', axis=1)

    # Limpar descrições que começam com "- "
    df_cnae_desc['DESCRIÇÃO'] = df_cnae_desc['DESCRIÇÃO'].str.replace('^- ', '', regex=True)

    # Preparação das variáveis
    df['CEP_str'] = df['CEP'].astype(str).str.replace('.0', '').str.zfill(8)
    df['CEP_regiao'] = df['CEP_str'].str[:5]

    # CNAE completo formatado (6 dígitos)
    df['cnae_fiscal_int'] = df['cnae_fiscal_principal'].fillna(0).astype(int)
    df['CNAE_completo'] = df['cnae_fiscal_int'].apply(formatar_cnae)

    df['CNAE_str'] = df['cnae_fiscal_principal'].astype(str).str.replace('.0', '').str.zfill(7)
    df['CNAE_divisao'] = df['CNAE_str'].str[:2]

    # Mapear para seção
    divisao_para_secao = get_divisao_para_secao()
    df['CNAE_secao'] = df['CNAE_divisao'].map(divisao_para_secao)

    # Merge com descrições CNAE (já sem duplicatas)
    df = df.merge(df_cnae_desc, left_on='cnae_fiscal_int', right_on='CNAE', how='left')
    df['cnae_descricao'] = df['DESCRIÇÃO'].fillna('Não especificado')

    # Criar label completo: CNAE - Descrição
    df['CNAE_label'] = df.apply(
        lambda row: f"{row['CNAE_completo']} - {row['cnae_descricao']}" if row['CNAE_completo'] != 'N/A' else 'N/A',
        axis=1
    )

    # Categorizar posts
    def categorizar_posts(n_posts):
        if n_posts == 0:
            return 'Sem posts'
        elif n_posts <= 100:
            return 'Baixa (1-100)'
        elif n_posts <= 500:
            return 'Média (101-500)'
        elif n_posts <= 1000:
            return 'Alta (501-1000)'
        else:
            return 'Muito Alta (>1000)'

    df['categoria_posts'] = df['numero_posts'].apply(categorizar_posts)

    # Porte atualizado
    porte_map = {1.0: 'Microempresa', 3.0: 'Pequeno Porte', 5.0: 'Médio e Grande Porte'}
    df['porte_desc'] = df['PORTE'].map(porte_map)

    return df

# ==============================================================================
# CARREGAR DADOS
# ==============================================================================

df = load_data()
secoes_cnae = get_secoes_cnae()

# ==============================================================================
# SIDEBAR - FILTROS
# ==============================================================================

st.sidebar.markdown("## 🎯 Filtros")

# Filtro de porte
portes_disponiveis = ['Todos'] + sorted(df['porte_desc'].dropna().unique().tolist())
porte_selecionado = st.sidebar.selectbox("Porte da Empresa", portes_disponiveis)

# Filtro de SEÇÃO CNAE (visão macro) - COM DIVISÕES
secoes_disponiveis = df['CNAE_secao'].dropna().unique()
secoes_opcoes = ['Todas'] + sorted([
    f"{s} ({', '.join(secoes_cnae[s]['divisoes'])}) - {secoes_cnae[s]['nome']}" 
    for s in secoes_disponiveis if s in secoes_cnae
])
secao_selecionada = st.sidebar.selectbox(
    "Seção CNAE (Visão Macro)", 
    secoes_opcoes,
    help="Filtre por seção CNAE para ver análise detalhada. Os números entre parênteses indicam as divisões da seção."
)

# Aplicar filtros
df_filtered = df.copy()

if porte_selecionado != 'Todos':
    df_filtered = df_filtered[df_filtered['porte_desc'] == porte_selecionado]

if secao_selecionada != 'Todas':
    # Extrair código da seção (primeira letra)
    secao_code = secao_selecionada.split(' ')[0]
    df_filtered = df_filtered[df_filtered['CNAE_secao'] == secao_code]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 Resumo Filtrado")
st.sidebar.metric("Total de Empresas", f"{len(df_filtered):,}")
st.sidebar.metric("Com Instagram", f"{df_filtered['tem_instagram'].sum():,}")
st.sidebar.metric("Taxa de Adoção", f"{df_filtered['tem_instagram'].mean()*100:.1f}%")

if secao_selecionada != 'Todas':
    secao_code = secao_selecionada.split(' ')[0]
    divisoes_secao = secoes_cnae[secao_code]['divisoes']
    st.sidebar.info(f"**Divisões:** {', '.join(divisoes_secao)}")

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<p class="main-header">📊 Dashboard para análise da presença digital de empresas no Instagram</p>', 
            unsafe_allow_html=True)

# ==============================================================================
# MÉTRICAS PRINCIPAIS
# ==============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

df_insta_filtered = df_filtered[df_filtered['tem_instagram'] == 1]

with col1:
    st.metric(
        "Total de Empresas da Amostra",
        f"{len(df_filtered):,}",
        delta=f"{len(df_filtered)/len(df)*100:.1f}% do total"
    )

with col2:
    st.metric(
        "Empresas Com Instagram",
        f"{df_filtered['tem_instagram'].sum():,}",
        delta=f"68.27% do total de empresas"
    )

with col3:
    media_posts = df_insta_filtered['numero_posts'].mean() if len(df_insta_filtered) > 0 else 0
    st.metric(
        "Média de Posts",
        f"{media_posts:,.0f}",
        delta=f"σ = {df_insta_filtered['numero_posts'].std():.0f}" if len(df_insta_filtered) > 0 else "N/A"
    )

with col4:
    mediana_posts = df_insta_filtered['numero_posts'].median() if len(df_insta_filtered) > 0 else 0
    st.metric(
        "Mediana de Posts",
        f"{mediana_posts:,.0f}",
        delta="273 Posts a menos que a média"
    )

with col5:
    max_posts = df_insta_filtered['numero_posts'].max() if len(df_insta_filtered) > 0 else 0
    st.metric(
        "Empresa com Mais Posts",
        f"{max_posts:,.0f}",
        delta="Outlier"
    )

st.markdown("---")

# ==============================================================================
# SEÇÃO 1: DISTRIBUIÇÕES GERAIS
# ==============================================================================

st.markdown('<p class="section-header">📊 Distribuições Gerais</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Distribuição por Categoria de Posts")
    cat_order = ['Sem posts', 'Baixa (1-100)', 'Média (101-500)', 
                 'Alta (501-1000)', 'Muito Alta (>1000)']
    cat_data = df_filtered['categoria_posts'].value_counts()
    cat_data = cat_data.reindex(cat_order, fill_value=0)

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_order,
        x=cat_data.values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v} ({v/len(df_filtered)*100:.1f}%)" if len(df_filtered) > 0 else "0" for v in cat_data.values],
        textposition='auto',
    ))
    fig.update_layout(
        xaxis_title="Número de Empresas",
        yaxis_title="",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Distribuição por Porte")
    porte_data = df_filtered['porte_desc'].value_counts()

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=porte_data.index,
        values=porte_data.values,
        hole=0.4,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("#### Histograma de Posts")
    if len(df_insta_filtered) > 0:
        limite = df_insta_filtered['numero_posts'].quantile(0.95)
        df_hist_plot = df_insta_filtered[df_insta_filtered['numero_posts'] <= limite]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_hist_plot['numero_posts'],
            nbinsx=30,
            marker=dict(color='steelblue', line=dict(color='black', width=1))
        ))
        fig.add_vline(
            x=df_insta_filtered['numero_posts'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Média: {df_insta_filtered['numero_posts'].mean():.0f}",
            annotation_position="top right"
        )
        fig.update_layout(
            xaxis_title="Número de Posts",
            yaxis_title="Frequência",
            height=400,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SEÇÃO 2: ANÁLISE POR PORTE
# ==============================================================================

st.markdown('<p class="section-header">🏢 Análise por Porte da Empresa</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

df_porte = df_filtered[df_filtered['porte_desc'].notna()].groupby('porte_desc').agg({
    'tem_instagram': ['sum', 'count', 'mean'],
    'numero_posts': ['mean', 'median']
}).round(2)
df_porte.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts']
df_porte['taxa_pct'] = df_porte['taxa'] * 100

ordem_porte = ['Microempresa', 'Pequeno Porte', 'Médio e Grande Porte']
df_porte = df_porte.reindex([p for p in ordem_porte if p in df_porte.index])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_porte.index,
        y=df_porte['taxa_pct'],
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c']),
        text=[f"{v:.1f}%" for v in df_porte['taxa_pct']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Taxa de Adoção do Instagram por Porte",
        xaxis_title="Porte",
        yaxis_title="Taxa de Adoção (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_porte.index,
        y=df_porte['media_posts'],
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c']),
        text=[f"{v:.0f}" for v in df_porte['media_posts']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Média de Posts por Porte",
        xaxis_title="Porte",
        yaxis_title="Média de Posts",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SEÇÃO 3: ANÁLISE POR SEÇÃO CNAE (MACRO)
# ==============================================================================

st.markdown('<p class="section-header">🏭 Análise por Seção CNAE (Visão Macro)</p>', unsafe_allow_html=True)

# Agrupar por seção CNAE
df_secao = df_filtered.groupby('CNAE_secao').agg({
    'tem_instagram': ['sum', 'count', 'mean'],
    'numero_posts': ['mean', 'median']
}).round(2)

df_secao.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts']
df_secao['taxa_pct'] = df_secao['taxa'] * 100
df_secao = df_secao[df_secao['total'] >= 5]
df_secao['nome_secao'] = df_secao.index.map(lambda x: secoes_cnae[x]['nome'] if x in secoes_cnae else 'Outros')
df_secao['divisoes'] = df_secao.index.map(lambda x: ', '.join(secoes_cnae[x]['divisoes']) if x in secoes_cnae else '')

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Seções por Total de Empresas")
    top_secoes = df_secao.nlargest(15, 'total').sort_values('total', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_secoes['total'],
        y=[f"{idx} ({row['divisoes']}) - {row['nome_secao'][:30]}" for idx, row in top_secoes.iterrows()],
        orientation='h',
        marker=dict(color='lightseagreen'),
        text=top_secoes['total'].astype(int),
        textposition='auto'
    ))
    fig.update_layout(
        xaxis_title="Número de Empresas",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Seções por Taxa de Adoção")
    top_taxa_secao = df_secao.nlargest(15, 'taxa_pct').sort_values('taxa_pct', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_taxa_secao['taxa_pct'],
        y=[f"{idx} ({row['divisoes']}) - {row['nome_secao'][:30]}" for idx, row in top_taxa_secao.iterrows()],
        orientation='h',
        marker=dict(color='steelblue'),
        text=[f"{v:.1f}%" for v in top_taxa_secao['taxa_pct']],
        textposition='auto'
    ))
    fig.update_layout(
        xaxis_title="Taxa de Adoção (%)",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SEÇÃO 4: ANÁLISE DETALHADA POR DIVISÃO E CNAEs ESPECÍFICOS (MICRO)
# ==============================================================================

st.markdown('<p class="section-header">🔬 Análise Detalhada por CNAEs Específicos (Visão Micro)</p>', unsafe_allow_html=True)

if secao_selecionada != 'Todas':
    secao_code = secao_selecionada.split(' ')[0]
    divisoes_secao = secoes_cnae[secao_code]['divisoes']

    st.info(f"**Seção {secao_code}:** {secoes_cnae[secao_code]['nome']}")
    st.markdown(f"**Divisões incluídas:** {', '.join(divisoes_secao)}")

    # Filtrar apenas divisões dessa seção
    df_divisao_filtered = df_filtered[df_filtered['CNAE_divisao'].isin(divisoes_secao)]

    if len(df_divisao_filtered) > 0:
        # Análise por divisão
        df_divisao = df_divisao_filtered.groupby('CNAE_divisao').agg({
            'tem_instagram': ['sum', 'count', 'mean'],
            'numero_posts': ['mean', 'median']
        }).round(2)

        df_divisao.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts']
        df_divisao['taxa_pct'] = df_divisao['taxa'] * 100
        df_divisao = df_divisao.sort_values('total', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Análise por Divisão (2 primeiros dígitos)")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_divisao.index,
                y=df_divisao['total'],
                marker=dict(color='coral'),
                text=df_divisao['total'].astype(int),
                textposition='auto',
                name='Total'
            ))
            fig.add_trace(go.Bar(
                x=df_divisao.index,
                y=df_divisao['com_insta'],
                marker=dict(color='lightgreen'),
                text=df_divisao['com_insta'].astype(int),
                textposition='auto',
                name='Com Instagram'
            ))
            fig.update_layout(
                xaxis_title="Divisão CNAE",
                yaxis_title="Número de Empresas",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_divisao.index,
                y=df_divisao['taxa_pct'],
                marker=dict(color='mediumpurple'),
                text=[f"{v:.1f}%" for v in df_divisao['taxa_pct']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Taxa de Adoção por Divisão",
                xaxis_title="Divisão CNAE",
                yaxis_title="Taxa de Adoção (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # CNAEs COMPLETOS específicos da seção selecionada
        st.markdown("#### CNAEs Completos nesta Seção")

        min_empresas_cnae = st.slider("Mínimo de empresas por CNAE:", 3, 20, 5, 1)

        # AGRUPAR POR CNAE COMPLETO SEM DUPLICATAS
        df_cnae_especifico = df_divisao_filtered[df_divisao_filtered['CNAE_completo'] != 'N/A'].groupby(
            ['CNAE_completo', 'cnae_descricao']
        ).agg({
            'tem_instagram': ['sum', 'count', 'mean'],
            'numero_posts': ['mean', 'median']
        }).round(2)

        df_cnae_especifico.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts']
        df_cnae_especifico['taxa_pct'] = df_cnae_especifico['taxa'] * 100
        df_cnae_especifico = df_cnae_especifico[df_cnae_especifico['total'] >= min_empresas_cnae]
        df_cnae_especifico = df_cnae_especifico.sort_values('total', ascending=False).head(30)
        df_cnae_especifico = df_cnae_especifico.reset_index()

        if len(df_cnae_especifico) > 0:
            col1, col2 = st.columns(2)

            with col1:
                top_cnaes_total = df_cnae_especifico.nlargest(15, 'total')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_cnaes_total['total'],
                    y=[f"{row['CNAE_completo']} - {row['cnae_descricao'][:35]}" 
                       for _, row in top_cnaes_total.iterrows()],
                    orientation='h',
                    marker=dict(color='teal'),
                    text=top_cnaes_total['total'].astype(int),
                    textposition='auto'
                ))
                fig.update_layout(
                    title=f"Top 15 CNAEs por Total (mín. {min_empresas_cnae} empresas)",
                    xaxis_title="Número de Empresas",
                    yaxis_title="",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                top_cnaes_taxa = df_cnae_especifico.nlargest(15, 'taxa_pct')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_cnaes_taxa['taxa_pct'],
                    y=[f"{row['CNAE_completo']} - {row['cnae_descricao'][:35]}" 
                       for _, row in top_cnaes_taxa.iterrows()],
                    orientation='h',
                    marker=dict(color='darkseagreen'),
                    text=[f"{v:.1f}%" for v in top_cnaes_taxa['taxa_pct']],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Top 15 CNAEs por Taxa de Adoção",
                    xaxis_title="Taxa de Adoção (%)",
                    yaxis_title="",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tabela detalhada
            st.markdown("#### Tabela Detalhada de CNAEs Completos")

            df_cnae_display = df_cnae_especifico[[
                'CNAE_completo', 'cnae_descricao', 'total', 'com_insta', 
                'taxa_pct', 'media_posts', 'mediana_posts'
            ]].copy()
            df_cnae_display.columns = ['CNAE', 'Descrição', 'Total', 'Com Instagram', 
                                       'Taxa (%)', 'Média Posts', 'Mediana Posts']

            st.dataframe(
                df_cnae_display.style.format({
                    'Total': '{:.0f}',
                    'Com Instagram': '{:.0f}',
                    'Taxa (%)': '{:.1f}',
                    'Média Posts': '{:.1f}',
                    'Mediana Posts': '{:.1f}'
                }).background_gradient(subset=['Taxa (%)'], cmap='RdYlGn'),
                use_container_width=True,
                height=500
            )
        else:
            st.warning(f"Nenhum CNAE com pelo menos {min_empresas_cnae} empresas nesta seção.")
    else:
        st.warning("Nenhuma empresa encontrada para esta seção com os filtros aplicados.")
else:
    st.info("**Selecione uma seção CNAE específica no filtro lateral** para visualizar a análise detalhada por CNAEs completos dessa seção.")

    # Análise geral de CNAEs mais frequentes
    st.markdown("#### CNAEs Completos Mais Frequentes (Todas as Seções)")

    min_empresas_geral = st.slider("Mínimo de empresas por CNAE:", 10, 50, 20, 5)

    # AGRUPAR SEM DUPLICATAS
    df_cnae_geral = df_filtered[df_filtered['CNAE_completo'] != 'N/A'].groupby(
        ['CNAE_completo', 'cnae_descricao', 'CNAE_secao']
    ).agg({
        'tem_instagram': ['sum', 'count', 'mean'],
        'numero_posts': ['mean', 'median']
    }).round(2)

    df_cnae_geral.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts']
    df_cnae_geral['taxa_pct'] = df_cnae_geral['taxa'] * 100
    df_cnae_geral = df_cnae_geral[df_cnae_geral['total'] >= min_empresas_geral]
    df_cnae_geral = df_cnae_geral.sort_values('total', ascending=False).head(20)
    df_cnae_geral = df_cnae_geral.reset_index()

    if len(df_cnae_geral) > 0:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_cnae_geral.head(15)['total'],
                y=[f"{row['CNAE_completo']} - {row['cnae_descricao'][:30]}" 
                   for _, row in df_cnae_geral.head(15).iterrows()],
                orientation='h',
                marker=dict(color='coral'),
                text=df_cnae_geral.head(15)['total'].astype(int),
                textposition='auto'
            ))
            fig.update_layout(
                title=f"Top 15 CNAEs (mín. {min_empresas_geral} empresas)",
                xaxis_title="Número de Empresas",
                yaxis_title="",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_taxa_geral = df_cnae_geral.nlargest(15, 'taxa_pct')

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_taxa_geral['taxa_pct'],
                y=[f"{row['CNAE_completo']} - {row['cnae_descricao'][:30]}" 
                   for _, row in top_taxa_geral.iterrows()],
                orientation='h',
                marker=dict(color='mediumpurple'),
                text=[f"{v:.1f}%" for v in top_taxa_geral['taxa_pct']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 15 CNAEs por Taxa de Adoção",
                xaxis_title="Taxa de Adoção (%)",
                yaxis_title="",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SEÇÃO 5: TOP EMPRESAS
# ==============================================================================

st.markdown('<p class="section-header">🏆 Top Empresas por Número de Posts</p>', unsafe_allow_html=True)

n_top = st.slider("Número de empresas a exibir:", 10, 50, 20, 5, key='top_empresas_slider')

df_top = df_filtered[df_filtered['tem_instagram']==1].nlargest(n_top, 'numero_posts')[
    ['CNPJ_COMPLETO', 'numero_posts', 'CNAE_secao', 'CNAE_divisao', 'CNAE_completo', 
     'cnae_descricao', 'CEP_regiao', 'porte_desc']
].copy()

df_top['Seção'] = df_top['CNAE_secao'].map(
    lambda x: f"{x} ({', '.join(secoes_cnae[x]['divisoes'])}) - {secoes_cnae[x]['nome'][:35]}" 
    if x in secoes_cnae else x
)
df_top['CNAE Completo'] = df_top['CNAE_completo'] + ' - ' + df_top['cnae_descricao']

df_top_display = df_top[[
    'CNPJ_COMPLETO', 'numero_posts', 'Seção', 'CNAE_divisao', 'CNAE Completo',
    'CEP_regiao', 'porte_desc'
]]
df_top_display.columns = ['CNPJ', 'Posts', 'Seção CNAE', 'Divisão', 'CNAE Completo', 'CEP Região', 'Porte']

st.dataframe(
    df_top_display.style.format({'Posts': '{:,.0f}'}).background_gradient(
        subset=['Posts'], cmap='RdYlGn'
    ),
    use_container_width=True,
    height=400
)

# ==============================================================================
# SEÇÃO 6: MAPA GEOGRÁFICO DOS MUNICÍPIOS DO RS
# ==============================================================================

st.markdown('<p class="section-header">🗺️ Mapa Geográfico - Municípios do Rio Grande do Sul</p>', unsafe_allow_html=True)

@st.cache_data
def load_geojson():
    """Carrega o arquivo GeoJSON dos municípios do RS"""
    import json
    with open('dados/municipios_rs.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Verificar se existe coluna nome_municipio
if 'nome_municipio' in df_filtered.columns:
    # Carregar GeoJSON
    try:
        geojson_municipios = load_geojson()

        # Normalizar nomes de municípios para garantir match
        # Remover acentos e converter para maiúsculas
        import unicodedata

        def normalizar_texto(texto):
            """Remove acentos e converte para maiúsculas"""
            if pd.isna(texto):
                return ''
            texto = str(texto).upper().strip()
            return ''.join(c for c in unicodedata.normalize('NFD', texto)
                          if unicodedata.category(c) != 'Mn')

        # Agregar dados por município
        df_municipio = df_filtered[df_filtered['nome_municipio'].notna()].groupby('nome_municipio').agg({
            'tem_instagram': ['sum', 'count', 'mean'],
            'numero_posts': ['mean', 'median', 'sum']
        }).round(2)

        df_municipio.columns = ['com_insta', 'total', 'taxa', 'media_posts', 'mediana_posts', 'total_posts']
        df_municipio['taxa_pct'] = df_municipio['taxa'] * 100
        df_municipio = df_municipio.reset_index()

        # Normalizar nomes para o merge
        df_municipio['municipio_norm'] = df_municipio['nome_municipio'].apply(normalizar_texto)

        # Criar mapeamento do GeoJSON
        for feature in geojson_municipios['features']:
            feature['properties']['municipio_norm'] = normalizar_texto(feature['properties']['name'])

        # Seletor de métrica a visualizar
        metrica_mapa = st.selectbox(
            "Selecione a métrica para visualizar no mapa:",
            options=[
                'total',
                'taxa_pct',
                'media_posts',
                'com_insta',
                'total_posts'
            ],
            format_func=lambda x: {
                'total': 'Total de Empresas',
                'taxa_pct': 'Taxa de Adoção do Instagram (%)',
                'media_posts': 'Média de Posts por Empresa',
                'com_insta': 'Empresas com Instagram',
                'total_posts': 'Total de Posts (soma)'
            }[x],
            index=1  # Padrão: Taxa de Adoção
        )

        # Criar DataFrame com TODOS os municípios do RS (do GeoJSON)
        # para garantir que todos apareçam no mapa, mesmo sem dados
        todos_municipios = pd.DataFrame([
            {
                'municipio_norm': normalizar_texto(feature['properties']['name']),
                'nome_municipio': feature['properties']['name']
            }
            for feature in geojson_municipios['features']
        ])

        # Fazer LEFT JOIN: todos os municípios + dados quando disponíveis
        df_municipio_mapa = todos_municipios.merge(
            df_municipio,
            on='municipio_norm',
            how='left',
            suffixes=('', '_data')
        )

        # Preencher valores NaN com 0 para municípios sem dados
        df_municipio_mapa['total'] = df_municipio_mapa['total'].fillna(0)
        df_municipio_mapa['com_insta'] = df_municipio_mapa['com_insta'].fillna(0)
        df_municipio_mapa['taxa_pct'] = df_municipio_mapa['taxa_pct'].fillna(0)
        df_municipio_mapa['media_posts'] = df_municipio_mapa['media_posts'].fillna(0)
        df_municipio_mapa['total_posts'] = df_municipio_mapa['total_posts'].fillna(0)

        # Usar o nome do GeoJSON quando não houver dados
        df_municipio_mapa['nome_municipio'] = df_municipio_mapa['nome_municipio_data'].fillna(df_municipio_mapa['nome_municipio'])
        df_municipio_mapa = df_municipio_mapa.drop('nome_municipio_data', axis=1, errors='ignore')

        # Criar o mapa coroplético
        if len(df_municipio_mapa) > 0:
            # Labels das métricas
            metrica_labels = {
                'total': 'Total de Empresas',
                'taxa_pct': 'Taxa de Adoção (%)',
                'media_posts': 'Média de Posts',
                'com_insta': 'Empresas com Instagram',
                'total_posts': 'Total de Posts'
            }

            # Escalas de cores por métrica (todas padronizadas em RdYlGn)
            color_scales = {
                'total': 'RdYlGn',
                'taxa_pct': 'RdYlGn',
                'media_posts': 'RdYlGn',
                'com_insta': 'RdYlGn',
                'total_posts': 'RdYlGn'
            }

            fig = px.choropleth(
                df_municipio_mapa,
                geojson=geojson_municipios,
                locations='municipio_norm',
                featureidkey='properties.municipio_norm',
                color=metrica_mapa,
                hover_name='nome_municipio',
                hover_data={
                    'municipio_norm': False,
                    'total': ':,',
                    'com_insta': ':,',
                    'taxa_pct': ':.1f',
                    'media_posts': ':.1f',
                    'total_posts': ':,',
                    metrica_mapa: False
                },
                labels={
                    'total': 'Total Empresas',
                    'com_insta': 'Com Instagram',
                    'taxa_pct': 'Taxa Adoção (%)',
                    'media_posts': 'Média Posts',
                    'total_posts': 'Total Posts'
                },
                color_continuous_scale=color_scales[metrica_mapa],
                title=f'Mapa do RS: {metrica_labels[metrica_mapa]} por Município'
            )

            fig.update_geos(
                fitbounds="locations",
                visible=False
            )

            fig.update_layout(
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_colorbar=dict(
                    title=metrica_labels[metrica_mapa],
                    thicknessmode="pixels",
                    thickness=15,
                    lenmode="pixels",
                    len=300
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ Arquivo 'municipios_rs.json' não encontrado. Certifique-se de que o arquivo está no mesmo diretório do aplicativo.")
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar o mapa: {str(e)}")
else:
    st.warning("⚠️ A coluna 'nome_municipio' não está disponível nos dados. Execute o script 'merge_municipio.py' primeiro para adicionar informações de município.")