from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.visual_utils import (
    aplicar_layout_padrao,
    grafico_barra_padrao,
    grafico_linha_padrao
)



# 1. CONFIGURAÇÃO DA PÁGINA


st.set_page_config(
    page_title="Previsão de demanda de bicicletas",
    page_icon="🚲",
    layout="wide"
)



# 2. CAMINHOS DO PROJETO


RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHO_TABELAS = RAIZ_PROJETO / "outputs" / "tabelas"



# 3. FUNÇÕES AUXILIARES


def validar_colunas(
    dataframe: pd.DataFrame,
    colunas_obrigatorias: list[str],
    nome_arquivo: str
) -> None:
    """
    Verifica se todas as colunas necessárias existem.
    """

    colunas_ausentes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in dataframe.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"O arquivo {nome_arquivo} não possui as colunas: "
            + ", ".join(colunas_ausentes)
        )


@st.cache_data
def carregar_dados():
    """
    Carrega e valida os arquivos produzidos pelos notebooks.
    """

    caminhos = {
        "serie_historica.csv": (
            CAMINHO_TABELAS / "serie_historica.csv"
        ),
        "metricas.csv": (
            CAMINHO_TABELAS / "metricas.csv"
        ),
        "previsoes.csv": (
            CAMINHO_TABELAS / "previsoes.csv"
        ),
        "resumo_projeto.csv": (
            CAMINHO_TABELAS / "resumo_projeto.csv"
        )
    }

    arquivos_ausentes = [
        nome
        for nome, caminho in caminhos.items()
        if not caminho.exists()
    ]

    if arquivos_ausentes:
        raise FileNotFoundError(
            "Arquivos não encontrados: "
            + ", ".join(arquivos_ausentes)
        )

    serie_historica = pd.read_csv(
        caminhos["serie_historica.csv"],
        parse_dates=["data_hora"]
    )

    metricas = pd.read_csv(
        caminhos["metricas.csv"]
    )

    previsoes = pd.read_csv(
        caminhos["previsoes.csv"],
        parse_dates=["data_hora"]
    )

    resumo_projeto = pd.read_csv(
        caminhos["resumo_projeto.csv"]
    )

    validar_colunas(
        dataframe=serie_historica,
        colunas_obrigatorias=[
            "data_hora",
            "demanda"
        ],
        nome_arquivo="serie_historica.csv"
    )

    validar_colunas(
        dataframe=metricas,
        colunas_obrigatorias=[
            "modelo",
            "MAE",
            "RMSE",
            "MAPE"
        ],
        nome_arquivo="metricas.csv"
    )

    validar_colunas(
        dataframe=previsoes,
        colunas_obrigatorias=[
            "data_hora",
            "demanda_real",
            "previsao_baseline",
            "previsao_arima",
            "previsao_sarima",
            "previsao_sarimax"
        ],
        nome_arquivo="previsoes.csv"
    )

    validar_colunas(
        dataframe=resumo_projeto,
        colunas_obrigatorias=[
            "modelo_final",
            "mae",
            "rmse",
            "mape",
            "inicio_historico",
            "fim_historico",
            "dias_historico",
            "dias_teste"
        ],
        nome_arquivo="resumo_projeto.csv"
    )

    if resumo_projeto.empty:
        raise ValueError(
            "O arquivo resumo_projeto.csv está vazio."
        )

    return (
        serie_historica,
        metricas,
        previsoes,
        resumo_projeto
    )


def criar_grafico_previsoes(
    previsoes: pd.DataFrame
) -> go.Figure:
    """
    Cria o gráfico com os valores reais e as previsões.
    """

    configuracao_linhas = [
        {
            "coluna": "demanda_real",
            "nome": "Real",
            "cor": "#0B1F4D",
            "largura": 3,
            "tracejado": None
        },
        {
            "coluna": "previsao_baseline",
            "nome": "Baseline",
            "cor": "#7A7A7A",
            "largura": 2,
            "tracejado": "dash"
        },
        {
            "coluna": "previsao_arima",
            "nome": "ARIMA",
            "cor": "#6FA8FF",
            "largura": 2,
            "tracejado": None
        },
        {
            "coluna": "previsao_sarima",
            "nome": "SARIMA",
            "cor": "#F28E2B",
            "largura": 2,
            "tracejado": None
        },
        {
            "coluna": "previsao_sarimax",
            "nome": "SARIMAX",
            "cor": "#2CA02C",
            "largura": 3,
            "tracejado": None
        }
    ]

    figura = go.Figure()

    for configuracao in configuracao_linhas:
        figura.add_trace(
            go.Scatter(
                x=previsoes["data_hora"],
                y=previsoes[
                    configuracao["coluna"]
                ],
                mode="lines",
                name=configuracao["nome"],
                line=dict(
                    color=configuracao["cor"],
                    width=configuracao["largura"],
                    dash=configuracao["tracejado"]
                )
            )
        )

    figura = aplicar_layout_padrao(
        figura,
        titulo=(
            "Comparação das previsões "
            "no período de teste"
        ),
        altura=520
    )

    figura.update_xaxes(
        title="Data"
    )

    figura.update_yaxes(
        title="Demanda diária"
    )

    return figura


def formatar_data(valor) -> str:
    """
    Converte a data para o formato dia/mês/ano.
    """

    return pd.to_datetime(valor).strftime(
        "%d/%m/%Y"
    )



# 4. CARREGAMENTO DOS DADOS


try:
    (
        serie_historica,
        metricas,
        previsoes,
        resumo_projeto
    ) = carregar_dados()

except (
    FileNotFoundError,
    ValueError,
    pd.errors.EmptyDataError
) as erro:
    st.error(
        f"Não foi possível carregar os dados: {erro}"
    )
    st.stop()


resumo = resumo_projeto.iloc[0]



# 5. CABEÇALHO DA APLICAÇÃO


st.title(
    "Previsão de demanda de bicicletas"
)

st.write(
    "Análise da demanda diária e comparação "
    "entre baseline, ARIMA, SARIMA e SARIMAX."
)

st.caption(
    "Projeto desenvolvido com Python, "
    "pandas, statsmodels, Plotly e Streamlit."
)



# 6. INDICADORES PRINCIPAIS


coluna_1, coluna_2, coluna_3, coluna_4 = (
    st.columns(4)
)

coluna_1.metric(
    label="Modelo final",
    value=resumo["modelo_final"]
)

coluna_2.metric(
    label="MAPE do modelo final",
    value=f"{resumo['mape']:.2f}%"
)

coluna_3.metric(
    label="Dias no histórico",
    value=int(resumo["dias_historico"])
)

coluna_4.metric(
    label="Dias no teste",
    value=int(resumo["dias_teste"])
)

inicio_historico = formatar_data(
    resumo["inicio_historico"]
)

fim_historico = formatar_data(
    resumo["fim_historico"]
)

st.caption(
    f"Período analisado: "
    f"{inicio_historico} a {fim_historico}."
)



# 7. SÉRIE HISTÓRICA


st.divider()

st.subheader(
    "Comportamento histórico da demanda"
)

st.write(
    "A série apresenta a quantidade diária "
    "de bicicletas alugadas ao longo do tempo."
)

figura_historico = grafico_linha_padrao(
    df=serie_historica,
    x="data_hora",
    y="demanda",
    titulo=(
        "Série histórica da demanda "
        "de bicicletas"
    ),
    labels={
        "data_hora": "Data",
        "demanda": "Demanda diária"
    },
    altura=500
)

st.plotly_chart(
    figura_historico,
    width="stretch"
)



# 8. COMPARAÇÃO DOS MODELOS


st.divider()

st.subheader(
    "Comparação dos modelos"
)

st.write(
    "Quanto menor o MAPE, menor foi o erro "
    "percentual médio no período de teste."
)

metricas_ordenadas = (
    metricas
    .sort_values("MAPE")
    .reset_index(drop=True)
)

figura_metricas = grafico_barra_padrao(
    df=metricas_ordenadas,
    x="modelo",
    y="MAPE",
    titulo="MAPE por modelo",
    labels={
        "modelo": "Modelo",
        "MAPE": "MAPE (%)"
    },
    altura=480
)

figura_metricas.update_traces(
    texttemplate="%{y:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    figura_metricas,
    width="stretch"
)

metricas_exibicao = (
    metricas_ordenadas
    .copy()
    .round(
        {
            "MAE": 2,
            "RMSE": 2,
            "MAPE": 2
        }
    )
)

st.dataframe(
    metricas_exibicao,
    hide_index=True,
    width="stretch"
)



# 9. PREVISÕES NO PERÍODO DE TESTE


st.divider()

st.subheader(
    "Previsões no período de teste"
)

st.write(
    "O gráfico compara a demanda real com "
    "as previsões produzidas por cada abordagem."
)

figura_previsoes = criar_grafico_previsoes(
    previsoes
)

st.plotly_chart(
    figura_previsoes,
    width="stretch"
)



# 10. OBSERVAÇÃO SOBRE O USO DO MODELO


st.info(
    "As previsões apresentadas correspondem "
    "ao período reservado para teste. Para prever "
    "novos dias com o SARIMAX, também precisamos "
    "de valores futuros para as variáveis externas, "
    "como clima e calendário."
)



# 11. INFORMAÇÕES COMPLEMENTARES


with st.expander(
    "Como adaptar esta aplicação"
):
    st.markdown(
        """
        Algumas possibilidades de adaptação:

        - trocar os arquivos usados como fonte;
        - adicionar ou remover modelos;
        - alterar cores, títulos e textos;
        - incluir novas métricas;
        - criar filtros por período;
        - utilizar dados de outro problema;
        - publicar a aplicação em um serviço de hospedagem.
        """
    )