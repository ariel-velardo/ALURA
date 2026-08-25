"""Camada visual do curso: paleta, layout padrão e gráficos em Plotly."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


CORES = {
    "azul_escuro": "#0B1F4D",
    "azul_medio": "#123C8C",
    "azul_principal": "#1D5BFF",
    "azul_claro": "#6FA8FF",
    "azul_muito_claro": "#EAF2FF",
    "verde": "#00C86F",
    "roxo": "#A259FF",
    "cinza_texto": "#2B2B2B",
    "cinza_suave": "#6B7280",
    "branco": "#FFFFFF",
    "fundo_claro": "#F7FAFF",
}

SEQUENCIA_CORES = [
    CORES["azul_principal"],
    CORES["verde"],
    CORES["roxo"],
    CORES["azul_claro"],
    CORES["azul_medio"],
]

ESCALA_AZUL = [
    CORES["branco"],
    CORES["azul_muito_claro"],
    CORES["azul_claro"],
    CORES["azul_principal"],
    CORES["azul_escuro"],
]

CORES_METRICAS = {
    "Precision": CORES["azul_principal"],
    "Recall": CORES["verde"],
    "F1": CORES["roxo"],
}


def aplicar_layout_padrao(
    fig,
    titulo=None,
    altura=460,
    mostrar_legenda=False,
    mostrar_grade=True,
):
    """Aplica o mesmo padrão visual a todos os gráficos do curso."""
    fig.update_layout(
        title=titulo,
        title_x=0.02,
        title_font=dict(size=18, color=CORES["azul_escuro"]),
        height=altura,
        template="plotly_white",
        paper_bgcolor=CORES["branco"],
        plot_bgcolor=CORES["fundo_claro"],
        font=dict(family="Arial, sans-serif", size=14, color=CORES["cinza_texto"]),
        colorway=SEQUENCIA_CORES,
        separators=",.",
        margin=dict(l=70, r=40, t=80, b=60),
        showlegend=mostrar_legenda,
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_size=13),
    )

    eixo = dict(
        showgrid=mostrar_grade,
        gridcolor=CORES["azul_muito_claro"],
        linecolor=CORES["azul_claro"],
        zeroline=False,
        tickfont=dict(color=CORES["cinza_texto"]),
        title_font=dict(color=CORES["cinza_texto"]),
    )
    fig.update_xaxes(**eixo)
    fig.update_yaxes(**eixo)
    return fig


def grafico_barras_padrao(
    dados,
    x,
    y,
    titulo,
    cor=None,
    labels=None,
    texto=None,
    altura=460,
):
    """Gráfico de barras simples, para situações diretas de exploração da base."""
    fig = px.bar(dados, x=x, y=y, labels=labels, text=texto)
    fig.update_traces(marker_color=cor or CORES["azul_principal"])
    if texto is not None:
        fig.update_traces(textposition="outside")
    return aplicar_layout_padrao(fig, titulo=titulo, altura=altura)


def grafico_boxplot_padrao(
    dados,
    y,
    x=None,
    titulo=None,
    cor=None,
    labels=None,
    altura=460,
):
    """Boxplot simples, disponível para investigações pontuais da base."""
    fig = px.box(dados, x=x, y=y, labels=labels)
    fig.update_traces(
        marker_color=cor or CORES["azul_principal"],
        line=dict(color=CORES["azul_medio"]),
    )
    return aplicar_layout_padrao(fig, titulo=titulo, altura=altura)


def grafico_distribuicao_alvo(tabela, titulo="Distribuição da inadimplência"):
    """Mostra quantos clientes há em cada classe do alvo, em quantidade e proporção."""
    fig = px.bar(
        x=["Não inadimplente", "Inadimplente"],
        y=tabela["clientes"],
        labels={"x": "Classe", "y": "Clientes"},
    )
    fig.update_traces(
        marker_color=[CORES["azul_principal"], CORES["roxo"]],
        customdata=tabela["proporcao"],
        texttemplate="%{y:,.0f} (%{customdata:.1%})",
        textposition="outside",
        hovertemplate="%{x}: %{y:,.0f} clientes<extra></extra>",
    )
    fig.update_yaxes(range=[0, tabela["clientes"].max() * 1.18])
    return aplicar_layout_padrao(fig, titulo=titulo)


def grafico_comparacao_modelos(
    resultados,
    titulo="PR-AUC dos modelos de referência",
    coluna_metrica="pr_auc",
):
    """Compara os modelos pela AP/PR-AUC, com eixo de 0 a 1 para não exagerar diferenças."""
    tabela = resultados.sort_values(coluna_metrica, ascending=False)
    fig = px.bar(
        tabela,
        x="modelo",
        y=coluna_metrica,
        labels={"modelo": "Modelo", coluna_metrica: "AP / PR-AUC"},
    )
    fig.update_traces(
        marker_color=SEQUENCIA_CORES[: len(tabela)],
        texttemplate="%{y:.3f}",
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    )
    fig.update_yaxes(range=[0, 1])
    return aplicar_layout_padrao(fig, titulo=titulo)


def grafico_historico_optuna(historico, titulo="Histórico dos trials"):
    """Mostra o valor de cada trial e o melhor resultado acumulado da busca."""
    fig = go.Figure()
    fig.add_scatter(
        x=historico["number"],
        y=historico["value"],
        mode="markers",
        name="Trial",
        marker=dict(color=CORES["azul_claro"], size=11),
        hovertemplate="Trial %{x}: %{y:.4f}<extra></extra>",
    )
    fig.add_scatter(
        x=historico["number"],
        y=historico["melhor_acumulado"],
        mode="lines",
        name="Melhor acumulado",
        line=dict(color=CORES["azul_principal"], width=3),
        hovertemplate="Trial %{x}: %{y:.4f}<extra></extra>",
    )
    fig.update_xaxes(title_text="Trial")
    fig.update_yaxes(title_text="AP / PR-AUC")
    return aplicar_layout_padrao(fig, titulo=titulo, mostrar_legenda=True)


def grafico_importancia_hiperparametros(
    importancias,
    titulo="Importância dos hiperparâmetros",
):
    """Mostra a importância relativa de cada hiperparâmetro calculada pelo Optuna."""
    tabela = pd.DataFrame(
        {
            "hiperparametro": list(importancias),
            "importancia": list(importancias.values()),
        }
    ).sort_values("importancia")
    fig = px.bar(
        tabela,
        x="importancia",
        y="hiperparametro",
        orientation="h",
        labels={"importancia": "Importância", "hiperparametro": ""},
    )
    fig.update_traces(
        marker_color=CORES["azul_principal"],
        texttemplate="%{x:.2f}",
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )
    return aplicar_layout_padrao(fig, titulo=titulo, altura=400)


def grafico_metricas_por_limiar(
    resultados_limiares,
    titulo="Precision, Recall e F1 por limiar",
):
    """Compara Precision, Recall e F1 nos limiares avaliados no projeto."""
    nomes = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    tabela = resultados_limiares.melt(
        id_vars="limiar",
        value_vars=list(nomes),
        var_name="metrica",
        value_name="valor",
    )
    tabela["metrica"] = tabela["metrica"].map(nomes)
    fig = px.line(
        tabela,
        x="limiar",
        y="valor",
        color="metrica",
        markers=True,
        labels={"limiar": "Limiar", "valor": "Métrica", "metrica": ""},
        color_discrete_map=CORES_METRICAS,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=9))
    fig.update_xaxes(autorange="reversed", tickvals=resultados_limiares["limiar"])
    fig.update_yaxes(range=[0, 1])
    fig = aplicar_layout_padrao(fig, titulo=titulo, mostrar_legenda=True)
    fig.update_layout(hovermode="x unified")
    return fig


def grafico_importancia_variaveis(
    importancias,
    coluna_nome="variavel",
    coluna_valor="importancia",
    titulo="Importância das variáveis",
    top_n=12,
    coluna_desvio=None,
):
    """Mostra as variáveis mais importantes; aceita desvio para Permutation Importance."""
    tabela = importancias.nlargest(top_n, coluna_valor).sort_values(coluna_valor)
    fig = px.bar(
        tabela,
        x=coluna_valor,
        y=coluna_nome,
        orientation="h",
        error_x=coluna_desvio,
        labels={coluna_valor: "Importância", coluna_nome: "Variável"},
    )
    fig.update_traces(
        marker_color=CORES["azul_principal"],
        error_x_color=CORES["cinza_suave"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    )
    return aplicar_layout_padrao(fig, titulo=titulo, altura=500)


def grafico_matriz_confusao(
    y_verdadeiro,
    previsoes,
    titulo="Matriz de confusão no teste",
):
    """Mostra a matriz de confusão com os valores absolutos em cada célula."""
    matriz = confusion_matrix(y_verdadeiro, previsoes, labels=[0, 1])
    rotulos = ["Não inadimplente", "Inadimplente"]
    fig = px.imshow(
        matriz,
        x=rotulos,
        y=rotulos,
        text_auto=True,
        color_continuous_scale=ESCALA_AZUL,
        labels=dict(x="Classe prevista", y="Classe real", color="Clientes"),
    )
    fig.update_traces(
        textfont=dict(size=18),
        hovertemplate="Real: %{y}<br>Prevista: %{x}<br>Clientes: %{z}<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False)
    return aplicar_layout_padrao(fig, titulo=titulo, altura=420, mostrar_grade=False)
