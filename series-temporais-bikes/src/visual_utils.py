import plotly.express as px
import plotly.graph_objects as go

CORES = {
    "azul_escuro": "#0B1F4D",
    "azul_medio": "#123C8C",
    "azul_principal": "#1D5BFF",
    "azul_claro": "#6FA8FF",
    "azul_muito_claro": "#EAF2FF",
    "cinza_texto": "#2B2B2B",
    "cinza_suave": "#6B7280",
    "branco": "#FFFFFF",
    "fundo_claro": "#F7FAFF"
}

SEQUENCIA_CORES = [
    CORES["azul_principal"],
    CORES["azul_medio"],
    CORES["azul_claro"],
    "#4C78A8",
    "#7DA6FF",
    "#9EC5FF"
]


def aplicar_layout_padrao(
    fig,
    titulo=None,
    altura=500,
    mostrar_grade=True,
    legenda_horizontal=True
):
    """
    Aplica um layout visual padrão para os gráficos Plotly do projeto.
    """

    fig.update_layout(
        title=titulo,
        title_x=0.02,
        height=altura,
        template="plotly_white",
        paper_bgcolor=CORES["branco"],
        plot_bgcolor=CORES["fundo_claro"],
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color=CORES["cinza_texto"]
        ),
        colorway=SEQUENCIA_CORES,
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified"
    )

    if legenda_horizontal:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

    fig.update_xaxes(
        showgrid=mostrar_grade,
        gridcolor=CORES["azul_muito_claro"],
        linecolor=CORES["azul_claro"],
        tickfont=dict(color=CORES["cinza_texto"]),
        title_font=dict(color=CORES["cinza_texto"])
    )

    fig.update_yaxes(
        showgrid=mostrar_grade,
        gridcolor=CORES["azul_muito_claro"],
        linecolor=CORES["azul_claro"],
        tickfont=dict(color=CORES["cinza_texto"]),
        title_font=dict(color=CORES["cinza_texto"])
    )

    return fig


def grafico_linha_padrao(
    df,
    x,
    y,
    titulo,
    cor=None,
    labels=None,
    altura=500,
    markers=False
):
    """
    Cria um gráfico de linha com layout padrão do projeto.
    """

    fig = px.line(
        df,
        x=x,
        y=y,
        labels=labels,
        markers=markers
    )

    fig.update_traces(
        line=dict(
            color=cor or CORES["azul_principal"],
            width=3
        )
    )

    fig = aplicar_layout_padrao(
        fig,
        titulo=titulo,
        altura=altura
    )

    return fig


def grafico_barra_padrao(
    df,
    x,
    y,
    titulo,
    cor=None,
    labels=None,
    altura=500
):
    """
    Cria um gráfico de barras com layout padrão do projeto.
    """

    fig = px.bar(
        df,
        x=x,
        y=y,
        labels=labels
    )

    fig.update_traces(
        marker_color=cor or CORES["azul_principal"]
    )

    fig = aplicar_layout_padrao(
        fig,
        titulo=titulo,
        altura=altura
    )

    return fig


def grafico_linha_multiplas_series_padrao(
    df,
    x,
    y,
    color,
    titulo,
    labels=None,
    altura=500
):
    """
    Cria um gráfico de linha com múltiplas séries usando a paleta do projeto.
    """

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        labels=labels,
        color_discrete_sequence=SEQUENCIA_CORES
    )

    fig.update_traces(line=dict(width=3))

    fig = aplicar_layout_padrao(
        fig,
        titulo=titulo,
        altura=altura
    )

    return fig