"""Gráficos do projeto (Plotly), com uma identidade visual única.

Fundo claro, azuis como cores principais e uma cor de contraste só para
distinguir informações. A formatação fica aqui, fora dos notebooks.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

AZUL_ESCURO = "#0B3C7D"
AZUL = "#1F6FD1"
AZUL_MEDIO = "#4A90E2"
AZUL_CLARO = "#8DBDF0"
AZUL_PALIDO = "#DCEBFB"
CINZA = "#6B7280"
CINZA_CLARO = "#D1D5DB"
CONTRASTE = "#E07B39"   # usado só para separar duas informações (ex.: real × reconstruído)

CORES_METODOS = {
    "Mediana": CINZA,
    "KNNImputer": CONTRASTE,
    "IterativeImputer": AZUL_CLARO,
    "Autoencoder comum": AZUL_MEDIO,
    "Denoising autoencoder": AZUL_ESCURO,
}


def configurar_tema():
    tema = go.layout.Template()
    tema.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color="#1F2937"),
        title=dict(x=0.0, xanchor="left", font=dict(size=19, color="#0F172A")),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=[AZUL_ESCURO, CONTRASTE, AZUL_CLARO, CINZA, AZUL_MEDIO],
        xaxis=dict(showgrid=False, linecolor=CINZA_CLARO, ticks="outside", tickcolor=CINZA_CLARO,
                   zeroline=False, title=dict(standoff=10)),
        yaxis=dict(showgrid=True, gridcolor="#EEF2F7", linecolor=CINZA_CLARO, zeroline=False,
                   title=dict(standoff=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        margin=dict(l=70, r=30, t=90, b=60),
        hoverlabel=dict(font_size=14),
    )
    pio.templates["sensores"] = tema
    pio.templates.default = "sensores"
    # Saída leve no notebook: o VS Code e o Jupyter desenham a partir dos dados do gráfico.
    pio.renderers.default = "plotly_mimetype+notebook_connected"


configurar_tema()


# ================================================================ Distribuições e relações entre sensores
def distribuicao_original_e_log(serie, nome, unidade):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=(f"{nome} em {unidade}", f"log1p({nome})"))
    fig.add_trace(go.Histogram(x=serie, nbinsx=80, marker_color=AZUL, showlegend=False), 1, 1)
    fig.add_trace(go.Histogram(x=np.log1p(serie), nbinsx=80, marker_color=AZUL_ESCURO, showlegend=False), 1, 2)
    fig.update_yaxes(title_text="Registros", row=1, col=1)
    fig.update_xaxes(title_text=unidade, row=1, col=1)
    fig.update_xaxes(title_text=f"log1p({unidade})", row=1, col=2)
    fig.update_layout(title=f"Distribuição de {nome} no treino (2011–2012)", height=420, bargap=0.02)
    return fig


def mapa_correlacao(df, titulo="Correlação entre as leituras no treino (2011–2012)"):
    corr = df.corr().round(2)
    escala = [[0.0, CONTRASTE], [0.5, "#FFFFFF"], [1.0, AZUL_ESCURO]]
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=escala, text=corr.to_numpy(), texttemplate="%{text:.2f}",
                               textfont=dict(size=12), colorbar=dict(title="r")))
    fig.update_layout(title=titulo, height=600, width=720, yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def dispersao_pares(df, pares, unidades, n_pontos=3000, semente=0):
    amostra = df.sample(min(n_pontos, len(df)), random_state=semente)
    titulos = [f"{y} × {x}  (r = {df[x].corr(df[y]):.2f})" for x, y in pares]
    fig = make_subplots(rows=1, cols=len(pares), subplot_titles=titulos, horizontal_spacing=0.1)
    for i, (x, y) in enumerate(pares, start=1):
        fig.add_trace(go.Scattergl(x=amostra[x], y=amostra[y], mode="markers", showlegend=False,
                                   marker=dict(size=4, color=AZUL, opacity=0.35)), 1, i)
        fig.update_xaxes(title_text=f"{x} ({unidades[x]})", row=1, col=i)
        fig.update_yaxes(title_text=f"{y} ({unidades[y]})", row=1, col=i)
    fig.update_layout(title="Leituras relacionadas e leituras quase independentes (amostra do treino)",
                      height=430)
    return fig


def barras_agrupadas(df, x, y, grupo, titulo, eixo_y, erro=None, cores=None, horizontal=False,
                     eixo_x=None):
    fig = go.Figure()
    padrao = [AZUL_MEDIO, AZUL_ESCURO, CONTRASTE, AZUL_CLARO, CINZA]
    for i, (nome, parte) in enumerate(df.groupby(grupo, sort=False)):
        cor = (cores or {}).get(nome, padrao[i % len(padrao)])
        barras = dict(name=str(nome), marker_color=cor)
        if erro is not None:
            barras["error_x" if horizontal else "error_y"] = dict(type="data", array=parte[erro])
        if horizontal:
            fig.add_trace(go.Bar(y=parte[x].astype(str), x=parte[y], orientation="h", **barras))
        else:
            fig.add_trace(go.Bar(x=parte[x].astype(str), y=parte[y], **barras))
    fig.update_layout(title=titulo, barmode="group", height=460)
    if horizontal:
        fig.update_layout(xaxis_title=eixo_y, yaxis_title=eixo_x or "", height=560,
                          yaxis=dict(autorange="reversed", showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor="#EEF2F7"))
        fig.update_xaxes(rangemode="tozero")
    else:
        fig.update_layout(yaxis_title=eixo_y, xaxis_title=eixo_x or "")
        fig.update_yaxes(rangemode="tozero")
    return fig
