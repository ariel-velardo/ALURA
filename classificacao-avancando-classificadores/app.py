import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Classificação de inadimplência",
    page_icon="📊",
    layout="centered",
)


PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_PROJETO / "models" / "modelo_inadimplencia.joblib"
CAMINHO_METRICAS = PASTA_PROJETO / "models" / "metricas_modelo.json"

GRUPOS_CAMPOS = {
    "Perfil": [
        "limite_credito",
        "sexo",
        "escolaridade",
        "estado_civil",
        "idade",
    ],
    "Histórico de pagamento": [
        "status_pagamento_set",
        "status_pagamento_ago",
        "status_pagamento_jul",
        "status_pagamento_jun",
        "status_pagamento_mai",
        "status_pagamento_abr",
    ],
    "Faturas": [
        "valor_fatura_set",
        "valor_fatura_ago",
        "valor_fatura_jul",
        "valor_fatura_jun",
        "valor_fatura_mai",
        "valor_fatura_abr",
    ],
    "Pagamentos": [
        "valor_pago_set",
        "valor_pago_ago",
        "valor_pago_jul",
        "valor_pago_jun",
        "valor_pago_mai",
        "valor_pago_abr",
    ],
}

ROTULOS = {
    "limite_credito": "Limite de crédito",
    "sexo": "Sexo (código)",
    "escolaridade": "Escolaridade (código)",
    "estado_civil": "Estado civil (código)",
    "idade": "Idade",
    "status_pagamento_set": "Status de pagamento — setembro",
    "status_pagamento_ago": "Status de pagamento — agosto",
    "status_pagamento_jul": "Status de pagamento — julho",
    "status_pagamento_jun": "Status de pagamento — junho",
    "status_pagamento_mai": "Status de pagamento — maio",
    "status_pagamento_abr": "Status de pagamento — abril",
    "valor_fatura_set": "Valor da fatura — setembro",
    "valor_fatura_ago": "Valor da fatura — agosto",
    "valor_fatura_jul": "Valor da fatura — julho",
    "valor_fatura_jun": "Valor da fatura — junho",
    "valor_fatura_mai": "Valor da fatura — maio",
    "valor_fatura_abr": "Valor da fatura — abril",
    "valor_pago_set": "Valor pago — setembro",
    "valor_pago_ago": "Valor pago — agosto",
    "valor_pago_jul": "Valor pago — julho",
    "valor_pago_jun": "Valor pago — junho",
    "valor_pago_mai": "Valor pago — maio",
    "valor_pago_abr": "Valor pago — abril",
}

CAMPOS_CATEGORICOS = {"sexo", "escolaridade", "estado_civil"}
CAMPOS_STATUS = set(GRUPOS_CAMPOS["Histórico de pagamento"])
CAMPOS_MONETARIOS = {
    "limite_credito",
    *GRUPOS_CAMPOS["Faturas"],
    *GRUPOS_CAMPOS["Pagamentos"],
}


@st.cache_resource
def carregar_modelo(caminho: str) -> Mapping[str, Any]:
    payload = joblib.load(caminho)
    if not isinstance(payload, Mapping):
        raise TypeError("o artefato do modelo não contém um dicionário")

    chaves = {
        "pipeline",
        "limiar",
        "colunas_features",
        "alvo",
        "valores_padrao",
        "opcoes_categoricas",
        "versoes",
    }
    ausentes = sorted(chaves.difference(payload))
    if ausentes:
        raise KeyError(f"chaves ausentes no artefato: {', '.join(ausentes)}")
    return payload


@st.cache_data
def carregar_metricas(caminho: str) -> Mapping[str, Any]:
    with open(caminho, encoding="utf-8") as arquivo:
        metricas = json.load(arquivo)

    if not isinstance(metricas, Mapping):
        raise TypeError("o arquivo de métricas não contém um objeto JSON")

    chaves = {
        "modelo",
        "limiar",
        "validacao",
        "teste",
        "matriz_confusao_teste",
        "observacao",
    }
    ausentes = sorted(chaves.difference(metricas))
    if ausentes:
        raise KeyError(f"chaves ausentes no arquivo de métricas: {', '.join(ausentes)}")
    return metricas


def carregar_artefato(caminho: Path, carregador: Any, nome: str) -> tuple[Any, str | None]:
    if not caminho.is_file():
        return None, (
            f"O artefato de {nome} não foi encontrado em `{caminho}`. "
            "Execute o notebook 06_modelo_final.ipynb para gerá-lo."
        )

    try:
        return carregador(str(caminho)), None
    except Exception as erro:
        return None, f"Não foi possível carregar o artefato de {nome}: {erro}"


def valor_python(valor: Any) -> Any:
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return [valor_python(item) for item in valor.tolist()]
    return valor


def obter(mapeamento: Mapping[str, Any], *chaves: str) -> Any:
    chaves_normalizadas = {str(chave).lower(): chave for chave in mapeamento}
    for chave in chaves:
        if chave in mapeamento:
            return mapeamento[chave]
        chave_real = chaves_normalizadas.get(chave.lower())
        if chave_real is not None:
            return mapeamento[chave_real]
    return None


def numero(valor: Any) -> float:
    valor = valor_python(valor)
    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")
    resultado = float(valor)
    if not np.isfinite(resultado):
        raise ValueError(f"valor numérico inválido: {valor}")
    return resultado


def formatar_decimal(valor: Any, casas: int = 3) -> str:
    try:
        return f"{numero(valor):.{casas}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def lista_opcoes(valor: Any) -> list[Any]:
    if isinstance(valor, Mapping):
        itens = list(valor.keys())
    elif isinstance(valor, np.ndarray):
        itens = valor.tolist()
    elif isinstance(valor, Sequence) and not isinstance(valor, (str, bytes)):
        itens = list(valor)
    else:
        itens = [valor]

    opcoes: list[Any] = []
    for item in itens:
        item = valor_python(item)
        if not any(item == existente for existente in opcoes):
            opcoes.append(item)
    return opcoes


def indice_padrao(opcoes: list[Any], padrao: Any) -> int:
    padrao = valor_python(padrao)
    for indice, opcao in enumerate(opcoes):
        try:
            if bool(opcao == padrao):
                return indice
        except (TypeError, ValueError):
            continue
    return 0


def rotulo_opcao(valor: Any) -> str:
    valor = valor_python(valor)
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor)


def validar_campos(payload: Mapping[str, Any]) -> list[str]:
    colunas = [str(coluna) for coluna in payload["colunas_features"]]
    esperadas = [campo for campos in GRUPOS_CAMPOS.values() for campo in campos]
    if len(colunas) != 23 or set(colunas) != set(esperadas):
        raise ValueError(
            "o artefato deve conter as 23 features esperadas em português"
        )

    valores_padrao = payload["valores_padrao"]
    opcoes = payload["opcoes_categoricas"]
    if not isinstance(valores_padrao, Mapping) or not isinstance(opcoes, Mapping):
        raise TypeError("valores_padrao e opcoes_categoricas devem ser dicionários")

    sem_padrao = sorted(set(colunas).difference(valores_padrao))
    sem_opcoes = sorted((CAMPOS_CATEGORICOS | CAMPOS_STATUS).difference(opcoes))
    if sem_padrao:
        raise KeyError(f"valores padrão ausentes para: {', '.join(sem_padrao)}")
    if sem_opcoes:
        raise KeyError(f"opções ausentes para: {', '.join(sem_opcoes)}")
    return colunas


def campo_do_formulario(
    campo: str,
    valores_padrao: Mapping[str, Any],
    opcoes_categoricas: Mapping[str, Any],
) -> Any:
    rotulo = ROTULOS[campo]
    padrao = valor_python(valores_padrao[campo])

    if campo in CAMPOS_CATEGORICOS | CAMPOS_STATUS:
        opcoes = lista_opcoes(opcoes_categoricas[campo])
        if not opcoes:
            raise ValueError(f"não há opções cadastradas para {campo}")
        return st.selectbox(
            rotulo,
            options=opcoes,
            index=indice_padrao(opcoes, padrao),
            format_func=rotulo_opcao,
            key=f"entrada_{campo}",
        )

    if campo == "idade":
        return st.number_input(
            rotulo,
            value=int(numero(padrao)),
            step=1,
            key=f"entrada_{campo}",
        )

    formato = "%.2f" if campo in CAMPOS_MONETARIOS else "%.0f"
    return st.number_input(
        rotulo,
        value=numero(padrao),
        step=100.0 if campo in CAMPOS_MONETARIOS else 1.0,
        format=formato,
        key=f"entrada_{campo}",
    )


def dataframe_matriz(metricas: Mapping[str, Any]) -> pd.DataFrame:
    teste = metricas["teste"]
    matriz = metricas.get("matriz_confusao_teste")
    if matriz is None and isinstance(teste, Mapping):
        matriz = [
            [obter(teste, "vn", "tn"), obter(teste, "fp")],
            [obter(teste, "fn"), obter(teste, "vp", "tp")],
        ]

    valores = np.asarray(matriz, dtype=object)
    if valores.shape != (2, 2):
        raise ValueError("a matriz de confusão deve ter formato 2 × 2")

    valores = [[valor_python(item) for item in linha] for linha in valores.tolist()]
    return pd.DataFrame(
        valores,
        index=["Real: não inadimplente (0)", "Real: inadimplente (1)"],
        columns=["Previsto: não inadimplente (0)", "Previsto: inadimplente (1)"],
    )


def exibir_avaliacao(metricas: Mapping[str, Any] | None, erro: str | None) -> None:
    st.subheader("Avaliação no conjunto de teste")
    if erro:
        st.error(erro)
        return

    teste = metricas["teste"]
    if not isinstance(teste, Mapping):
        st.error("O campo `teste` do arquivo de métricas não é um dicionário válido.")
        return

    nome_modelo = metricas.get("modelo") or obter(teste, "modelo")
    if nome_modelo:
        st.caption(f"Modelo final: {nome_modelo}")

    colunas = st.columns(4)
    indicadores = [
        ("Precision", obter(teste, "precision")),
        ("Recall", obter(teste, "recall")),
        ("F1", obter(teste, "f1")),
        ("AP / PR-AUC", obter(teste, "pr_auc", "ap_pr_auc")),
    ]
    for coluna, (rotulo, valor) in zip(colunas, indicadores):
        coluna.metric(rotulo, formatar_decimal(valor))

    st.markdown("#### Matriz de confusão")
    try:
        st.table(dataframe_matriz(metricas))
    except (TypeError, ValueError) as erro_matriz:
        st.warning(f"Não foi possível exibir a matriz de confusão: {erro_matriz}")

    limiar = metricas.get("limiar")
    st.info(
        f"O limiar de classificação ({formatar_decimal(limiar, 2)}) foi definido "
        "antes da avaliação final. O conjunto de teste foi avaliado uma única vez "
        "e não foi usado para ajustar novamente o modelo."
    )
    observacao = metricas.get("observacao")
    if observacao:
        st.caption(str(observacao))


def exibir_scoring(payload: Mapping[str, Any] | None, erro: str | None) -> None:
    st.subheader("Scoring individual")
    st.write("Preencha os dados de um cliente para obter a estimativa do modelo.")
    if erro:
        st.error(erro)
        return

    try:
        colunas_features = validar_campos(payload)
        limiar = numero(payload["limiar"])
    except (KeyError, TypeError, ValueError) as erro_schema:
        st.error(f"O artefato do modelo é incompatível com o formulário: {erro_schema}")
        return

    pipeline = payload["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        st.error("O pipeline salvo não oferece o método `predict_proba()`.")
        return

    valores_padrao = payload["valores_padrao"]
    opcoes_categoricas = payload["opcoes_categoricas"]
    respostas: dict[str, Any] = {}

    try:
        with st.form("formulario_scoring"):
            for grupo, campos in GRUPOS_CAMPOS.items():
                with st.expander(grupo, expanded=grupo == "Perfil"):
                    for inicio in range(0, len(campos), 2):
                        colunas_formulario = st.columns(2)
                        for coluna, campo in zip(
                            colunas_formulario, campos[inicio : inicio + 2]
                        ):
                            with coluna:
                                respostas[campo] = campo_do_formulario(
                                    campo, valores_padrao, opcoes_categoricas
                                )

            enviado = st.form_submit_button(
                "Calcular probabilidade", type="primary", use_container_width=True
            )
    except (KeyError, TypeError, ValueError) as erro_formulario:
        st.error(f"Não foi possível montar o formulário: {erro_formulario}")
        return

    if not enviado:
        return

    entrada = pd.DataFrame(
        [{coluna: respostas[coluna] for coluna in colunas_features}],
        columns=colunas_features,
    )
    try:
        probabilidades = np.asarray(pipeline.predict_proba(entrada))
        if probabilidades.ndim != 2 or probabilidades.shape[1] < 2:
            raise ValueError("predict_proba() não retornou duas classes")
        probabilidade = numero(probabilidades[0, 1])
    except Exception as erro_previsao:
        st.error(f"Não foi possível calcular o scoring: {erro_previsao}")
        return

    classe = int(probabilidade >= limiar)
    resultado, detalhe = st.columns(2)
    resultado.metric("Probabilidade de inadimplência", f"{probabilidade:.1%}")
    detalhe.metric("Limiar", formatar_decimal(limiar, 2))

    if classe == 1:
        st.error("Classificação: inadimplente (1)")
    else:
        st.success("Classificação: não inadimplente (0)")
    st.caption(
        "Este resultado é uma estimativa preditiva do modelo; não mede efeito causal "
        "nem retorno financeiro."
    )


modelo, erro_modelo = carregar_artefato(CAMINHO_MODELO, carregar_modelo, "modelo")
metricas, erro_metricas = carregar_artefato(
    CAMINHO_METRICAS, carregar_metricas, "métricas"
)

st.title("Classificação de inadimplência")
st.write("Aplicação educacional para avaliação do modelo e scoring individual.")

aba_avaliacao, aba_scoring = st.tabs(["Avaliação", "Scoring individual"])
with aba_avaliacao:
    exibir_avaliacao(metricas, erro_metricas)
with aba_scoring:
    exibir_scoring(modelo, erro_modelo)
