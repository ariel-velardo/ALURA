"""Recursos compartilhados pelos notebooks do curso."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SEMENTE = 42
ALVO = "inadimplente"
COLUNA_ID = "id_cliente"

COLUNAS_NOMINAIS = ["sexo", "escolaridade", "estado_civil"]
COLUNAS_STATUS = [
    "status_pagamento_set",
    "status_pagamento_ago",
    "status_pagamento_jul",
    "status_pagamento_jun",
    "status_pagamento_mai",
    "status_pagamento_abr",
]

MAPEAMENTO_COLUNAS = {
    "ID": COLUNA_ID,
    "LIMIT_BAL": "limite_credito",
    "SEX": "sexo",
    "EDUCATION": "escolaridade",
    "MARRIAGE": "estado_civil",
    "AGE": "idade",
    "PAY_0": "status_pagamento_set",
    "PAY_2": "status_pagamento_ago",
    "PAY_3": "status_pagamento_jul",
    "PAY_4": "status_pagamento_jun",
    "PAY_5": "status_pagamento_mai",
    "PAY_6": "status_pagamento_abr",
    "BILL_AMT1": "valor_fatura_set",
    "BILL_AMT2": "valor_fatura_ago",
    "BILL_AMT3": "valor_fatura_jul",
    "BILL_AMT4": "valor_fatura_jun",
    "BILL_AMT5": "valor_fatura_mai",
    "BILL_AMT6": "valor_fatura_abr",
    "PAY_AMT1": "valor_pago_set",
    "PAY_AMT2": "valor_pago_ago",
    "PAY_AMT3": "valor_pago_jul",
    "PAY_AMT4": "valor_pago_jun",
    "PAY_AMT5": "valor_pago_mai",
    "PAY_AMT6": "valor_pago_abr",
    "default.payment.next.month": ALVO,
}

PARAMETROS_REFERENCIA = {
    "n_estimators": 220,
    "learning_rate": 0.0303,
    "max_depth": 4,
    "min_samples_leaf": 35,
    "subsample": 0.9,
}


def encontrar_raiz(inicio=None):
    """Localiza a raiz do projeto a partir de um caminho ou do diretório atual."""
    pontos_de_partida = [Path(inicio or Path.cwd()).resolve()]
    raiz_do_modulo = Path(__file__).resolve().parent.parent
    if raiz_do_modulo not in pontos_de_partida:
        pontos_de_partida.append(raiz_do_modulo)

    for ponto in pontos_de_partida:
        ponto = ponto.parent if ponto.is_file() else ponto
        for candidato in (ponto, *ponto.parents):
            arquivo_bruto = candidato / "data" / "raw" / "UCI_Credit_Card.csv"
            if arquivo_bruto.exists() and (candidato / "src").is_dir():
                return candidato

    raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")


def carregar_base_preparada(raiz=None):
    """Carrega a versão em português produzida pelo notebook de preparação."""
    raiz_projeto = Path(raiz).resolve() if raiz is not None else encontrar_raiz()
    caminho = raiz_projeto / "data" / "processed" / "cartao_credito_portugues.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            "Base preparada não encontrada. Execute primeiro o notebook "
            "00_preparacao_base.ipynb."
        )
    return pd.read_csv(caminho)


def separar_dados(dados):
    """Separa features e alvo em treino, validação e teste estratificados (60/20/20)."""
    colunas_ausentes = {COLUNA_ID, ALVO}.difference(dados.columns)
    if colunas_ausentes:
        nomes = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"Colunas obrigatórias ausentes: {nomes}.")

    X = dados.drop(columns=[COLUNA_ID, ALVO])
    y = dados[ALVO]

    X_treino_validacao, X_teste, y_treino_validacao, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEMENTE,
    )
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_treino_validacao,
        y_treino_validacao,
        test_size=0.25,
        stratify=y_treino_validacao,
        random_state=SEMENTE,
    )

    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste


def _criar_preprocessamento(escalar):
    tratamento_restante = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "nominais",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_NOMINAIS,
            )
        ],
        remainder=tratamento_restante,
        verbose_feature_names_out=False,
    )


def criar_modelo_logistico():
    """Cria o benchmark logístico com one-hot e padronização das demais colunas."""
    return Pipeline(
        steps=[
            ("preprocessamento", _criar_preprocessamento(escalar=True)),
            ("modelo", LogisticRegression(max_iter=2000, random_state=SEMENTE)),
        ]
    )


def criar_modelo_floresta():
    """Cria a Random Forest usada como exemplo de bagging."""
    return Pipeline(
        steps=[
            ("preprocessamento", _criar_preprocessamento(escalar=False)),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=SEMENTE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def criar_modelo_gradiente(parametros=None):
    """Cria o Gradient Boosting padrão ou com os parâmetros informados."""
    parametros_modelo = dict(parametros or {})
    parametros_modelo["random_state"] = SEMENTE
    return Pipeline(
        steps=[
            ("preprocessamento", _criar_preprocessamento(escalar=False)),
            ("modelo", GradientBoostingClassifier(**parametros_modelo)),
        ]
    )


def criar_modelo_xgboost():
    """Cria a comparação avançada com a configuração simples da prova técnica."""
    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.10,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEMENTE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessamento", _criar_preprocessamento(escalar=False)),
            ("modelo", modelo),
        ]
    )


def avaliar_probabilidades(
    nome_modelo,
    y_verdadeiro,
    probabilidades,
    limiar=0.50,
    tempo_treino=np.nan,
):
    """Calcula métricas de classificação e AP/PR-AUC para um vetor de probabilidades."""
    probabilidades = np.asarray(probabilidades)
    previsoes = (probabilidades >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(
        y_verdadeiro,
        previsoes,
        labels=[0, 1],
    ).ravel()

    return {
        "modelo": nome_modelo,
        "limiar": limiar,
        "precision": precision_score(y_verdadeiro, previsoes, zero_division=0),
        "recall": recall_score(y_verdadeiro, previsoes, zero_division=0),
        "f1": f1_score(y_verdadeiro, previsoes, zero_division=0),
        "pr_auc": average_precision_score(y_verdadeiro, probabilidades),
        "vn": int(vn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(vp),
        "tempo_treino_s": tempo_treino,
    }


def metricas_limiares(y_verdadeiro, probabilidades, limiares):
    """Compara Precision, Recall e F1 em poucos limiares escolhidos."""
    linhas = []
    for limiar in limiares:
        metricas = avaliar_probabilidades(
            "Gradient Boosting",
            y_verdadeiro,
            probabilidades,
            limiar=limiar,
        )
        linhas.append(
            {
                chave: metricas[chave]
                for chave in ("limiar", "precision", "recall", "f1", "pr_auc")
            }
        )
    return pd.DataFrame(linhas)
