"""Recursos de apoio compartilhados pelos notebooks."""

from pathlib import Path

import pandas as pd


ALVO = "inadimplente"
COLUNA_ID = "id_cliente"

COLUNAS_NOMINAIS = [
    "sexo",
    "escolaridade",
    "estado_civil",
]

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


def encontrar_raiz(inicio=None):
    """Localiza a raiz do projeto."""

    pontos_de_partida = [
        Path(inicio or Path.cwd()).resolve()
    ]

    raiz_do_modulo = (
        Path(__file__).resolve().parent.parent
    )

    if raiz_do_modulo not in pontos_de_partida:
        pontos_de_partida.append(raiz_do_modulo)

    for ponto in pontos_de_partida:
        if ponto.is_file():
            ponto = ponto.parent

        for candidato in (
            ponto,
            *ponto.parents,
        ):
            arquivo_bruto = (
                candidato
                / "data"
                / "raw"
                / "UCI_Credit_Card.csv"
            )

            if (
                arquivo_bruto.exists()
                and (candidato / "src").is_dir()
            ):
                return candidato

    raise FileNotFoundError(
        "Não foi possível localizar "
        "a raiz do projeto."
    )


def carregar_base_preparada(raiz=None):
    """Carrega a base produzida na preparação."""

    if raiz is None:
        raiz_projeto = encontrar_raiz()
    else:
        raiz_projeto = Path(raiz).resolve()

    caminho = (
        raiz_projeto
        / "data"
        / "processed"
        / "cartao_credito_portugues.csv"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            "Base preparada não encontrada. "
            "Execute primeiro o notebook "
            "00_preparacao_base.ipynb."
        )

    return pd.read_csv(caminho)
