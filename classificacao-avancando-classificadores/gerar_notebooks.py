"""Sincroniza a infraestrutura dos notebooks sem preencher as células ao vivo."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
PASTA_NOTEBOOKS = RAIZ / "notebooks"

LOCALIZAR_RAIZ = '''from pathlib import Path
import sys

ponto_atual = Path.cwd().resolve()
RAIZ = next(
    caminho for caminho in (ponto_atual, *ponto_atual.parents)
    if (caminho / "data" / "raw" / "UCI_Credit_Card.csv").exists()
)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
'''

SPLIT_ESTRATIFICADO = '''dados = carregar_base_preparada(RAIZ)

X = dados.drop(columns=["id_cliente", "inadimplente"])
y = dados["inadimplente"]

X_treino_validacao, X_teste, y_treino_validacao, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X_treino_validacao,
    y_treino_validacao,
    test_size=0.25,
    stratify=y_treino_validacao,
    random_state=42,
)
'''

CARREGAR_PARAMETROS = '''caminho_parametros = RAIZ / "models" / "parametros_gradient_boosting.json"
if not caminho_parametros.exists():
    raise FileNotFoundError(
        "Execute primeiro o notebook 03_otimizacao.ipynb."
    )

parametros = json.loads(caminho_parametros.read_text(encoding="utf-8"))
'''

CELULAS_SETUP = {
    "01_problema_benchmark.ipynb": LOCALIZAR_RAIZ
    + '''

import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.auxiliares import COLUNAS_NOMINAIS, carregar_base_preparada
from src.visual_utils import grafico_comparacao_modelos
''',
    "02_ensembles.ipynb": LOCALIZAR_RAIZ
    + '''

import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.auxiliares import COLUNAS_NOMINAIS, carregar_base_preparada
from src.visual_utils import grafico_comparacao_modelos

'''
    + SPLIT_ESTRATIFICADO,
    "03_otimizacao.ipynb": LOCALIZAR_RAIZ
    + '''

import json
import time
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.auxiliares import COLUNAS_NOMINAIS, carregar_base_preparada
from src.visual_utils import (
    grafico_historico_optuna,
    grafico_importancia_hiperparametros,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

'''
    + SPLIT_ESTRATIFICADO,
    "04_desbalanceamento_threshold.ipynb": LOCALIZAR_RAIZ
    + '''

import json
import time
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.auxiliares import (
    COLUNAS_NOMINAIS,
    COLUNAS_STATUS,
    carregar_base_preparada,
)
from src.visual_utils import grafico_metricas_por_limiar

'''
    + SPLIT_ESTRATIFICADO
    + "\n"
    + CARREGAR_PARAMETROS,
    "05_semisupervisionado_importancia.ipynb": LOCALIZAR_RAIZ
    + '''

import json
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

from src.auxiliares import COLUNAS_NOMINAIS, carregar_base_preparada
from src.visual_utils import grafico_importancia_variaveis

'''
    + SPLIT_ESTRATIFICADO
    + "\n"
    + CARREGAR_PARAMETROS,
    "06_modelo_final.ipynb": LOCALIZAR_RAIZ
    + '''

import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.auxiliares import (
    ALVO,
    COLUNAS_NOMINAIS,
    COLUNAS_STATUS,
    carregar_base_preparada,
)
from src.visual_utils import grafico_matriz_confusao

LIMIAR_FINAL = 0.27

'''
    + SPLIT_ESTRATIFICADO
    + "\n"
    + CARREGAR_PARAMETROS,
}

CELULAS_FIXAS = {
    "03_otimizacao.ipynb": {
        1: '''## Objetivo

Um tuning curto melhora o Gradient Boosting de forma útil para a aula?

Variamos quatro hiperparâmetros em 20 trials: `n_estimators`, `learning_rate`,
`max_depth` e `subsample`. Os demais permanecem com os valores padrão do
scikit-learn.
''',
        14: '''## Resultado

O ganho observado é modesto. Isso é pedagogicamente útil: tuning organiza a
busca, mas não garante salto grande. O teste continua intocado.
''',
    },
    "04_desbalanceamento_threshold.ipynb": {
        7: '''## O SMOTENC melhora o ranking?

Os códigos nominais e de status de pagamento são categorias do domínio. Para
evitar sua interpolação como valores contínuos, usamos
`COLUNAS_NOMINAIS + COLUNAS_STATUS` como features categóricas do SMOTENC.
''',
        11: '''colunas_comparacao = [
    "modelo",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "fp",
]

resultados_balanceamento[
    colunas_comparacao
].sort_values("average_precision", ascending=False)
''',
        12: '''A comparação considera o ranking probabilístico pela Average Precision e
também o efeito em Precision, Recall e falsos positivos. Pesos elevam Recall,
mas aumentam os falsos positivos; o SMOTENC adiciona complexidade sem melhorar
o ranking neste experimento. Por isso, seguimos com os dados originais.
''',
        16: '''## Resultado

O modelo com dados originais permanece como solução final. Na validação,
avaliamos diferentes limiares e observamos o trade-off entre Precision e
Recall. Para este projeto didático seguimos com 0,27, decisão tomada antes de
abrir o teste. Esse valor não é universal e depende do contexto de uso.
''',
    },
    "06_modelo_final.ipynb": {
        9: '''previsoes_teste = (probabilidades_teste >= LIMIAR_FINAL).astype(int)
matriz_teste = confusion_matrix(y_teste, previsoes_teste, labels=[0, 1])

fig = grafico_matriz_confusao(matriz_teste)
fig.show()
''',
    },
}

NOTEBOOKS = [
    "00_preparacao_base.ipynb",
    *CELULAS_SETUP,
]

KERNELSPEC = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3",
}


def como_linhas(texto: str) -> list[str]:
    """Converte uma string no formato de ``source`` usado pelo nbformat."""
    linhas = texto.rstrip().splitlines(keepends=True)
    if linhas:
        linhas[-1] = linhas[-1].removesuffix("\n")
    return linhas


def atualizar_notebook(caderno: dict, nome: str) -> dict:
    """Atualiza apenas setup e nomenclatura, preservando células e metadados."""
    atualizado = copy.deepcopy(caderno)
    if nome in CELULAS_SETUP:
        atualizado["cells"][2]["source"] = como_linhas(CELULAS_SETUP[nome])
    for indice, texto in CELULAS_FIXAS.get(nome, {}).items():
        atualizado["cells"][indice]["source"] = como_linhas(texto)

    atualizado["metadata"]["kernelspec"] = KERNELSPEC.copy()

    for celula in atualizado["cells"]:
        texto = "".join(celula.get("source", []))
        texto = texto.replace("AP / PR-AUC", "Average Precision (AP)")
        texto = texto.replace("PR-AUC", "Average Precision")
        texto = texto.replace("pr_auc", "average_precision")
        texto = texto.replace(
            "Average Precision (Average Precision (AP))",
            "Average Precision (AP)",
        )
        celula["source"] = como_linhas(texto) if texto else []

    return atualizado


def sincronizar(apenas_conferir: bool) -> list[str]:
    """Confere ou grava os notebooks e devolve os nomes desatualizados."""
    desatualizados = []
    for nome in NOTEBOOKS:
        caminho = PASTA_NOTEBOOKS / nome
        caderno = json.loads(caminho.read_text(encoding="utf-8"))
        atualizado = atualizar_notebook(caderno, nome)
        if atualizado == caderno:
            continue

        desatualizados.append(nome)
        if not apenas_conferir:
            caminho.write_text(
                json.dumps(atualizado, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )

    return desatualizados


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="apenas verifica se os notebooks refletem o gerador",
    )
    argumentos = parser.parse_args()
    desatualizados = sincronizar(argumentos.check)

    if argumentos.check and desatualizados:
        print("Notebooks desatualizados:", ", ".join(desatualizados))
        return 1

    if desatualizados:
        print("Notebooks atualizados:", ", ".join(desatualizados))
    else:
        print("Notebooks já estavam atualizados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
