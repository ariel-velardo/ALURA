"""Gera os notebooks completos de desenvolvimento do curso."""

from pathlib import Path

import nbformat as nbf


RAIZ = Path(__file__).resolve().parent.parent
PASTA_NOTEBOOKS = RAIZ / "notebooks"


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip())


def codigo(texto):
    return nbf.v4.new_code_cell(texto.strip())


def salvar(nome, titulo, celulas):
    notebook = nbf.v4.new_notebook(
        cells=[md(f"# {titulo}"), *celulas],
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "curso": {
                "idioma": "pt-BR",
                "tipo": "desenvolvimento",
                "referencia_tecnica": "00_validacao_tecnica.ipynb",
            },
        },
    )
    caminho = PASTA_NOTEBOOKS / nome
    nbf.write(notebook, caminho)
    print(f"Criado: {caminho.relative_to(RAIZ)}")


CONFIGURAR_RAIZ = """
from pathlib import Path
import sys

ponto_atual = Path.cwd().resolve()
RAIZ = next(
    caminho for caminho in (ponto_atual, *ponto_atual.parents)
    if (caminho / "data" / "raw" / "UCI_Credit_Card.csv").exists()
)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
"""


def notebook_00():
    celulas = [
        md(
            """
## Objetivo

Como transformar a base original em uma versão curta e legível para o curso?

Este notebook valida o contrato mínimo, traduz as colunas e salva uma cópia
preparada. O CSV bruto é somente lido.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import hashlib
import pandas as pd

from src.auxiliares import ALVO, COLUNA_ID, MAPEAMENTO_COLUNAS
from src.visual_utils import grafico_distribuicao_alvo

CAMINHO_BRUTO = RAIZ / "data" / "raw" / "UCI_Credit_Card.csv"
CAMINHO_PREPARADO = RAIZ / "data" / "processed" / "cartao_credito_portugues.csv"
"""
        ),
        md(
            """
## Como é a base?

A unidade observada é um registro por ID de cliente/conta. Não há datas por
linha: os meses aparecem apenas no significado das colunas.
"""
        ),
        codigo(
            """
hash_bruto = hashlib.sha256(CAMINHO_BRUTO.read_bytes()).hexdigest()
dados_brutos = pd.read_csv(CAMINHO_BRUTO)

print(f"Dimensões: {dados_brutos.shape[0]:,} linhas x {dados_brutos.shape[1]} colunas")
print(f"SHA-256: {hash_bruto}")
dados_brutos.head()
"""
        ),
        md(
            """
## O contrato permite modelagem?

Antes de qualquer modelo, verificamos chave, granularidade, missing, duplicados
e outcome. A base não contém tratamento nem controle; este projeto é preditivo,
não causal.
"""
        ),
        codigo(
            """
alvo_original = "default.payment.next.month"
resumo_qualidade = pd.Series({
    "linhas": len(dados_brutos),
    "colunas": dados_brutos.shape[1],
    "missing": int(dados_brutos.isna().sum().sum()),
    "duplicatas_exatas": int(dados_brutos.duplicated().sum()),
    "ids_duplicados": int(dados_brutos["ID"].duplicated().sum()),
    "taxa_inadimplencia": dados_brutos[alvo_original].mean(),
})
assert dados_brutos["ID"].is_unique
assert set(dados_brutos[alvo_original].unique()) == {0, 1}
resumo_qualidade.to_frame("resultado")
"""
        ),
        md(
            """
Os 30 mil IDs são únicos, não há valores ausentes e o target é binário.
Perfis coincidentes sem o ID são mantidos: eles não comprovam duplicação do
mesmo cliente.
"""
        ),
        md(
            """
## Como ficam os nomes em português?

Os valores e códigos permanecem iguais aos da fonte. Somente os nomes mudam.
"""
        ),
        codigo(
            """
dicionario_colunas = MAPEAMENTO_COLUNAS.copy()
pd.DataFrame(
    dicionario_colunas.items(),
    columns=["nome_original", "nome_no_curso"],
)
"""
        ),
        codigo(
            """
dados = dados_brutos.rename(columns=dicionario_colunas)
CAMINHO_PREPARADO.parent.mkdir(parents=True, exist_ok=True)
dados.to_csv(CAMINHO_PREPARADO, index=False)

assert dados.shape == dados_brutos.shape
assert COLUNA_ID in dados and ALVO in dados
print(f"Base preparada salva em: {CAMINHO_PREPARADO}")
"""
        ),
        md(
            """
## Qual é o desbalanceamento do problema?

A classe positiva é minoritária, mas ainda possui milhares de exemplos.
"""
        ),
        codigo(
            """
distribuicao_alvo = (
    dados[ALVO].value_counts().sort_index()
    .rename_axis("inadimplente").to_frame("clientes")
)
distribuicao_alvo["proporcao"] = distribuicao_alvo["clientes"] / len(dados)

display(distribuicao_alvo)
fig = grafico_distribuicao_alvo(distribuicao_alvo)
fig.show()
"""
        ),
        md(
            """
## Existem códigos ou valores que exigem cuidado?

Os códigos não documentados são registrados, não corrigidos. Valores
monetários negativos podem representar saldo credor e também são preservados.
"""
        ),
        codigo(
            """
colunas_categoricas = ["sexo", "escolaridade", "estado_civil"]
codigos = pd.Series({
    coluna: sorted(dados[coluna].unique().tolist())
    for coluna in colunas_categoricas
}, name="valores_observados")

faixas = dados[
    ["limite_credito", "idade", "valor_fatura_set", "valor_pago_set"]
].agg(["min", "median", "max"]).T
display(codigos.to_frame(), faixas)
"""
        ),
        md(
            """
## Resultado

A cópia em português preserva 30.000 linhas e 25 colunas. O ID será excluído
apenas das features; inadimplente será o outcome. A taxa positiva é 22,12%.
"""
        ),
    ]
    salvar("00_preparacao_base.ipynb", "00 — Preparação da base", celulas)


def notebook_01():
    celulas = [
        md(
            """
## Objetivo

Qual referência simples deve orientar as próximas etapas?

Comparamos Regressão Logística e Random Forest na mesma validação. O teste fica
reservado até o notebook final.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import time
import pandas as pd

from src.auxiliares import (
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_floresta,
    criar_modelo_logistico,
    separar_dados,
)
from src.visual_utils import grafico_comparacao_modelos
"""
        ),
        md("## Como separar features, target e conjuntos?"),
        codigo(
            """
dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)

pd.DataFrame({
    "conjunto": ["treino", "validação", "teste protegido"],
    "registros": [len(X_treino), len(X_validacao), len(X_teste)],
    "proporcao_positiva": [y_treino.mean(), y_validacao.mean(), "protegido"],
})
"""
        ),
        md(
            """
O split estratificado produz 60% para treino, 20% para validação e 20% para
teste. ID e target não entram nas 23 features.
"""
        ),
        md("## O que a Regressão Logística entrega?"),
        codigo(
            """
modelo_logistico = criar_modelo_logistico()
inicio = time.perf_counter()
modelo_logistico.fit(X_treino, y_treino)
tempo_logistico = time.perf_counter() - inicio
previsoes_logisticas = modelo_logistico.predict(X_validacao)
probabilidades_logisticas = modelo_logistico.predict_proba(X_validacao)[:, 1]
"""
        ),
        codigo(
            """
pd.DataFrame({
    "classe_prevista": previsoes_logisticas[:5],
    "probabilidade": probabilidades_logisticas[:5],
})
"""
        ),
        md("## O bagging melhora a referência?"),
        codigo(
            """
modelo_floresta = criar_modelo_floresta()
inicio = time.perf_counter()
modelo_floresta.fit(X_treino, y_treino)
tempo_floresta = time.perf_counter() - inicio
previsoes_floresta = modelo_floresta.predict(X_validacao)
probabilidades_floresta = modelo_floresta.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## Como comparar os dois modelos?"),
        codigo(
            """
resultados = pd.DataFrame([
    avaliar_probabilidades(
        "Regressão Logística", y_validacao, probabilidades_logisticas,
        tempo_treino=tempo_logistico,
    ),
    avaliar_probabilidades(
        "Random Forest", y_validacao, probabilidades_floresta,
        tempo_treino=tempo_floresta,
    ),
])
resultados
"""
        ),
        codigo(
            """
fig = grafico_comparacao_modelos(resultados)
fig.show()
"""
        ),
        md(
            """
## Resultado

A Random Forest melhora o ranking probabilístico em relação à logística e será
o exemplo prático de bagging. Precision, Recall e F1 ainda usam limiar 0,50.
"""
        ),
    ]
    salvar("01_problema_benchmark.ipynb", "01 — Problema e benchmark", celulas)


def notebook_02():
    celulas = [
        md(
            """
## Objetivo

Como Bagging e Boosting se comportam no mesmo problema?

Random Forest representa Bagging; Gradient Boosting é o boosting principal;
XGBoost aparece uma única vez como comparação avançada, sem tuning.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import time
import pandas as pd

from src.auxiliares import (
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_floresta,
    criar_modelo_gradiente,
    criar_modelo_xgboost,
    separar_dados,
)
from src.visual_utils import grafico_comparacao_modelos

dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)
"""
        ),
        md("## Bagging: qual é a referência?"),
        codigo(
            """
modelo_floresta = criar_modelo_floresta()
inicio = time.perf_counter()
modelo_floresta.fit(X_treino, y_treino)
tempo_floresta = time.perf_counter() - inicio
probabilidades_floresta = modelo_floresta.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## Boosting: o Gradient Boosting avança?"),
        codigo(
            """
modelo_gradiente = criar_modelo_gradiente()
inicio = time.perf_counter()
modelo_gradiente.fit(X_treino, y_treino)
tempo_gradiente = time.perf_counter() - inicio
probabilidades_gradiente = modelo_gradiente.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## Quanto o XGBoost acrescenta sem tuning?"),
        codigo(
            """
modelo_xgboost = criar_modelo_xgboost()
inicio = time.perf_counter()
modelo_xgboost.fit(X_treino, y_treino)
tempo_xgboost = time.perf_counter() - inicio
probabilidades_xgboost = modelo_xgboost.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## A diferença muda a escolha pedagógica?"),
        codigo(
            """
resultados = pd.DataFrame([
    avaliar_probabilidades("Random Forest", y_validacao, probabilidades_floresta, tempo_treino=tempo_floresta),
    avaliar_probabilidades("Gradient Boosting", y_validacao, probabilidades_gradiente, tempo_treino=tempo_gradiente),
    avaliar_probabilidades("XGBoost", y_validacao, probabilidades_xgboost, tempo_treino=tempo_xgboost),
]).sort_values("pr_auc", ascending=False)

resultados
"""
        ),
        codigo(
            """
fig = grafico_comparacao_modelos(resultados, "Bagging e Boosting na validação")
fig.show()
"""
        ),
        md(
            """
## Resultado

O Gradient Boosting fica muito próximo do XGBoost e mantém a implementação
principal dentro do scikit-learn. O ganho pequeno do XGBoost não muda o foco do
curso.
"""
        ),
    ]
    salvar("02_ensembles.ipynb", "02 — Ensembles", celulas)


def notebook_03():
    celulas = [
        md(
            """
## Objetivo

Um tuning curto melhora o Gradient Boosting de forma útil para a aula?

Variamos quatro hiperparâmetros em 20 trials. min_samples_leaf fica fixo em 35,
decisão já validada na prova técnica.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import json
import time
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.auxiliares import (
    PARAMETROS_REFERENCIA,
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_gradiente,
    separar_dados,
)
from src.visual_utils import (
    grafico_historico_optuna,
    grafico_importancia_hiperparametros,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)
"""
        ),
        md("## Qual é o ponto de partida?"),
        codigo(
            """
modelo_inicial = criar_modelo_gradiente()
modelo_inicial.fit(X_treino, y_treino)
probabilidades_iniciais = modelo_inicial.predict_proba(X_validacao)[:, 1]
resultado_inicial = avaliar_probabilidades(
    "Gradient Boosting inicial",
    y_validacao,
    probabilidades_iniciais,
)
pd.DataFrame([resultado_inicial])
"""
        ),
        md("## Que combinação o Optuna deve avaliar?"),
        codigo(
            """
def objetivo(trial):
    parametros = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.1),
        "min_samples_leaf": 35,
    }
    modelo = criar_modelo_gradiente(parametros)
    modelo.fit(X_treino, y_treino)
    probabilidades = modelo.predict_proba(X_validacao)[:, 1]
    return average_precision_score(y_validacao, probabilidades)
"""
        ),
        md("## Como executar uma busca controlada?"),
        codigo(
            """
estudo = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
estudo.enqueue_trial({
    "n_estimators": 220,
    "learning_rate": 0.0303,
    "max_depth": 4,
    "subsample": 0.9,
})
inicio = time.perf_counter()
estudo.optimize(objetivo, n_trials=20, show_progress_bar=False)
tempo_busca = time.perf_counter() - inicio
print(f"Melhor AP / PR-AUC: {estudo.best_value:.4f}")
print(f"Tempo da busca: {tempo_busca:.1f} s")
"""
        ),
        md("## A busca convergiu e quais parâmetros importaram?"),
        codigo(
            """
historico = estudo.trials_dataframe(attrs=("number", "value"))
historico["melhor_acumulado"] = historico["value"].cummax()
importancias = optuna.importance.get_param_importances(estudo)

fig = grafico_historico_optuna(historico)
fig.show()

fig = grafico_importancia_hiperparametros(importancias)
fig.show()
"""
        ),
        md("## O modelo otimizado melhora a referência?"),
        codigo(
            """
parametros_otimizados = {
    **estudo.best_params,
    "min_samples_leaf": 35,
}
modelo_otimizado = criar_modelo_gradiente(parametros_otimizados)
modelo_otimizado.fit(X_treino, y_treino)
probabilidades_otimizadas = modelo_otimizado.predict_proba(X_validacao)[:, 1]
resultado_otimizado = avaliar_probabilidades(
    "Gradient Boosting otimizado",
    y_validacao,
    probabilidades_otimizadas,
)
pd.DataFrame([resultado_inicial, resultado_otimizado])
"""
        ),
        codigo(
            """
caminho_parametros = RAIZ / "models" / "parametros_gradient_boosting.json"
caminho_parametros.parent.mkdir(parents=True, exist_ok=True)
caminho_parametros.write_text(
    json.dumps(parametros_otimizados, indent=2),
    encoding="utf-8",
)
print(f"Parâmetros salvos em: {caminho_parametros}")
parametros_otimizados
"""
        ),
        md(
            """
## Resultado

O ganho esperado é modesto, próximo ao observado na prova técnica. Isso é
pedagogicamente útil: tuning organiza a busca, mas não garante salto grande.
O teste continua intocado.
"""
        ),
    ]
    salvar("03_otimizacao.ipynb", "03 — Otimização", celulas)


def notebook_04():
    celulas = [
        md(
            """
## Objetivo

O desbalanceamento deve ser tratado por pesos, SMOTENC ou por uma decisão de
limiar?

Comparamos somente as três estratégias já validadas. Nenhum novo resampler é
adicionado.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import json
import time
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.utils.class_weight import compute_sample_weight

from src.auxiliares import (
    COLUNAS_NOMINAIS,
    PARAMETROS_REFERENCIA,
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_gradiente,
    metricas_limiares,
    separar_dados,
)
from src.visual_utils import grafico_metricas_por_limiar

dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)
caminho_parametros = RAIZ / "models" / "parametros_gradient_boosting.json"
parametros = json.loads(caminho_parametros.read_text()) if caminho_parametros.exists() else PARAMETROS_REFERENCIA.copy()
"""
        ),
        md("## Como se comporta o modelo original?"),
        codigo(
            """
modelo_original = criar_modelo_gradiente(parametros)
inicio = time.perf_counter()
modelo_original.fit(X_treino, y_treino)
tempo_original = time.perf_counter() - inicio
probabilidades_originais = modelo_original.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## Pesos aumentam o alcance da classe positiva?"),
        codigo(
            """
pesos = compute_sample_weight(class_weight="balanced", y=y_treino)
modelo_pesos = criar_modelo_gradiente(parametros)
inicio = time.perf_counter()
modelo_pesos.fit(X_treino, y_treino, modelo__sample_weight=pesos)
tempo_pesos = time.perf_counter() - inicio
probabilidades_pesos = modelo_pesos.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## O SMOTENC melhora o ranking?"),
        codigo(
            """
indices_nominais = [X_treino.columns.get_loc(coluna) for coluna in COLUNAS_NOMINAIS]
smote = SMOTENC(categorical_features=indices_nominais, random_state=42)
X_treino_smote, y_treino_smote = smote.fit_resample(X_treino, y_treino)
modelo_smote = criar_modelo_gradiente(parametros)
inicio = time.perf_counter()
modelo_smote.fit(X_treino_smote, y_treino_smote)
tempo_smote = time.perf_counter() - inicio
probabilidades_smote = modelo_smote.predict_proba(X_validacao)[:, 1]
"""
        ),
        md("## Qual estratégia preserva melhor a AP / PR-AUC?"),
        codigo(
            """
resultados_balanceamento = pd.DataFrame([
    avaliar_probabilidades("Original", y_validacao, probabilidades_originais, tempo_treino=tempo_original),
    avaliar_probabilidades("Pesos", y_validacao, probabilidades_pesos, tempo_treino=tempo_pesos),
    avaliar_probabilidades("SMOTENC", y_validacao, probabilidades_smote, tempo_treino=tempo_smote),
]).sort_values("pr_auc", ascending=False)

print("Antes do SMOTENC:", y_treino.value_counts().sort_index().to_dict())
print("Depois do SMOTENC:", pd.Series(y_treino_smote).value_counts().sort_index().to_dict())
resultados_balanceamento
"""
        ),
        codigo(
            """
ap_original = resultados_balanceamento.query("modelo == 'Original'")["pr_auc"].iloc[0]
ap_alternativas = resultados_balanceamento.query("modelo != 'Original'")["pr_auc"]
ganho_maximo = ap_alternativas.max() - ap_original
estrategia_escolhida = "Original" if ganho_maximo <= 0.005 else "Reavaliar"

pd.Series({
    "estrategia_escolhida": estrategia_escolhida,
    "maior_ganho_alternativo_em_ap": ganho_maximo,
    "criterio_minimo": 0.005,
}).to_frame("resultado")
"""
        ),
        md(
            """
Pesos elevam Recall, mas também os falsos positivos. Um ganho de AP inferior a
0,005 não compensa essa mudança neste projeto. SMOTENC não precisa vencer para
ensinar: balancear classes não garante ranking melhor.
"""
        ),
        md("## O que muda quando alteramos o limiar?"),
        codigo(
            """
resultados_limiares = metricas_limiares(
    y_validacao,
    probabilidades_originais,
    [0.50, 0.40, 0.30, 0.27],
)
resultados_limiares
"""
        ),
        codigo(
            """
fig = grafico_metricas_por_limiar(resultados_limiares)
fig.show()
"""
        ),
        md(
            """
## Resultado

O modelo original permanece como solução final. O limiar 0,27 é uma escolha
ilustrativa, definida na prova técnica para aumentar Recall com perda de
Precision. Não há custo financeiro disponível para otimizá-lo.
"""
        ),
    ]
    salvar(
        "04_desbalanceamento_threshold.ipynb",
        "04 — Desbalanceamento e limiar",
        celulas,
    )


def notebook_05():
    celulas = [
        md(
            """
## Objetivo

Self-training melhora o ranking quando parte dos rótulos é ocultada? E como
interpretar globalmente o modelo escolhido?

A demonstração semissupervisionada é curta. Depois usamos importâncias nativas
e Permutation Importance, sem explicações individuais.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import json
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

from src.auxiliares import (
    COLUNAS_NOMINAIS,
    PARAMETROS_REFERENCIA,
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_gradiente,
    separar_dados,
)
from src.visual_utils import grafico_importancia_variaveis

dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)
caminho_parametros = RAIZ / "models" / "parametros_gradient_boosting.json"
parametros = json.loads(caminho_parametros.read_text()) if caminho_parametros.exists() else PARAMETROS_REFERENCIA.copy()
"""
        ),
        md("## Como simular poucos rótulos disponíveis?"),
        codigo(
            """
posicoes = np.arange(len(X_treino))
posicoes_rotuladas, _ = train_test_split(
    posicoes, train_size=0.25, stratify=y_treino, random_state=42,
)
X_treino_array = X_treino.to_numpy()
X_validacao_array = X_validacao.to_numpy()
y_treino_array = y_treino.to_numpy()
rotulos_parciais = np.full(len(y_treino_array), -1, dtype=int)
rotulos_parciais[posicoes_rotuladas] = y_treino_array[posicoes_rotuladas]
"""
        ),
        md("## Qual estimador será usado no self-training?"),
        codigo(
            """
indices_nominais = [X_treino.columns.get_loc(coluna) for coluna in COLUNAS_NOMINAIS]
indices_numericos = [i for i in range(X_treino.shape[1]) if i not in indices_nominais]

def criar_estimador_base():
    preprocessamento = ColumnTransformer([
        ("nominais", OneHotEncoder(handle_unknown="ignore", sparse_output=False), indices_nominais),
        ("numericas", StandardScaler(), indices_numericos),
    ])
    return Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", LogisticRegression(max_iter=2000, random_state=42)),
    ])
"""
        ),
        md("## O self-training supera o mesmo subconjunto rotulado?"),
        codigo(
            """
modelo_supervisionado = criar_estimador_base()
modelo_supervisionado.fit(
    X_treino_array[posicoes_rotuladas],
    y_treino_array[posicoes_rotuladas],
)
probabilidades_supervisionadas = modelo_supervisionado.predict_proba(X_validacao_array)[:, 1]

modelo_self_training = SelfTrainingClassifier(
    estimator=criar_estimador_base(),
    criterion="threshold",
    threshold=0.90,
    max_iter=10,
)
modelo_self_training.fit(X_treino_array, rotulos_parciais)
probabilidades_self_training = modelo_self_training.predict_proba(X_validacao_array)[:, 1]
"""
        ),
        codigo(
            """
resultados_semissupervisionado = pd.DataFrame([
    avaliar_probabilidades("Supervisionado — 25% rotulado", y_validacao, probabilidades_supervisionadas),
    avaliar_probabilidades("Self-training", y_validacao, probabilidades_self_training),
])
pseudorrotulos = modelo_self_training.transduction_[rotulos_parciais == -1]
print("Pseudorrótulos adicionados:", int((pseudorrotulos != -1).sum()))
resultados_semissupervisionado
"""
        ),
        md(
            """
O self-training demonstra a técnica, mas não melhora a AP / PR-AUC neste
experimento. Ele não será a solução final.
"""
        ),
        md("## Quais variáveis o modelo usa globalmente?"),
        codigo(
            """
modelo_interpretado = criar_modelo_gradiente(parametros)
modelo_interpretado.fit(X_treino, y_treino)
nomes_transformados = modelo_interpretado.named_steps[
    "preprocessamento"
].get_feature_names_out()
importancias_nativas = pd.DataFrame({
    "variavel": nomes_transformados,
    "importancia": modelo_interpretado.named_steps["modelo"].feature_importances_,
}).sort_values("importancia", ascending=False)
importancias_nativas.head(12)
"""
        ),
        codigo(
            """
fig = grafico_importancia_variaveis(
    importancias_nativas,
    titulo="Importância nativa do Gradient Boosting",
)
fig.show()
"""
        ),
        md("## A importância permanece ao embaralhar cada feature original?"),
        codigo(
            """
inicio = time.perf_counter()
permutacao = permutation_importance(
    modelo_interpretado, X_validacao, y_validacao,
    scoring="average_precision", n_repeats=10,
    random_state=42, n_jobs=-1,
)
tempo_permutacao = time.perf_counter() - inicio
importancias_permutacao = pd.DataFrame({
    "variavel": X_validacao.columns,
    "importancia_media": permutacao.importances_mean,
    "desvio": permutacao.importances_std,
}).sort_values("importancia_media", ascending=False)
print(f"Tempo: {tempo_permutacao:.1f} s")
importancias_permutacao.head(12)
"""
        ),
        codigo(
            """
fig = grafico_importancia_variaveis(
    importancias_permutacao,
    coluna_valor="importancia_media",
    titulo="Permutation Importance na validação",
    coluna_desvio="desvio",
)
fig.show()
"""
        ),
        md(
            """
## Resultado

O status de pagamento mais recente lidera as duas leituras. Importância
preditiva não implica causalidade, e variáveis mensais correlacionadas podem
dividir importância.
"""
        ),
    ]
    salvar(
        "05_semisupervisionado_importancia.ipynb",
        "05 — Semissupervisionado e importância",
        celulas,
    )


def notebook_06():
    celulas = [
        md(
            """
## Objetivo

Como congelar a decisão e avaliar o teste uma única vez?

O modelo final é o Gradient Boosting otimizado, sem pesos ou SMOTENC, com
limiar 0,27. Para preservar a comparação técnica, ele permanece treinado apenas
no conjunto de treino.
"""
        ),
        codigo(
            CONFIGURAR_RAIZ
            + """

import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.auxiliares import (
    ALVO,
    COLUNAS_NOMINAIS,
    COLUNAS_STATUS,
    PARAMETROS_REFERENCIA,
    avaliar_probabilidades,
    carregar_base_preparada,
    criar_modelo_gradiente,
    separar_dados,
)
from src.visual_utils import grafico_matriz_confusao

LIMIAR_FINAL = 0.27
dados = carregar_base_preparada(RAIZ)
X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = separar_dados(dados)
caminho_parametros = RAIZ / "models" / "parametros_gradient_boosting.json"
parametros = json.loads(caminho_parametros.read_text()) if caminho_parametros.exists() else PARAMETROS_REFERENCIA.copy()
"""
        ),
        md("## O desempenho de validação continua coerente?"),
        codigo(
            """
modelo_final = criar_modelo_gradiente(parametros)
modelo_final.fit(X_treino, y_treino)
probabilidades_validacao = modelo_final.predict_proba(X_validacao)[:, 1]
resultado_validacao = avaliar_probabilidades(
    "Validação",
    y_validacao,
    probabilidades_validacao,
    limiar=LIMIAR_FINAL,
)
pd.DataFrame([resultado_validacao])
"""
        ),
        md("## Quais decisões ficam congeladas antes do teste?"),
        codigo(
            """
decisoes = pd.Series({
    "modelo": "GradientBoostingClassifier otimizado",
    "parametros": parametros,
    "balanceamento": "dados originais",
    "limiar": LIMIAR_FINAL,
    "metrica_selecao": "Average Precision (AP / PR-AUC)",
    "refit_com_validacao": False,
})
decisoes.to_frame("decisao")
"""
        ),
        md("## Qual é o resultado no teste protegido?"),
        codigo(
            """
avaliacoes_teste = 0
probabilidades_teste = modelo_final.predict_proba(X_teste)[:, 1]
avaliacoes_teste += 1
resultado_teste = avaliar_probabilidades(
    "Teste final",
    y_teste,
    probabilidades_teste,
    limiar=LIMIAR_FINAL,
)
assert avaliacoes_teste == 1
pd.DataFrame([resultado_validacao, resultado_teste])
"""
        ),
        codigo(
            """
previsoes_teste = (probabilidades_teste >= LIMIAR_FINAL).astype(int)
fig = grafico_matriz_confusao(y_teste, previsoes_teste)
fig.show()
"""
        ),
        md("## Como salvar o pipeline para a aplicação?"),
        codigo(
            """
colunas_features = X_treino.columns.tolist()
campos_com_opcoes = [*COLUNAS_NOMINAIS, *COLUNAS_STATUS]
opcoes_categoricas = {
    coluna: sorted(int(valor) for valor in X_treino[coluna].unique())
    for coluna in campos_com_opcoes
}
valores_padrao = {
    coluna: (
        int(X_treino[coluna].mode().iloc[0])
        if coluna in campos_com_opcoes or coluna == "idade"
        else float(X_treino[coluna].median())
    )
    for coluna in colunas_features
}
versoes = {
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "joblib": joblib.__version__,
}
"""
        ),
        codigo(
            """
payload_modelo = {
    "pipeline": modelo_final,
    "limiar": LIMIAR_FINAL,
    "colunas_features": colunas_features,
    "alvo": ALVO,
    "valores_padrao": valores_padrao,
    "opcoes_categoricas": opcoes_categoricas,
    "versoes": versoes,
}
caminho_modelo = RAIZ / "models" / "modelo_inadimplencia.joblib"
joblib.dump(payload_modelo, caminho_modelo)
print(f"Modelo salvo em: {caminho_modelo}")
"""
        ),
        codigo(
            """
def valor_json(valor):
    if isinstance(valor, dict):
        return {chave: valor_json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [valor_json(item) for item in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor

matriz_teste = [
    [resultado_teste["vn"], resultado_teste["fp"]],
    [resultado_teste["fn"], resultado_teste["vp"]],
]
metricas_modelo = {
    "modelo": "Gradient Boosting otimizado",
    "limiar": LIMIAR_FINAL,
    "validacao": resultado_validacao,
    "teste": resultado_teste,
    "matriz_confusao_teste": matriz_teste,
    "observacao": "Modelo treinado somente no treino; teste avaliado uma vez.",
}
caminho_metricas = RAIZ / "models" / "metricas_modelo.json"
caminho_metricas.write_text(
    json.dumps(valor_json(metricas_modelo), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Métricas salvas em: {caminho_metricas}")
"""
        ),
        md("## O artefato restaurado produz a mesma probabilidade?"),
        codigo(
            """
modelo_restaurado = joblib.load(caminho_modelo)
amostra = X_validacao.iloc[:5]
probabilidades_antes = modelo_final.predict_proba(amostra)[:, 1]
probabilidades_depois = modelo_restaurado["pipeline"].predict_proba(amostra)[:, 1]
assert np.allclose(probabilidades_antes, probabilidades_depois)

pd.DataFrame({
    "probabilidade": probabilidades_depois,
    "limiar": LIMIAR_FINAL,
    "classe": (probabilidades_depois >= LIMIAR_FINAL).astype(int),
})
"""
        ),
        md(
            """
## Resultado

O modelo, o limiar, os valores padrão e as métricas foram serializados para o
Streamlit. O teste só foi consultado depois do congelamento das decisões.
"""
        ),
    ]
    salvar("06_modelo_final.ipynb", "06 — Modelo final", celulas)


if __name__ == "__main__":
    PASTA_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    notebook_00()
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    notebook_05()
    notebook_06()
