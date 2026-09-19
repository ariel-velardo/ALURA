import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS_AULA = [
    RAIZ / "notebooks" / f"0{numero}_{nome}.ipynb"
    for numero, nome in [
        (1, "problema_benchmark"),
        (2, "ensembles"),
        (3, "otimizacao"),
        (4, "desbalanceamento_threshold"),
        (5, "semisupervisionado_importancia"),
        (6, "modelo_final"),
    ]
]

HELPERS_REMOVIDOS = {
    "PARAMETROS_REFERENCIA",
    "avaliar_probabilidades",
    "criar_modelo_floresta",
    "criar_modelo_gradiente",
    "criar_modelo_logistico",
    "criar_modelo_xgboost",
    "metricas_limiares",
    "separar_dados",
}

CELULAS_VAZIAS_ESPERADAS = {
    "01_problema_benchmark.ipynb": [4, 7, 10, 12],
    "02_ensembles.ipynb": [4, 6, 8, 10],
    "03_otimizacao.ipynb": [4, 6, 8, 12],
    "04_desbalanceamento_threshold.ipynb": [4, 6, 8, 10, 14],
    "05_semisupervisionado_importancia.ipynb": [4, 8, 9, 12, 15],
    "06_modelo_final.ipynb": [4, 8, 12],
}

CONCEITOS_POR_NOTEBOOK = {
    "01_problema_benchmark.ipynb": {
        "train_test_split",
        "ColumnTransformer",
        "OneHotEncoder",
        "StandardScaler",
        "Pipeline",
        "LogisticRegression",
        "RandomForestClassifier",
        "average_precision_score",
    },
    "02_ensembles.ipynb": {
        "train_test_split",
        "ColumnTransformer",
        "OneHotEncoder",
        "Pipeline",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "average_precision_score",
    },
    "03_otimizacao.ipynb": {
        "train_test_split",
        "GradientBoostingClassifier",
        "average_precision_score",
        "optuna",
    },
    "04_desbalanceamento_threshold.ipynb": {
        "train_test_split",
        "GradientBoostingClassifier",
        "compute_sample_weight",
        "SMOTENC",
        "average_precision_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "confusion_matrix",
    },
    "05_semisupervisionado_importancia.ipynb": {
        "train_test_split",
        "SelfTrainingClassifier",
        "ColumnTransformer",
        "OneHotEncoder",
        "StandardScaler",
        "LogisticRegression",
        "Pipeline",
        "GradientBoostingClassifier",
        "average_precision_score",
        "permutation_importance",
    },
    "06_modelo_final.ipynb": {
        "train_test_split",
        "GradientBoostingClassifier",
        "ColumnTransformer",
        "OneHotEncoder",
        "Pipeline",
        "average_precision_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "confusion_matrix",
        "joblib",
    },
}


def carregar_notebook(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def fontes(caderno):
    return "\n".join(
        "".join(celula.get("source", []))
        for celula in caderno["cells"]
    )


class TestRefatoracaoPedagogica(unittest.TestCase):
    def test_kernels_sao_genericos_e_portateis(self):
        caminhos = sorted((RAIZ / "notebooks").glob("0[0-6]_*.ipynb"))
        esperado = {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        }
        for caminho in caminhos:
            kernelspec = carregar_notebook(caminho)["metadata"]["kernelspec"]
            with self.subTest(caderno=caminho.name):
                self.assertEqual(esperado, kernelspec)

    def test_material_do_aluno_nao_revela_bastidores_ou_regras_magicas(self):
        termos_proibidos = [
            "prova técnica",
            "decisão já validada",
            "ganho esperado",
            "min_samples_leaf",
            "0.005",
        ]
        for caminho in NOTEBOOKS_AULA:
            texto = fontes(carregar_notebook(caminho)).lower()
            for termo in termos_proibidos:
                with self.subTest(caderno=caminho.name, termo=termo):
                    self.assertNotIn(termo, texto)

    def test_notebook_01_nao_importa_metricas_nao_utilizadas(self):
        caminho = RAIZ / "notebooks" / "01_problema_benchmark.ipynb"
        caderno = carregar_notebook(caminho)
        codigo = "\n".join(
            "".join(celula.get("source", []))
            for celula in caderno["cells"]
            if celula["cell_type"] == "code"
        )
        arvore = ast.parse(codigo)
        nomes_importados = {
            alias.name
            for no in ast.walk(arvore)
            if isinstance(no, (ast.Import, ast.ImportFrom))
            for alias in no.names
        }
        for nome in [
            "confusion_matrix",
            "precision_score",
            "recall_score",
            "f1_score",
        ]:
            with self.subTest(nome=nome):
                self.assertNotIn(nome, nomes_importados)

    def test_notebook_04_disponibiliza_todas_as_categorias_para_smotenc(self):
        caminho = RAIZ / "notebooks" / "04_desbalanceamento_threshold.ipynb"
        texto = fontes(carregar_notebook(caminho))
        self.assertIn("COLUNAS_NOMINAIS", texto)
        self.assertIn("COLUNAS_STATUS", texto)
        self.assertIn("COLUNAS_NOMINAIS + COLUNAS_STATUS", texto)

    def test_funcoes_visuais_sem_uso_foram_removidas(self):
        texto = (RAIZ / "src" / "visual_utils.py").read_text(encoding="utf-8")
        self.assertNotIn("grafico_barras_padrao", texto)
        self.assertNotIn("grafico_boxplot_padrao", texto)

    def test_auxiliares_nao_importa_bibliotecas_de_modelagem(self):
        caminho = RAIZ / "src" / "auxiliares.py"
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        modulos = {
            no.module.split(".")[0]
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom) and no.module
        }
        modulos.update(
            alias.name.split(".")[0]
            for no in ast.walk(arvore)
            if isinstance(no, ast.Import)
            for alias in no.names
        )
        self.assertTrue(modulos.isdisjoint({"numpy", "sklearn", "xgboost", "imblearn", "optuna"}))

    def test_visual_utils_nao_calcula_metricas(self):
        caminho = RAIZ / "src" / "visual_utils.py"
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        importacoes_sklearn = [
            no
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom)
            and no.module
            and no.module.startswith("sklearn")
        ]
        self.assertEqual([], importacoes_sklearn)

    def test_helpers_removidos_nao_sao_exportados_nem_usados(self):
        caminhos = [RAIZ / "src" / "auxiliares.py", RAIZ / "src" / "__init__.py", *NOTEBOOKS_AULA]
        for caminho in caminhos:
            texto = caminho.read_text(encoding="utf-8")
            for nome in HELPERS_REMOVIDOS:
                with self.subTest(caminho=caminho.name, nome=nome):
                    self.assertNotIn(nome, texto)

    def test_conceitos_de_ml_estao_visiveis_nos_notebooks(self):
        for caminho in NOTEBOOKS_AULA:
            texto = fontes(carregar_notebook(caminho))
            for conceito in CONCEITOS_POR_NOTEBOOK[caminho.name]:
                with self.subTest(caderno=caminho.name, conceito=conceito):
                    self.assertIn(conceito, texto)

    def test_celulas_de_codigo_ao_vivo_continuam_vazias(self):
        for caminho in NOTEBOOKS_AULA:
            caderno = carregar_notebook(caminho)
            vazias = [
                indice
                for indice, celula in enumerate(caderno["cells"])
                if celula["cell_type"] == "code"
                and not "".join(celula.get("source", [])).strip()
            ]
            self.assertEqual(CELULAS_VAZIAS_ESPERADAS[caminho.name], vazias)

    def test_aulas_dependentes_falham_sem_parametros_otimizados(self):
        for caminho in NOTEBOOKS_AULA[3:]:
            texto = fontes(carregar_notebook(caminho))
            with self.subTest(caderno=caminho.name):
                self.assertIn("if not caminho_parametros.exists()", texto)
                self.assertIn("FileNotFoundError", texto)
                self.assertIn("03_otimizacao.ipynb", texto)

    def test_nomenclatura_pedagogica_usa_average_precision(self):
        caminhos = [*NOTEBOOKS_AULA, RAIZ / "src" / "visual_utils.py"]
        for caminho in caminhos:
            with self.subTest(caminho=caminho.name):
                self.assertNotIn("PR-AUC", caminho.read_text(encoding="utf-8"))

    def test_gerador_confere_que_notebooks_estao_atualizados(self):
        processo = subprocess.run(
            [sys.executable, str(RAIZ / "gerar_notebooks.py"), "--check"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, processo.returncode, processo.stdout + processo.stderr)


if __name__ == "__main__":
    unittest.main()
