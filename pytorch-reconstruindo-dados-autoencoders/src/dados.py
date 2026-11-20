"""Caminhos do projeto, obtenção da base e carregamento dos conjuntos por ano.

Base: Gas Turbine CO and NOx Emission Data Set (UCI 551, CC BY 4.0).
DOI: https://doi.org/10.24432/C5WC95
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- caminhos (relativos à raiz do projeto)
RAIZ = Path(__file__).resolve().parents[1]
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_PROCESSADO = RAIZ / "data" / "processado"
PASTA_EXEMPLOS = RAIZ / "data" / "exemplos"
PASTA_ARTEFATOS = RAIZ / "artefatos"
PASTA_MODELOS = PASTA_ARTEFATOS / "modelos"
PASTA_RESULTADOS = RAIZ / "resultados"

# ---------------------------------------------------------------- fonte oficial
URL_UCI = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
ANOS = (2011, 2012, 2013, 2014, 2015)
SHA256_ARQUIVOS = {
    "gt_2011.csv": "d87ceef9aa59533cc7d924d10de241b1b06ecd11f9b26bab59191ea0f8a76b9a",
    "gt_2012.csv": "be54b9d0e1a7de40c55d32fa489e75de892b000c066b5a09f09a19124ee29100",
    "gt_2013.csv": "13c437bb440ec2045bd12057e6654c41dd4107a661eac16ba2e878e897a08f9e",
    "gt_2014.csv": "c2a03c92c9c3207aad0c6be7de8d9b5b4bfa4720ad0efb2c1f21b6cec4d3f3fa",
    "gt_2015.csv": "9b08f35fde0d4b138232a605db4093c2b8bf9d6757e6f1fbd9534ad616c13591",
}

# ---------------------------------------------------------------- variáveis
COLUNAS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]
DESCRICAO = {
    "AT": ("Temperatura ambiente", "°C"),
    "AP": ("Pressão ambiente", "mbar"),
    "AH": ("Umidade ambiente", "%"),
    "AFDP": ("Diferença de pressão no filtro de ar", "mbar"),
    "GTEP": ("Pressão de exaustão da turbina", "mbar"),
    "TIT": ("Temperatura de entrada da turbina", "°C"),
    "TAT": ("Temperatura após a turbina", "°C"),
    "TEY": ("Energia gerada", "MWh"),
    "CDP": ("Pressão de descarga do compressor", "mbar"),
    "CO": ("Monóxido de carbono", "mg/m³"),
    "NOX": ("Óxidos de nitrogênio", "mg/m³"),
}

# ---------------------------------------------------------------- divisão por ano
ANOS_POR_CONJUNTO = {"treino": (2011, 2012), "validacao": (2013,), "teste": (2014, 2015)}


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def baixar_base(pasta=PASTA_RAW):
    """Baixa os cinco CSV anuais da UCI, se ainda não existirem, e confere o SHA-256."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    faltando = [nome for nome in SHA256_ARQUIVOS if not (pasta / nome).exists()]
    if faltando:
        with urllib.request.urlopen(URL_UCI, timeout=60) as resposta:
            conteudo = resposta.read()
        with zipfile.ZipFile(io.BytesIO(conteudo)) as arquivo_zip:
            for nome in faltando:
                (pasta / nome).write_bytes(arquivo_zip.read(nome))
    for nome, esperado in SHA256_ARQUIVOS.items():
        if sha256(pasta / nome) != esperado:
            raise ValueError(f"{nome}: SHA-256 diferente do arquivo publicado pela UCI")
    return faltando


def ler_arquivos_anuais(pasta=PASTA_RAW):
    """Lê gt_2011.csv ... gt_2015.csv. O ano vem do nome do arquivo e só separa os conjuntos."""
    partes = [pd.read_csv(Path(pasta) / f"gt_{ano}.csv").assign(ano=ano) for ano in ANOS]
    return pd.concat(partes, ignore_index=True)


def carregar_conjunto(nome, liberar_teste=False):
    """Carrega treino, validacao ou teste, separados pelo notebook 00.

    O teste fica reservado para a avaliação final e só é carregado com liberar_teste=True.
    """
    if nome not in ANOS_POR_CONJUNTO:
        raise ValueError(f"conjunto desconhecido: {nome}")
    if nome == "teste" and not liberar_teste:
        raise PermissionError("O conjunto de teste é reservado para a avaliação final (liberar_teste=True).")
    caminho = PASTA_PROCESSADO / f"{nome}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Execute primeiro o notebook 00_preparacao_base.")
    return pd.read_csv(caminho)
