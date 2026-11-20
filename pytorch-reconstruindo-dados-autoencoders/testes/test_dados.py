"""Colunas da base e divisão por ano."""
import pytest

from src import dados

pytestmark = pytest.mark.skipif(not (dados.PASTA_PROCESSADO / "treino.csv").exists(),
                                reason="execute o notebook 00_preparacao_base antes")


def test_colunas_esperadas():
    base = dados.ler_arquivos_anuais()
    assert list(base.columns) == dados.COLUNAS + ["ano"]
    assert base[dados.COLUNAS].notna().all().all()
    for nome in ("treino", "validacao"):
        assert list(dados.carregar_conjunto(nome).columns) == dados.COLUNAS + ["ano"]


def test_divisao_por_ano():
    treino = dados.carregar_conjunto("treino")
    validacao = dados.carregar_conjunto("validacao")
    teste = dados.carregar_conjunto("teste", liberar_teste=True)
    assert set(treino["ano"]) == {2011, 2012} and len(treino) == 15039
    assert set(validacao["ano"]) == {2013} and len(validacao) == 7152
    assert set(teste["ano"]) == {2014, 2015} and len(teste) == 14542


def test_teste_reservado():
    with pytest.raises(PermissionError):
        dados.carregar_conjunto("teste")
