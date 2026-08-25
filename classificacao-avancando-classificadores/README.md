# Classificação: avançando em classificadores

Projeto educacional da Alura sobre classificação, ensembles, otimização, desbalanceamento, escolha de limiar, aprendizado semissupervisionado e importância de variáveis.

O problema é prever a inadimplência de clientes de cartão de crédito com o conjunto [Default of Credit Card Clients, da UCI](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients). A base bruta é preservada; os notebooks trabalham com uma cópia cujas colunas foram traduzidas para português.

## Estrutura

- `data/raw`: base original, sem alterações;
- `data/processed`: base preparada para o curso;
- `notebooks`: preparação, desenvolvimento e avaliação dos modelos;
- `models`: modelo final e artefatos serializados;
- `src`: funções auxiliares compartilhadas;
- `docs`: dicionário de dados e materiais complementares;
- `app.py`: aplicação local em Streamlit.

## Ambiente local

No PowerShell, crie e use o ambiente virtual do próprio projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nenhuma instalação é feita automaticamente pelo projeto.

## Execução

Com o ambiente `.venv` ativado, abra os notebooks:

```powershell
python -m jupyter notebook
```

Execute primeiro `notebooks/00_preparacao_base.ipynb`; os demais notebooks usam o arquivo preparado por ele. Para iniciar a aplicação depois de gerar o modelo final:

```powershell
python -m streamlit run app.py
```

O mapeamento completo das 25 colunas está em `docs/dicionario_dados.md`.
