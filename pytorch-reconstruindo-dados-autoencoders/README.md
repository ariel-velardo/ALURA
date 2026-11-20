# Recuperação de leituras de sensores com PyTorch

Sensores de uma turbina a gás falham, entram em manutenção ou perdem registros. Quando uma leitura
falta, ainda temos as outras leituras da mesma hora. Este projeto investiga se uma rede neural do
tipo autoencoder consegue estimar a leitura perdida a partir das demais, e como ela se compara a
métodos mais simples de imputação.

## Base de dados

**Gas Turbine CO and NOx Emission Data Set** — UCI Machine Learning Repository (2019).
https://archive.ics.uci.edu/dataset/551 · DOI [10.24432/C5WC95](https://doi.org/10.24432/C5WC95) ·
licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

36.733 registros horários de 11 sensores, de 2011 a 2015: condições do ambiente (temperatura,
pressão, umidade), funcionamento da turbina (pressões e temperaturas), energia gerada e emissões
de CO e NOx. A base não tem valores ausentes; por isso as falhas são simuladas escondendo leituras de
registros completos, o que permite comparar cada valor recuperado com o valor real.

| Conjunto | Anos | Uso |
|---|---|---|
| Treino | 2011–2012 | ajuste dos modelos e do pré-processamento |
| Validação | 2013 | comparação e escolha das configurações |
| Teste | 2014–2015 | avaliação final, depois de todas as escolhas |

O ano só separa os conjuntos e não entra nos modelos.

## Abordagem

1. Explorar os sensores e simular leituras perdidas.
2. Construir em PyTorch um autoencoder denso: um encoder que resume o registro em poucas dimensões
   e um decoder que reconstrói as 11 leituras a partir desse resumo.
3. Investigar o tamanho do espaço latente e a diferença entre reconstruir e recuperar.
4. Treinar o modelo com leituras escondidas (denoising) e comparar.
5. Comparar com mediana, KNN e regressão iterativa nas mesmas posições escondidas.
6. Avaliar a configuração escolhida no teste e usá-la pelo terminal com registros novos.

## Organização

```
├── data/raw/          CSV originais da UCI (gt_2011.csv … gt_2015.csv)
├── data/processado/   treino, validação e teste separados por ano
├── notebooks/         00_preparacao_base e as etapas da investigação
├── src/               código reutilizável
├── artefatos/         arquivos gerados pelos notebooks
└── testes/            testes automatizados (pytest)
```

## Instalação

Windows (PowerShell), dentro da pasta do projeto:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Se o PowerShell recusar a ativação com a mensagem de que a execução de scripts foi desabilitada,
libere scripts só nesta janela e ative de novo (a configuração do computador não muda):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Linux ou macOS: `python3 -m venv .venv` e `source .venv/bin/activate`, depois os mesmos comandos `pip`.
Testado com Python 3.10, 3.11 e 3.13.

Todo o projeto roda em CPU. Os caminhos são resolvidos a partir da raiz do projeto, então ele pode
ficar em qualquer pasta.

## Execução

Execute os notebooks em ordem, começando por `00_preparacao_base.ipynb`, que baixa os dados da UCI
(se ainda não estiverem em `data/raw/`), confere sua integridade e separa os conjuntos.

```powershell
python -m pytest testes
```

## Atribuição

Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). UCI Machine Learning Repository.
https://doi.org/10.24432/C5WC95. Licença CC BY 4.0. Os arquivos em `data/raw/` são redistribuídos
sem alteração.
