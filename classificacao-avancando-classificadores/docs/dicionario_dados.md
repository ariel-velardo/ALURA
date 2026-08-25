# Dicionário de dados

Os nomes foram traduzidos sem alterar os valores ou os códigos da base original. As descrições seguem a documentação do conjunto [Default of Credit Card Clients, da UCI](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients). Alguns códigos não são definidos pela documentação oficial e permanecem inalterados: `-2` e `0` em `PAY_*`, `0`, `5` e `6` em `EDUCATION` e `0` em `MARRIAGE`.

| Nome original | Nome no curso | Descrição curta |
|---|---|---|
| `ID` | `id_cliente` | Identificador do cliente. |
| `LIMIT_BAL` | `limite_credito` | Limite de crédito concedido, em NT$. |
| `SEX` | `sexo` | Código de sexo informado na base. |
| `EDUCATION` | `escolaridade` | Código do nível de escolaridade. |
| `MARRIAGE` | `estado_civil` | Código do estado civil. |
| `AGE` | `idade` | Idade em anos. |
| `PAY_0` | `status_pagamento_set` | Status de pagamento em setembro de 2005. |
| `PAY_2` | `status_pagamento_ago` | Status de pagamento em agosto de 2005. |
| `PAY_3` | `status_pagamento_jul` | Status de pagamento em julho de 2005. |
| `PAY_4` | `status_pagamento_jun` | Status de pagamento em junho de 2005. |
| `PAY_5` | `status_pagamento_mai` | Status de pagamento em maio de 2005. |
| `PAY_6` | `status_pagamento_abr` | Status de pagamento em abril de 2005. |
| `BILL_AMT1` | `valor_fatura_set` | Valor da fatura em setembro de 2005, em NT$. |
| `BILL_AMT2` | `valor_fatura_ago` | Valor da fatura em agosto de 2005, em NT$. |
| `BILL_AMT3` | `valor_fatura_jul` | Valor da fatura em julho de 2005, em NT$. |
| `BILL_AMT4` | `valor_fatura_jun` | Valor da fatura em junho de 2005, em NT$. |
| `BILL_AMT5` | `valor_fatura_mai` | Valor da fatura em maio de 2005, em NT$. |
| `BILL_AMT6` | `valor_fatura_abr` | Valor da fatura em abril de 2005, em NT$. |
| `PAY_AMT1` | `valor_pago_set` | Valor pago em setembro de 2005, em NT$. |
| `PAY_AMT2` | `valor_pago_ago` | Valor pago em agosto de 2005, em NT$. |
| `PAY_AMT3` | `valor_pago_jul` | Valor pago em julho de 2005, em NT$. |
| `PAY_AMT4` | `valor_pago_jun` | Valor pago em junho de 2005, em NT$. |
| `PAY_AMT5` | `valor_pago_mai` | Valor pago em maio de 2005, em NT$. |
| `PAY_AMT6` | `valor_pago_abr` | Valor pago em abril de 2005, em NT$. |
| `default.payment.next.month` | `inadimplente` | Inadimplência no mês seguinte: 1 sim, 0 não. |
