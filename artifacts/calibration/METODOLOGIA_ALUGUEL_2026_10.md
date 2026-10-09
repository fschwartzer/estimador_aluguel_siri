# Calibração não residencial — outubro de 2026

## Decompor

Fonte: `SIRI_pesquisa_aluguel-09.10.2026 10.47.31.xlsx`, cópia binária local identificada pelo SHA-256 no JSON. São 149.842 linhas, de 04/10/2019 a 01/09/2026, todas classificadas como oferta de aluguel. O alvo é aluguel mensal anunciado por m². Nenhum preço contratado está disponível. A coleta não é a data do negócio nem a data da avaliação.

A normalização de finalidade e área é a mesma do aplicativo. Somente finalidades não residenciais entram na nova pesquisa. Salas e lojas usam área privativa; galpões usam construída; imóveis comerciais usam construída pela cobertura histórica de treinamento (área privativa em apenas 58 de 6.374 registros totais). Terrenos e glebas usam área de lote e testada. O perfil especializado só é aplicado ao denominador calibrado; outra área mantém o perfil legado.

## Resolver

Hipótese técnica: proximidade física e geográfica entre imóveis da mesma finalidade aproxima o aluguel anunciado. A recência controla a influência relativa de coletas antigas; não atualiza preços por um índice e não aplica desconto oferta/transação.

Identidades são componentes conectados por inscrição SIAT, URL e origem+código do anúncio, sem utilizar preço. Cada janela mantém o registro mais recente por componente. Todas as identidades elegíveis da janela, inclusive as não amostradas, são retiradas do treino. Treinamento contém somente datas anteriores à janela. A normalização robusta, seleção espacial, filtros de cauda e winsorização são refeitos exclusivamente nos comparáveis de treinamento. Alvos de validação não são cortados pelo piso de preço, evitando uma melhora artificial por remoção de aluguéis baixos.

Desenvolvimento: março/2026, até 100 imóveis por finalidade; confirmação de desenvolvimento: setembro/2025, até 100. Validação posterior: agosto/setembro/2026, até 200. Amostragem aleatória com semente 20261009. Exigem-se pelo menos 35 alvos e 80 linhas históricas no desenvolvimento para calibrar. Esses mínimos são controles operacionais, não prova de precisão inferencial.

Busca em etapas: K 6–15/12–25/20–40, vizinhos efetivos 5/10/16, peso físico 0,20/0,50/0,80, potência 0,35/0,75, MAD 1,50/3,00 e filtro local ligado/desligado. Compara pisos zero/metade/inteiro do legado, teto individual 0,15/0,25 e bônus de edifício 1/2. Nos melhores perfis e no legado, compara recência desligada e meias-vidas 90/180/365/730 dias, com fator mínimo 0,10/0,35/0,65. A busca é finita e escalonada; não há garantia de ótimo global.

Score = MdAPE + 0,15×P90 APE + 0,50×|mediana da razão−1| + 0,20×COD/100 + 0,30×|PRD−1|. Na confirmação, o score é normalizado pelo legado de cada período. A recência exige ganho agregado de pelo menos 1% sobre a melhor opção sem recência no desenvolvimento. O candidato só substitui o legado quando a validação posterior tem ≥30 previsões, nenhuma falha, score pelo menos 1% menor e MdAPE não maior. A validação posterior é porta de promoção, portanto suas métricas não devem ser apresentadas como teste final prospectivo intocado. Não se escolhe uma alternativa diferente procurando o melhor resultado posterior.

## Verificar

Razão = aluguel estimado/aluguel observado. COD = 100×média dos desvios absolutos das razões em torno de sua mediana/mediana. PRD principal = média simples das razões/(soma dos aluguéis totais estimados/soma dos aluguéis totais observados). O PRD de valores unitários também é registrado, com outro significado de ponderação, e pode ser muito sensível a erros de área. As métricas abaixo são da validação temporal posterior, nunca do treino.

| Finalidade | n | MdAPE legado | MdAPE aplicado | COD | PRD | Mediana das razões |
|---|---:|---:|---:|---:|---:|---:|
| GALPÃO / DEPÓSITO | 80 | 29.0% | 25.5% | 31.94 | 1.126 | 0.813 |
| IMÓVEL COMERCIAL | 134 | 26.1% | 26.1% | 39.54 | 1.204 | 0.907 |
| LOJA | 200 | 37.2% | 37.2% | 62.95 | 1.335 | 0.858 |
| SALA COMERCIAL | 200 | 26.6% | 24.6% | 36.79 | 0.962 | 0.915 |
| TERRENO | 38 | 40.7% | 40.7% | 50.72 | 1.738 | 0.860 |

Intervalos exploratórios de 95% para a mudança de MdAPE (novo menos legado), em pontos percentuais:

- GALPÃO / DEPÓSITO: -9.41 a -0.19 p.p.
- SALA COMERCIAL: -3.54 a +0.17 p.p.

O intervalo das salas inclui zero: a melhora pontual não demonstra ganho estatístico estável. Confiança moderada no resultado de galpões e baixa a moderada nas salas; os intervalos não corrigem a seleção dos perfis.

Nas salas, o P90 APE passa de 63.11% para 65.45%, o PRD de 0.998 para 0.962 e o erro absoluto médio do aluguel total de R$ 1068.52 para R$ 1187.48. O novo perfil melhora o score composto e a mediana do erro, mas piora a cauda e o erro médio em reais; essa troca exige acompanhamento operacional.

Os segmentos mantidos no legado não passaram a porta de promoção. Isso evita aplicar candidatos cuja melhora em desenvolvimento não se sustentou no período posterior.

A comparação aplica os parâmetros legados e candidatos à mesma data de referência do alvo. Ela compara perfis sob uma regra temporal comum; não reproduz todos os detalhes da versão anterior, cuja referência de recência era a última coleta dos comparáveis.

Os atributos cadastrais SIAT da pesquisa não possuem histórico de vigência verificável. O corte temporal impede preços/coletas futuros e identidades compartilhadas no treino, mas não comprova que todo atributo cadastral já era conhecido na data histórica. Há 111 registros brutos com ano de construção posterior à coleta; nos perfis aplicados, nenhum permanece em salas, galpões, lojas ou terrenos após preparo, e dois permanecem no treino de imóveis comerciais (perfil legado). Essa inconsistência e a falta de versionamento cadastral limitam a interpretação do backtesting como simulação histórica estrita.

O JSON inclui candidatos rejeitados, métricas dos dois períodos de desenvolvimento, ablação da recência no candidato, decis de aluguel total, bairros com pelo menos cinco alvos, holdout espacial em blocos de 1 km e intervalos exploratórios por bootstrap pareado (2.000 reamostragens). O holdout remove o bloco do alvo antes de qualquer preparo, sem buffer; 30 imóveis por finalidade. Distâncias: aproximação equiretangular local sobre EPSG:4326, em km. O bônus pleno usa 30 m, com transição até 50 m; coordenadas coincidentes não provam identidade de edifício.

## Sintetizar

| Finalidade | K | Peso físico/geográfico | Recência | Piso (R$/m²/mês) | Situação |
|---|---|---|---|---:|---|
| GALPÃO / DEPÓSITO | 12–25 | 0.80/0.20 | 730 dias; mínimo 0.10 | 0.70 | novo perfil confirmado |
| IMÓVEL COMERCIAL | 12–25 | 0.35/0.65 | 180 dias; mínimo 0.35 | 3.00 | legado mantido |
| LOJA | 12–25 | 0.35/0.65 | 180 dias; mínimo 0.35 | 5.50 | legado mantido |
| SALA COMERCIAL | 12–25 | 0.20/0.80 | 180 dias; mínimo 0.10 | 4.00 | novo perfil confirmado |
| TERRENO | 12–25 | 0.35/0.65 | 180 dias; mínimo 0.35 | 0.15 | legado mantido |

Finalidades sem suporte: GARAGEM / VAGA (2 alvos no ajuste; 634 linhas históricas); GLEBA (5 alvos no ajuste; 265 linhas históricas); IMÓVEL ESPECIAL (3 alvos no ajuste; 79 linhas históricas); LOJA EM GALERIA (29 alvos no ajuste; 14 linhas históricas); LOJA EM SHOPPING (1 alvos no ajuste; 0 linhas históricas). Não são vendidas como recalibradas.

O PDF usa VU mensal e VU robusto, com endereço, referência de linha, data da coleta, idade, peso bruto de distância, fator de recência, bônus e peso final. O Excel conserva componentes, endereços, diagnósticos e exclusões. Soma dos pesos = 1; soma de peso final×VU robusto = VU estimado. A coluna interna anterior é preservada no core para compatibilidade com notebooks, sem exposição como VU ajustado nos relatórios.

## Ressalvas

Há overfitting possível pela busca de muitos perfis, amostras limitadas e snapshots correlacionados. A confiança é baixa para segmentos raros. O bootstrap não incorpora toda a incerteza de seleção, dependência espacial ou erros de identificação. Uma validação prospectiva com novas coletas continua necessária.

Viés: preços pedidos podem diferir dos contratados; duplicados sem identidade compartilhada podem persistir; cobertura espacial, padrão, conservação e conflitos de cadastro não são uniformes. Pisos e exclusão da cauda inferior podem elevar razões e agravar regressividade. COD e PRD observados não demonstram uniformidade satisfatória; comparar decis e regiões é indispensável. PRD acima de 1 sugere maior razão relativa em aluguéis de menor valor, mas a interpretação requer dados válidos e controle de heterogeneidade.

O alvo não é transformado em log. Logs de VU são usados somente nos filtros robustos de treinamento; não há retransposição exponencial de previsão nem smearing. Endereços cadastrados podem não ser o endereço efetivo do anúncio. O score de confiança do relatório é heurístico, sem interpretação de probabilidade ou intervalo estatístico.

Reprodução:

```powershell
python -m pip install -r requirements-dev.txt
python scripts/inspect_rental_source.py '<pesquisa.xlsx>' tmp/source.pkl --enriched tmp/enriched.pkl
python scripts/calibrate_rentals.py --cache tmp/enriched.pkl --source '<pesquisa.xlsx>' --limit 100
python scripts/validate_spatial_rentals.py --cache tmp/enriched.pkl
python scripts/summarize_calibration.py
python -m unittest discover -s tests -v
```

Dados individuais e exemplos de auditoria ficam fora do Git. O SHA-256 identifica a cópia usada; o JSON e a tabela CSV agregada são versionados.
