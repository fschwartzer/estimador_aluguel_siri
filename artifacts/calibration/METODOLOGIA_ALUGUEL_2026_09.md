# Calibração SIRI Aluguéis — setembro de 2026

## Base e unidade

- Fonte: `SIRI_pesquisa_aluguel-04.09.2026 11.09.10.xlsx`.
- 80.996 linhas; período de 05/09/2023 a 01/09/2026.
- 80.613 linhas com valor e área válidos; 70.919 com coordenadas válidas.
- Alvo: valor mensal anunciado por metro quadrado (R$/m²/mês).
- 21 linhas cuja URL indicava venda foram excluídas preventivamente.
- O arquivo-fonte não é redistribuído neste repositório.

## Prevenção de vazamento

A unidade de agrupamento foi anúncio + finalidade. Em cada corte, manteve-se
uma observação por anúncio e anúncios presentes na validação foram retirados do
treino. Nenhuma estatística do período de validação foi usada para normalizar
atributos ou construir comparáveis de treino.

- Ajuste: treino de 01/05/2025 a 28/02/2026; validação no snapshot de março/2026.
- Confirmação: treino de 01/05/2025 a 31/07/2026; validação em agosto/setembro/2026.
- Grade: 48 combinações de K, pesos físico/geográfico, potência da distância e
  limiar MAD.

A base contém snapshots trimestrais. Portanto, a validação mede estabilidade
temporal entre snapshots, mas não elimina toda dependência entre anúncios de
um mesmo imóvel publicados com identificadores distintos.

## Regra de seleção

O score combinou MdAPE, P90 do erro percentual absoluto, desvio da mediana da
razão estimado/observado, COD e PRD. Um perfil por finalidade só foi aceito
quando o ganho apareceu tanto no ajuste quanto na confirmação e não implicou
deterioração material de cauda ou viés.

Perfil global: K 12–25; mínimo de 10 vizinhos efetivos; peso físico 0,35;
geográfico 0,65; potência 0,75; peso individual máximo 0,25; MAD 1,50.

Perfil específico aceito:

- `GALPÃO / DEPÓSITO`: pesos físico/geográfico 0,65/0,35; demais parâmetros
  iguais ao global. O MdAPE caiu 4,2% no ajuste e 6,5% na confirmação.

O candidato de apartamentos foi rejeitado: reduziu o MdAPE no ajuste, mas o
aumentou 1,5% na confirmação e piorou cauda e viés. As demais finalidades usam
o perfil global por falta de ganho temporal estável.

## Pisos mensais por finalidade

| Finalidade | R$/m² |
|---|---:|
| Apartamento | 7,50 |
| Casa/residência | 4,00 |
| Cobertura | 7,00 |
| Flat/apart-hotel | 10,00 |
| Galpão/depósito | 0,70 |
| Garagem/vaga | 3,00 |
| Garagem residencial | 2,00 |
| Garagem não residencial | 6,00 |
| Gleba | 0,15 |
| Hotel | 0,40 |
| Imóvel comercial | 3,00 |
| Imóvel especial | 8,50 |
| Loja | 5,50 |
| Loja em galeria | 3,50 |
| Loja em shopping | 5,50 |
| Sala comercial | 4,00 |
| Terreno | 0,15 |

Os pisos são controles inferiores conservadores derivados da cauda da própria
finalidade. O filtro robusto posterior continua responsável por valores
atípicos contextuais; não foi imposto teto fixo, para não apagar submercados de
alto padrão de forma silenciosa.

## Resultado e ressalvas

Confirmação global, nove finalidades e `n=623`: MdAPE macro 26,2%; P90 APE
91,7%; desvio absoluto da mediana da razão 8,1 p.p.; COD macro 36,2; desvio
absoluto do PRD 0,248.

Hipótese técnica: proximidade física e espacial aproxima o aluguel unitário
pedido. Riscos: preço pedido difere do contratado; omissão de conservação,
andar, vagas e padrão; duplicação entre portais; cobertura espacial desigual;
e regressividade por faixa de valor. O COD e o PRD agregados ainda são altos,
logo a estimativa deve ser revisada com os comparáveis e estratificada antes de
uso em avaliação em massa.
