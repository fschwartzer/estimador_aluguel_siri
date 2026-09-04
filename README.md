![SIRI Aluguéis](./static/siri_alugueis_header.png)

# SIRI Aluguéis

Aplicativo Streamlit para estimar o aluguel mensal de imóveis por K-vizinhos
mais próximos (KNN), com comparáveis exclusivamente de ofertas de locação.
É um produto independente derivado do `estimador_knn_siri`, com identidade
visual vermelha e calibração própria para o mercado de aluguéis.

## Regra de mercado

O aplicativo:

1. normaliza o schema SIRI ou de planilhas genéricas;
2. mantém somente registros classificados como `Oferta Aluguel`;
3. rejeita ofertas de venda, Guias ITBI, valores/áreas inválidos e conflitos
   tipológicos;
4. não aplica fator de desconto: oferta de aluguel permanece no valor
   anunciado;
5. aplica pisos de valor unitário por finalidade e filtros robustos auditáveis;
6. seleciona comparáveis por atributos físicos e distância geográfica;
7. limita concentração de peso e reduz influência de extremos locais;
8. exporta comparáveis, diagnósticos, exclusões e alertas em Excel/PDF.

Bases genéricas sem coluna de natureza são assumidas como ofertas de aluguel,
e essa hipótese é registrada. Quando a natureza está explícita como venda ou
ITBI, o registro não entra na amostra.

## Parâmetros calibrados

A pesquisa `SIRI_pesquisa_aluguel-04.09.2026 11.09.10.xlsx` contém 80.996
registros entre setembro de 2023 e setembro de 2026. A calibração usou cortes
temporais e impediu que o mesmo anúncio aparecesse simultaneamente no treino e
na validação.

Perfil global:

| Parâmetro | Valor |
|---|---:|
| K mínimo / máximo | 12 / 25 |
| vizinhos efetivos mínimos | 10 |
| peso físico / geográfico | 0,35 / 0,65 |
| potência da distância | 0,75 |
| peso individual máximo | 0,25 |
| limiar MAD robusto | 1,50 |

Galpões/depósitos usam um perfil específico, com peso físico/geográfico de
`0,65 / 0,35`. As demais finalidades usam o perfil global; candidatos
específicos que não confirmaram ganho fora do período de ajuste foram
rejeitados para reduzir overfitting.

Os pisos de valor unitário mensal (R$/m²) e as evidências completas estão em
[`artifacts/calibration/METODOLOGIA_ALUGUEL_2026_09.md`](./artifacts/calibration/METODOLOGIA_ALUGUEL_2026_09.md).

## Métricas fora do período de ajuste

Na confirmação temporal agregada em nove finalidades (`n=623`), o perfil
global obteve MdAPE macro de 26,2%, P90 APE de 91,7%, mediana da razão
estimado/observado com desvio absoluto de 8,1 p.p., COD macro de 36,2 e desvio
absoluto do PRD de 0,248.

Esses níveis de dispersão variam materialmente por finalidade. O aplicativo é
instrumento de apoio e auditoria de comparáveis, não substitui análise técnica.
COD e PRD devem ser acompanhados por finalidade, faixa de valor, período e
região; o risco de regressividade permanece central.

## Execução local

```powershell
git clone https://github.com/fschwartzer/estimador_aluguel_siri.git
cd estimador_aluguel_siri
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Testes

```powershell
python -m unittest discover -s tests -v
```

## Estrutura principal

```text
app.py                           interface e orquestração
estimador_knn_core_v6120.py      preparação, KNN e diagnósticos
estimador_knn_schema_v6120.py    normalização de schemas e finalidades
siri_alugueis_pdf_report.py      relatório PDF e mapa dos comparáveis
artifacts/calibration/           calibração versionada e auditável
tests/                           testes automatizados
```

As distâncias geográficas são calculadas em quilômetros a partir de
latitude/longitude por Haversine. A calibração não normaliza nem agrega dados
antes da separação temporal e remove do treino anúncios presentes na validação,
evitando vazamento direto. Ainda assim, anúncios são preços pedidos, não
aluguéis contratados, e podem conter dependência espacial ou repetição entre
portais; essa limitação deve acompanhar qualquer uso decisório.
