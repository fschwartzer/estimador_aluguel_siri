# Histórico do estimador_knn_siri até a LITE 1.19.0

## Relatório sintético de inferência - LITE 1.19.0

- adiciona o PDF aos outputs da VERA, preservando a exportação Excel;
- apresenta valor unitário, valor total, COD dos comparáveis, score de
  confiança e a seleção final de comparáveis;
- reproduz avaliando e comparáveis em mapa estático com fundo Claro - CARTO
  Positron e atribuição ao OpenStreetMap/CARTO;
- usa uma grade neutra como contingência quando os tiles externos não estão
  disponíveis;
- define o COD exibido como diagnóstico local da dispersão dos valores
  unitários ajustados, sem apresentá-lo como COD de backtesting.

> Documento arquivado em 14/08/2026. Algumas instruções de implantação abaixo
> descrevem versões antigas e não devem ser usadas para publicar a versão
> atual. Consulte o `README.md` da raiz.

Edição LITE do estimador imobiliário por KNN, preparada para usuários sem
conhecimento estatístico ou de aprendizado de máquina.

## O que o usuário precisa fazer

1. Enviar a planilha Excel.
2. Escolher a finalidade.
3. Confirmar se é terreno/lote ou unidade construída.
4. Informar áreas, testada quando territorial, latitude e longitude.
5. Clicar em **Calcular estimativa**.

## Processamentos automáticos

A aplicação mantém internamente os tratamentos da versão técnica:

- filtro pela finalidade;
- exclusão de ofertas de aluguel;
- deduplicação das ofertas pela data mais recente;
- desconto das ofertas pela mediana de razões pareadas com Guias ITBI;
- K adaptativo;
- mínimo de cinco vizinhos efetivos;
- limite de 30% de peso por comparável;
- potência de distância igual a 1;
- média robusta por mediana e MAD;
- alertas de extrapolação;
- pontuação de confiança;
- exportação dos comparáveis e diagnósticos para Excel.

O backtesting e todos os controles estatísticos foram removidos da interface.

## Parâmetros internos fixos

| Parâmetro | Valor |
|---|---:|
| K inicial | 7 |
| K máximo | 30 |
| Vizinhos efetivos mínimos | 5 |
| Peso dos atributos físicos | 66,67% (2/3) |
| Peso da distância geográfica | 33,33% (1/3) |
| Peso máximo individual | 30% |
| Potência da distância | 1 |
| Limiar robusto | 2,5 MAD |
| Desconto máximo das ofertas | 20% |

## Execução local

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Publique na raiz do repositório:

- `app.py`;
- `knn_valuation.py`;
- `schema_utils.py`;
- `requirements.txt`;
- pasta `.streamlit`.

Defina `app.py` como arquivo principal.


## Alertas do fator de oferta

A edição LITE 1.1 mantém integralmente o cálculo anterior e acrescenta apenas
avisos interpretativos.

### Faixas do desconto

| Desconto empírico | Alerta |
|---:|---|
| 0% | Sem desconto positivo aplicado |
| Acima de 0% até 10% | Elasticidade usual ou moderada |
| Acima de 10% até 15% | Elasticidade relevante |
| Acima de 15% até 20% | Elasticidade elevada |
| Acima de 20% bruto | Alerta forte; o teto de 20% continua sendo aplicado |

O desconto permanece definido pela mediana de
`1 - VU_ITBI / VU_Oferta` em quantis pareados, limitado entre 0% e 20%.

### Composição efetiva da amostra

Os números apresentados nos alertas correspondem aos dados que permanecem
após:

- filtro pela finalidade;
- exclusão de ofertas de aluguel;
- deduplicação das ofertas;
- validação do valor unitário.

| Menor grupo entre ITBI e ofertas | Alerta |
|---:|---|
| Menos de 2 | Insuficiente para calcular o desconto |
| 2 a 9 | Amostra reduzida |
| 10 a 14 | Amostra restrita |
| 15 ou mais | Quantidade mínima operacional atendida |

Quando ambos os grupos possuem ao menos 15 dados, o aplicativo também informa
se existe desequilíbrio superior a 3:1 ou 5:1 entre as quantidades.

Essas faixas são diagnósticas: não bloqueiam nem modificam o cálculo.


## Regra do fator de oferta — LITE 1.2

A regra passa a distinguir claramente dois cenários.

### Amostra composta somente por ofertas

Quando existe ao menos uma oferta válida e nenhuma Guia ITBI válida, aplica-se:

\[
Fator\ de\ oferta = 0,90
\]

ou seja, desconto convencional de 10%.

### Amostra com Guias ITBI e ofertas

Quando existem pelo menos duas Guias ITBI e duas ofertas válidas, o desconto é
calculado pela mediana dos quantis pareados:

\[
d = \operatorname{mediana}
\left(
1-\frac{VU_{ITBI,q}}{VU_{Oferta,q}}
\right)
\]

O resultado empírico é limitado ao intervalo de 0% a 20%. Portanto, 20% não é
um desconto fixo: é exclusivamente o freio superior da razão calculada.

### Amostra mista insuficiente

Quando existe alguma Guia ITBI, mas não há pelo menos dois dados válidos em
cada grupo, o desconto permanece em zero. O fator convencional de 10% é
reservado à situação em que não existe nenhuma Guia ITBI.

Nenhuma outra regra do estimador foi alterada.


## Correção de colunas duplicadas — LITE 1.3

A área de referência também pode ser a área do lote, área privativa ou área
construída. Antes, essa coluna podia entrar duas vezes na tabela dos
comparáveis, causando:

`ValueError: Duplicate column names found`

A versão 1.3 remove repetições antes de montar a tabela e garante nomes únicos
após as renomeações. O cálculo do fator de oferta, o KNN e os valores estimados
não foram alterados.


## Correção e melhoria do mapa — LITE 1.4

O mapa simples foi substituído por uma visualização PyDeck com:

- marcador vermelho para o imóvel avaliando;
- marcadores verdes para os comparáveis;
- tooltip com peso no KNN, distância e coordenadas;
- zoom calculado automaticamente pela dispersão dos pontos;
- conversão de coordenadas com vírgula decimal;
- exclusão informada de coordenadas ausentes ou inválidas;
- mapa simplificado como fallback caso o PyDeck não esteja disponível.

A alteração afeta somente a apresentação. Cálculo do valor, fator de oferta,
seleção dos comparáveis, pesos e tratamentos robustos permanecem inalterados.


## Mapa autossuficiente — LITE 1.5

O componente PyDeck foi removido porque alguns navegadores e ambientes do
Streamlit Cloud reservavam o espaço do mapa, mas não renderizavam o canvas.

A nova visualização é gerada diretamente pelo Matplotlib e não depende de:

- WebGL;
- Mapbox;
- tiles de mapas externos;
- JavaScript de terceiros;
- permissões de rastreamento do navegador.

O avaliando é posicionado na origem. Os comparáveis são representados por seus
deslocamentos aproximados em quilômetros:

- eixo horizontal: leste–oeste;
- eixo vertical: norte–sul;
- tamanho do círculo: peso no KNN;
- número ao lado do ponto: ordem decrescente de peso.

A alteração é exclusivamente visual. Nenhuma regra do KNN ou do fator de
oferta foi modificada.


## Mapa Plotly MapLibre — LITE 1.6

O mapa de proximidade estático foi substituído por um mapa urbano interativo
em Plotly.

### Recursos

- mapa-base OpenStreetMap ou Carto Positron;
- opção interativa sem mapa-base caso a rede bloqueie tiles externos;
- zoom pela roda do mouse;
- navegação por arraste;
- linhas ligando o avaliando aos comparáveis;
- tamanho dos marcadores proporcional ao peso no KNN;
- numeração dos comparáveis por ordem decrescente de peso;
- hover com tipo da informação, peso, distância, valor unitário, área,
  testada, linha do Excel e coordenadas;
- legenda e tabela detalhada dos pontos.

A implementação utiliza `go.Scattermap`, baseado em MapLibre, e não requer
token do Mapbox para os estilos OpenStreetMap e Carto Positron.

Nenhuma regra de cálculo foi alterada.


## Auditoria da exclusão de aluguéis — LITE 1.6.1

O núcleo já admitia somente os tipos normalizados `Guia ITBI` e `Oferta`,
excluindo `Oferta aluguel` antes da deduplicação, do cálculo do fator de
oferta e do KNN.

A inconsistência estava apenas no indicador visual **Dados disponíveis**, que
contava todos os registros da finalidade antes desse filtro.

A versão 1.6.1 altera somente essa apresentação:

- **Dados utilizáveis** = Guias ITBI + ofertas de venda;
- **Aluguéis excluídos** = registros identificados como aluguel, locação ou
  arrendamento;
- verificação defensiva impede a continuação caso algum desses registros
  permaneça após a preparação da amostra.

Nenhuma regra de avaliação foi modificada.


## Pré-filtro seguro da amostra — LITE 1.7

A preparação passa a ocorrer antes do fator de oferta e do KNN.

### Exclusões determinísticas

São excluídos e registrados:

- valor inválido, nulo ou não positivo;
- valor simbólico de até R$ 1,00;
- área de referência inválida, nula ou não positiva;
- valor unitário inválido;
- natureza de transmissão identificada como doação, herança, partilha,
  usufruto, nua-propriedade, transmissão gratuita ou interesse parcial;
- fração transmitida inferior à totalidade, quando existe campo confiável;
- transmissão conjunta de múltiplos imóveis sem valor individualizado.

### Filtro robusto das Guias ITBI

O filtro utiliza `ln(valor unitário)` na finalidade selecionada.

- menos de 8 Guias: somente regras determinísticas;
- 8 a 14 Guias: extremos são sinalizados, mas não excluídos;
- 15 ou mais Guias: `|M| > 3,5` é excluído;
- `2,5 < |M| ≤ 3,5` permanece com alerta;
- MAD zero: cercas externas de `3×IQR` no logaritmo.

### Rastreabilidade

A exportação contém `Dados_excluidos` e `Dados_alertados`, com linha original,
motivo, etapa, valor unitário, escore robusto e limites.

O fator de oferta, o KNN, os pesos, a winsorização posterior, o mapa Plotly e
todas as demais regras permanecem inalterados.


## Deduplicação prioritária e proteção local — LITE 1.8

### Deduplicação

A primeira chave utilizada é:

`tipo_informacao + siat_inscricao + valor_oferta`

Dentro de cada grupo é mantido o registro de data mais recente. Em empate ou
ausência de data, prevalece a última linha do arquivo.

Essa chave também remove repetições exatas de Guias ITBI. Nas ofertas que não
possuem inscrição e valor utilizáveis, são empregados apenas identificadores
genuínos de anúncio. `origem_registro` deixou de ser usada como identificador,
pois descreve a fonte e pode ser comum a muitos imóveis diferentes.

### Proteção local de valores unitários muito baixos

Antes da seleção final do KNN, o estimador forma uma referência local sem
utilizar o valor desconhecido do avaliando:

1. procura imóveis com áreas entre 50% e 200% da área do avaliando e até 5 km;
2. se necessário, relaxa para 33% a 300% e até 10 km;
3. se ainda não houver dados suficientes, usa os candidatos mais próximos pela
   distância composta já existente.

Sobre os valores unitários ajustados dessa referência calcula uma barreira
inferior robusta no logaritmo. O limite considera:

- escore Z modificado de 3,5;
- cerca externa inferior de 3×IQR;
- piso conservador de 10% da mediana local.

Somente a cauda inferior é removida nessa etapa. A proteção é cancelada se a
exclusão deixar candidatos insuficientes para o KNN.

Os registros rejeitados são incorporados à planilha `Dados_excluidos`. Todas
as demais regras permanecem inalteradas.

## Pisos absolutos por finalidade — LITE 1.9

O piso é aplicado ao valor unitário original antes do fator de oferta, dos filtros robustos e do KNN. Valores inferiores são excluídos e registrados em `Dados_excluidos`.

Para finalidades com ao menos oito ofertas de venda válidas, foi usado aproximadamente 40% do percentil 10 dos valores unitários ofertados, com arredondamento conservador para baixo. Finalidades esparsas receberam pisos de famílias imobiliárias semelhantes ou da cauda inferior disponível.

| Finalidade | Piso (R$/m²) |
|---|---:|
| APART-HOTEL(FLAT) | 1.200,00 |
| APARTAMENTO | 1.200,00 |
| APARTAMENTO DE COBERTURA | 900,00 |
| CONSTRUCAO EM AREA PROJETADA DE GLEBA | 100,00 |
| DEPÓSITO / ARMAZÉM / PRÉDIO INDUSTRIAL | 650,00 |
| EDIFICIO GARAGEM | 200,00 |
| ESPACO DE ESTACIONAMENTO NAO RESIDENC DESCOBERTO | 200,00 |
| ESPACO DE ESTACIONAMENTO NAO RESIDENCIAL | 150,00 |
| ESPACO DE ESTACIONAMENTO RESIDENCIAL | 200,00 |
| ESPACO DE ESTACIONAMENTO RESIDENCIAL DESCOBERTO | 300,00 |
| GARAGEM COLETIVA | 250,00 |
| GLEBA | 20,00 |
| HOTEL OU SIMILAR | 400,00 |
| IMOVEL ESPECIAL | 400,00 |
| LOJA DE INTERIOR | 700,00 |
| LOJA EM GALERIA | 1.000,00 |
| LOJA EM SHOPPING | 1.200,00 |
| LOJA TERREA EM EDIFICIO | 1.000,00 |
| RESIDENCIA CONDOM HORIZ ABERTO SEM AREA USO COMUM | 1.000,00 |
| RESIDENCIA DE FRENTE COM INTERIORES | 1.300,00 |
| RESIDENCIA DE INTERIOR | 800,00 |
| RESIDENCIA ISOLADA | 950,00 |
| RESIDENCIA NAO PADRONIZ EM CONDOM HORIZONTAL FECHADO | 1.000,00 |
| RESIDENCIA NAO PADRONIZADA EM COND HORIZ ABERTO C/ ÁREA COMUM | 1.000,00 |
| RESIDENCIA PADRONIZADA COND HORIZ ABERTO C/ ÁREA USO COMUM | 900,00 |
| RESIDENCIA PADRONIZADA EM COND HORIZONTAL FECHADO | 1.000,00 |
| SALA COMERCIAL | 900,00 |
| SALA DE COBERTURA | 700,00 |
| TERRENO | 200,00 |
| TERRENO EM CONDOMINIO HORIZONTAL ABERTO | 150,00 |
| TERRENO EM CONDOMINIO HORIZONTAL FECHADO | 200,00 |
| TERRENOS CONDOMINIO HORIZ ABERTO SEM AREA USO COMUM | 100,00 |
| UNIDADE (DE COMÉRCIO OU SERVIÇOS) DE  FRENTE NÃO ISOLADA | 1.300,00 |
| UNIDADE DE COMERCIO E SERVICO ISOLADA | 850,00 |

A tabela completa, incluindo o critério de cada piso, está no arquivo `pisos_finalidades.csv`.

Nenhuma outra regra do estimador foi modificada.


## Hotfix de implantação — LITE 1.9.1

Esta edição não altera cálculos ou parâmetros. Ela apenas reforça que os três
arquivos abaixo devem ser publicados juntos:

- `app.py`;
- `knn_valuation.py`;
- `schema_utils.py`.

Compatibilidade esperada:

- aplicativo: LITE 1.9.1;
- núcleo KNN: 6.4.0;
- esquema SIRI: 6.4.0.


## Carregamento direto sem cache — LITE 1.9.2

O aplicativo não importa mais `knn_valuation.py` nem `schema_utils.py`.

Ele lê e executa diretamente, a partir da mesma pasta de `app.py`:

- `estimador_knn_core_v640.py`;
- `estimador_knn_schema_v640.py`.

Isso impede que o Streamlit reutilize módulos antigos do processo Python ou
localize arquivos homônimos em outra pasta do repositório.

Os antigos `knn_valuation.py` e `schema_utils.py` podem permanecer no
repositório, pois não serão utilizados. Os três arquivos indispensáveis desta
edição são `app.py` e os dois módulos exclusivos acima.


## Classificação tipológica assimétrica — LITE 1.10.0

A finalidade cadastral original e o tipo anunciado permanecem preservados.
Foram acrescentadas as colunas:

- `__segmento_siat`;
- `__segmento_crawler`;
- `__segmento_mercado`;
- `__fonte_classificacao`;
- `__conflito_tipologico`;
- `__confianca_classificacao`.

### Regra

- Guia ITBI: segmento definido por `siat_finalidade_descricao`;
- Oferta: prevalece `crawler_tipo_imovel_normalizado` quando específico;
- Oferta sem tipo específico do crawler: fallback ao SIAT.

A finalidade cadastral escolhida para o avaliando continua determinando o piso
de valor unitário. A seleção dos comparáveis passa a ocorrer pelo segmento de
mercado correspondente.

Exemplo: uma oferta anunciada como apartamento, mas ainda cadastrada como
unidade de comércio, será classificada no segmento `APARTAMENTO`, com fonte
`crawler`, conflito tipológico `Sim` e confiança `Média`.

Nenhuma regra de desconto, deduplicação, pisos, filtros robustos, KNN, pesos ou
mapa foi modificada.


## Taxonomia única `finalidade_crawler_normalizada` — LITE 1.11.0

A finalidade informada na tela passa a ser exclusivamente a finalidade
normalizada de mercado. As descrições originais permanecem disponíveis para
auditoria.

### Prioridade de normalização

1. coluna explícita `finalidade_crawler`, quando existente;
2. `crawler_tipo_imovel_normalizado`, para ofertas;
3. `siat_finalidade_descricao`, convertida para a mesma taxonomia.

As Guias ITBI são convertidas do SIAT para as mesmas categorias utilizadas
pelas ofertas.

### Colunas derivadas

- `__finalidade_crawler_informada_normalizada`;
- `__finalidade_siat_normalizada`;
- `__finalidade_tipo_crawler_normalizada`;
- `__finalidade_crawler_normalizada`;
- `__fonte_normalizacao`;
- `__conflito_tipologico`;
- `__confianca_normalizacao`.

### Área de referência

- apartamento, cobertura, flat, sala e garagem: área privativa;
- terreno e gleba: área total do lote;
- casa, galpão, hotel e imóvel especial: área construída;
- lojas e imóveis comerciais: área privativa ou construída.

Os pisos de valor unitário também passaram a seguir essa taxonomia única.

Deduplicação, fator de oferta, filtros estatísticos, KNN, pesos, mapa e
exportações permanecem com as regras da versão anterior.


## Pesos da distância composta — LITE 1.12.0

A distância utilizada para ordenar e ponderar os comparáveis passa a ser:

`D = sqrt((2/3) × D_atributos² + (1/3) × D_geográfica²)`

- atributos físicos: 2/3;
- distância geográfica: 1/3.

A ponderação posterior continua inversamente proporcional à distância
composta, com expoente 1,0 e limite máximo de 30% para um único comparável.

Nenhuma outra regra foi modificada.


## Parâmetros calibrados — LITE 1.13.0

A calibração utilizou 14,845 registros do período de 30/04/2026 a
30/07/2026 e 415 avaliações fora da amostra, com
agrupamento de duplicidades, fator de oferta recalculado por rodada e validação
temporal.

### Perfil global

- K inicial: 12;
- K máximo: 25;
- vizinhos efetivos mínimos: 11;
- peso físico: 45%;
- peso geográfico: 55%;
- peso máximo individual: 25%;
- potência da distância: 0.35;
- limiar robusto: 1.25.

### Perfis específicos

- CASA / RESIDÊNCIA: perfil específico validado;
- SALA COMERCIAL: perfil específico validado;
- GARAGEM / VAGA: parâmetros da versão 1.12 preservados.

Na amostra completa de backtesting, o MdAPE passou de
33.25% para 28.69%,
redução relativa de 13.7%.

Os parâmetros aplicados ficam registrados na exportação de diagnóstico.


## Conflito tipológico como contingência — LITE 1.14.0

O aplicativo prepara inicialmente uma amostra sem registros classificados
como conflito `Sim` ou `Moderado`. Se essa amostra atingir o K inicial do
perfil selecionado, os dados conflitantes permanecem excluídos.

Quando a amostra limpa não atinge esse mínimo, a preparação é refeita com os
dados conflitantes. Esses registros são marcados como contingenciais nos
comparáveis, alertas e exportações.

## Finalidades residenciais e não residenciais

A taxonomia passa a distinguir:

- `GARAGEM / VAGA RESIDENCIAL`;
- `GARAGEM / VAGA NÃO RESIDENCIAL`;
- `GARAGEM / VAGA`, apenas quando a natureza não puder ser determinada.

A coluna `__natureza_uso_normalizada` registra `RESIDENCIAL`,
`NÃO RESIDENCIAL`, `TERRITORIAL` ou `INDETERMINADA`.

As duas novas finalidades herdam o perfil KNN conservador anteriormente
utilizado para garagens. Os demais parâmetros calibrados foram preservados.


## Filtro local adaptativo seguro — LITE 1.15.0

A proteção da cauda inferior passa a trabalhar com o submercado local do
avaliando. A referência prioriza imóveis com áreas compatíveis e menor
distância geográfica ou composta.

O limite automático combina mediana local ponderada pela proximidade, MAD e
IQR em escala logarítmica, fração mínima da mediana e ruptura entre grupos de
valor.

Para salas comerciais, a fração de segurança é 55% da mediana local
ponderada. Uma mediana de R$ 9.000/m² gera referência de R$ 4.950/m², sem
fixar um corte nominal para toda a cidade.

A ruptura somente é aceita quando o grupo inferior representa no máximo 30%
da referência, o grupo superior preserva ao menos o K inicial e apresenta
compatibilidade física e geográfica suficiente.

Travas:

- não excluir mais de 30% da amostra;
- preservar ao menos o K inicial;
- cancelar a exclusão quando a cauda puder representar submercado legítimo;
- usar valores já ajustados pelo fator de oferta;
- exportar mediana, limites, ruptura, origem do corte e registros excluídos.

Parâmetros KNN, política de conflito tipológico e taxonomia da versão 1.14
foram preservados.


## Regime seguro de área — LITE 1.16.0

As finalidades construídas deixam de exigir universalmente a área privativa.

A interface oferece:

- `Automática`;
- `Área privativa`;
- `Área total/construída`.

Cada alternativa é preparada separadamente, com deduplicação, fator de
oferta, conflitos tipológicos, pisos e pré-filtros. Os valores unitários de
regimes diferentes nunca são misturados na mesma estimativa.

### Modo automático

1. identifica o regime preferencial da finalidade;
2. prepara a amostra privativa e a amostra total/construída;
3. mantém o regime preferencial se ele atingir o K inicial;
4. caso contrário, escolhe a alternativa quando ela atingir o K inicial;
5. se nenhuma atingir o mínimo, interrompe a seleção automática e informa
   as contagens de cada regime.

O usuário pode selecionar manualmente um regime com amostra inferior ao K
inicial. Nesse caso, o aplicativo registra e alerta a limitação.

### Pisos

O piso absoluto por finalidade é mantido quando o regime escolhido é
compatível com a base originalmente prevista:

- finalidade privativa com área privativa;
- finalidade construída com área total/construída;
- finalidades que admitem ambas as bases.

Quando o aplicativo usa uma base alternativa ainda não calibrada, o piso
absoluto é desativado. Permanecem ativos os filtros relativos, robustos,
locais e de ruptura entre grupos.

### Atributos físicos

Somente a área do regime selecionado participa da distância física do KNN.
Isso evita exigir simultaneamente área privativa e área total nos mesmos
comparáveis.

A exportação informa o modo solicitado, regime escolhido, motivo da escolha,
contagens por regime e situação do piso.


## Ano da construção opcional — LITE 1.17.0

O ano da construção (`siat_ano`) pode participar da distância física de
imóveis prediais. Ele só restringe a amostra quando a coluna existe e o ano do
avaliando é informado. Terrenos permanecem sem esse atributo.


## Área do terreno, importação genérica e endereço — LITE 1.18.0

- casas e outros imóveis térreos podem usar a área do terreno como atributo
  físico adicional, sem alterar o denominador do valor unitário;
- bases de anúncios reconhecem cabeçalhos usuais de tipo, finalidade, preço,
  área construída, área do terreno e endereço, inclusive unidades no nome ou
  no conteúdo das células;
- o esquema validado de extrações do Viva Real reconhece área anunciada,
  áreas descritas, valor de oferta em reais e endereço fragmentado; números de
  porta integrais não são enviados à geocodificação com sufixo decimal;
- a área do terreno só é ativada por padrão quando há cobertura para o K
  inicial e fica indisponível quando menos de dois comparáveis a informam,
  sem imputação silenciosa;
- na ausência do tipo da informação em uma base não SIRI, a hipótese auditável
  é que os registros são ofertas;
- endereços de Porto Alegre usam correspondência fuzzy, faixa par/ímpar e
  interpolação no CRS métrico TM-POA, convertida depois para EPSG:4326;
- o fallback online usa Nominatim com limitação de requisições;
- sem coordenadas utilizáveis, o peso geográfico é zerado e o componente
  físico é renormalizado para 100%, sem excluir linhas apenas pela localização.

A mudança espacial não cria agregações de vizinhança nem utiliza o alvo; logo,
não introduz leakage entre treino e teste. Ainda assim, cobertura desigual da
geocodificação pode gerar viés espacial e deve ser auditada por estrato e
região.
