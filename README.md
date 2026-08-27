![Logotipo VERA](./static/vera_header.png)
# VERA
## Valor Estimado por Referências Amostrais

Aplicativo web para estimar o valor de imóveis a partir de transações e
ofertas comparáveis. Combina similaridade física, proximidade geográfica e
tratamentos estatísticos robustos em um fluxo simples, auditável e preparado
para planilhas do SIRI ou bases de anúncios com cabeçalhos usuais.

| | |
|---|---|
| **Interface** | Aplicativo web em Streamlit |
| **Método** | K-vizinhos mais próximos (KNN) |
| **Entrada** | Planilhas Excel nos formatos `.xlsx`, `.xlsm` ou `.xls` |
| **Saída** | Estimativa, comparáveis, mapa e relatórios Excel/PDF |
| **Abrangência** | Imóveis prediais e territoriais |
| **Versão atual** | LITE 1.19.0 · núcleo 6.12.0 |

## O que o aplicativo faz

A VERA recebe uma planilha de dados imobiliários e encontra os
imóveis mais compatíveis com o bem avaliando. O processo:

1. reconhece e normaliza o schema da planilha;
2. seleciona registros da mesma finalidade imobiliária;
3. exclui aluguéis, duplicidades e observações inválidas;
4. ajusta ofertas em relação às transações observadas;
5. reconhece coordenadas ou geocodifica endereços, quando solicitado;
6. calcula a similaridade física e, quando disponível, a proximidade geográfica;
7. escolhe adaptativamente os comparáveis e limita concentrações de peso;
8. reduz a influência de valores extremos com tratamento robusto;
9. apresenta o valor estimado e os elementos necessários à auditoria.

Para imóveis prediais, a similaridade física considera a área do regime
selecionado, o ano da construção quando informado e, em imóveis térreos, a
área do terreno quando ativada. A área do terreno é um atributo adicional e
não altera o denominador do valor unitário. Para terrenos, considera a área do
lote e a testada. O aplicativo trabalha com um único regime de área em cada
estimativa, evitando misturar denominadores incompatíveis.

## Principais recursos

- reconhecimento automático das colunas usadas pelo SIRI e de cabeçalhos
  comuns de raspagens, inclusive unidades como `(m²)` e moeda `(R$)`;
- compatibilidade verificada com extrações do Viva Real que usam, entre outros,
  `valor_oferta_rs`, `area_anunciada_m2`, `area_terreno_m2`, `logradouro`,
  `numero`, `bairro_portal`, `municipio` e `uf`;
- taxonomia normalizada de finalidades imobiliárias;
- tratamento separado para imóveis prediais e territoriais;
- área privativa ou área total/construída como base da estimativa;
- ano da construção como atributo físico de imóveis prediais;
- área do terreno como atributo físico opcional de casas e outros imóveis
  térreos, ativada por padrão somente quando a cobertura alcança o K inicial;
- geocodificação por endereço pelo eixo viário de Porto Alegre, com fallback
  Nominatim;
- modo sem localização, com peso físico renormalizado para 100%;
- K adaptativo e perfis de parâmetros por finalidade;
- combinação de distância física e distância geográfica;
- deduplicação de ofertas e exclusão de ofertas de aluguel;
- fator de oferta calculado ou conservador, conforme a amostra;
- pré-filtros, controles de cauda e média ponderada robusta;
- controle de conflitos tipológicos;
- limite para o peso individual de cada comparável;
- mapa interativo do avaliando e dos comparáveis;
- relatório sintético de inferência em PDF, com mapa Claro - CARTO Positron,
  valor unitário, valor total, COD local, confiança e comparáveis;
- alertas de extrapolação, cobertura e concentração dos pesos;
- exportação de comparáveis, diagnósticos, exclusões e alertas para Excel.

## Como usar

1. Abra o aplicativo no navegador.
2. Envie a planilha Excel.
3. Selecione a finalidade do imóvel.
4. Escolha ou confirme o regime de área.
5. Informe as características do avaliando e escolha endereço, coordenadas ou
   o modo sem localização.
6. Clique em **Calcular estimativa**.
7. Analise o resultado, os comparáveis e o mapa.
8. Baixe o relatório sintético em PDF ou o relatório Excel para auditoria.

## Dados necessários

| Escopo | Informações esperadas |
|---|---|
| Todos os imóveis | Finalidade, valor e ao menos uma área utilizável |
| Localização | Latitude/longitude ou endereço; pode ser ignorada |
| Imóvel predial | Área privativa ou construída; ano e área do terreno são opcionais |
| Imóvel territorial | Área total do lote e testada |

O aplicativo possui aliases para diferentes nomes de colunas de área,
coordenadas, endereços e valores. A finalidade é convertida automaticamente
para a taxonomia utilizada pelo estimador. Quando uma base não SIRI não possui
tipo da informação, os registros são tratados explicitamente como ofertas e
essa hipótese é registrada no diagnóstico. Quando usado, o ano da construção
deve ser inteiro entre 1500 e o ano corrente.

## Linguagem e tecnologias

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.10 ou superior |
| Interface web | Streamlit 1.47+ |
| Manipulação de dados | pandas e NumPy |
| Leitura e escrita de Excel | openpyxl e xlrd |
| Mapas e visualizações | Plotly / MapLibre e CARTO Positron |
| Relatório PDF | ReportLab, Pillow e mapbox-vector-tile |
| Geocodificação | eixo TM-POA, pyproj, RapidFuzz e Nominatim/geopy |
| Testes | `unittest` da biblioteca padrão |

As versões aceitas das bibliotecas estão declaradas em `requirements.txt`.

## Recursos necessários

### Software

- Python 3.10 ou superior;
- `pip` para instalação das dependências;
- navegador web atualizado;
- Git, caso o projeto seja obtido diretamente do repositório.

Não é necessário banco de dados, servidor externo ou GPU. O cálculo é local.
Na primeira geocodificação de Porto Alegre, os quatro arquivos do eixo viário
são baixados do Space `fschwartzer/Geocode` e verificados por SHA-256. O
fallback Nominatim e o mapa-base também dependem de internet. Com coordenadas
já fornecidas ou no modo sem localização, a estimativa não depende desses
serviços.

O mapa do PDF usa os tiles vetoriais oficiais do CARTO Positron e mantém a
atribuição ao CARTO e ao OpenStreetMap. Se os tiles não puderem ser obtidos, o
relatório é emitido com uma grade neutra e registra a contingência.

### Hardware

- processador convencional de 64 bits;
- pelo menos 4 GB de memória disponível para planilhas pequenas ou médias;
- 8 GB ou mais para bases maiores.

As planilhas são processadas em memória. O limite configurado para upload é
de 200 MB, mas o consumo efetivo de RAM pode ser várias vezes maior que o
tamanho do arquivo Excel.

## Como rodar localmente

Clone o repositório e entre na pasta do projeto:

```powershell
git clone https://github.com/fschwartzer/estimador_knn_siri.git
cd estimador_knn_siri
```

Crie e ative um ambiente virtual no Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

No Linux ou macOS, a ativação é:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências e inicie o aplicativo:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

O Streamlit exibirá no terminal o endereço local, normalmente
`http://localhost:8501`.

## Testes

Execute a suíte automatizada a partir da raiz do projeto:

```powershell
python -m unittest discover -s tests -v
```

Os testes verificam a integridade da versão publicada, o mapeamento de bases
SIRI e genéricas, a ativação dos atributos físicos, o tratamento do ano da
construção e os caminhos com e sem localização.

## Estrutura do repositório

```text
estimador_knn_siri/
├── app.py                         # interface e orquestração
├── estimador_knn_core_v6120.py    # cálculo e diagnósticos do KNN
├── estimador_knn_schema_v6120.py  # normalização do schema SIRI
├── geocodificador_porto_alegre.py # geocodificação pelo eixo do Space Geocode
├── vera_pdf_report.py             # relatório PDF e mapa estático dos comparáveis
├── requirements.txt               # dependências de execução
├── tests/                         # testes automatizados
├── artifacts/calibration/         # evidências e resultados de calibração
├── archive/legacy/                # módulos de versões anteriores
├── docs/                          # documentação histórica
└── .streamlit/config.toml          # tema e configuração do servidor
```

Os módulos de `archive/legacy/` não participam da execução atual. Eles foram
preservados apenas para rastreabilidade e reprodução de resultados antigos.

## Publicação no Streamlit Community Cloud

Defina `app.py` como arquivo principal. Os arquivos indispensáveis no mesmo
repositório são:

- `app.py`;
- `estimador_knn_core_v6120.py`;
- `estimador_knn_schema_v6120.py`;
- `geocodificador_porto_alegre.py`;
- `vera_pdf_report.py`;
- `requirements.txt`;
- `.streamlit/config.toml`.

## Rastreabilidade e uso responsável

O aplicativo é um instrumento de apoio à avaliação, não um substituto para a
análise técnica. Os comparáveis, exclusões, pesos e diagnósticos devem ser
revisados antes do uso operacional do resultado.

No PDF, o "COD dos comparáveis" descreve a dispersão dos valores unitários
ajustados da seleção final em torno da mediana. Ele não é o COD de um estudo de
razões avaliação/preço fora da amostra. Da mesma forma, o score de confiança é
um indicador heurístico da qualidade da estimativa, não um intervalo de
confiança estatístico.

Em avaliações em massa, valide o modelo em dados separados do ajuste e
acompanhe, além dos erros preditivos, a mediana das razões, COD, PRD,
regressividade por faixa de valor e estabilidade temporal e espacial. A
ausência sistemática de atributos, como o ano da construção, pode reduzir a
cobertura e introduzir viés amostral.

Coordenadas interpoladas representam uma aproximação ao longo do eixo da via,
não a posição cadastral do lote. Se apenas parte da amostra for geocodificada,
revise o possível viés de cobertura espacial. A inclusão da área do terreno
também deve ser validada fora da amostra, observando mediana das razões, COD,
PRD, regressividade por faixa de valor e estabilidade espacial.

## Documentação complementar

- `VERSION.txt`: identificação da versão em produção;
- `README_DEPLOY.txt`: lista mínima para publicação;
- `docs/HISTORICO_VERSOES.md`: histórico funcional até a versão 1.19.0;
- `artifacts/calibration/`: parâmetros e relatórios de calibrações anteriores.
