# Plano de execução — SIRI Aluguéis

## Correção do PDF hospedado — 09/10/2026

- [x] Reproduzir a incompatibilidade de `diagnostics` com um módulo PDF antigo em `sys.modules`.
- [x] Carregar o gerador PDF pelo arquivo publicado, como o core/schema; usar seu hash na chave do cache.
- [x] Executar regressão do módulo antigo, suíte completa e fluxo com os dois downloads: 39 testes aprovados e AppTest sem erros.
- [x] Preparar a publicação em PR de correção, com causa, mudança e evidências.
- [ ] Integrar na main e validar a atualização do aplicativo hospedado.

O repositório principal já possui `diagnostics` no gerador; a importação comum preservava referências antigas no processo durante atualização. O teste de regressão reproduziu o TypeError antes da correção. O mecanismo de calibração não participa deste defeito. A rotina auxiliar de inicialização da skill falhou no ambiente Windows (`NtCreateDirectoryObject: 0xC0000022`); investigação e validação seguem diretamente no projeto.

O site `https://estimadoraluguel.streamlit.app/` abriu normalmente no navegador. A geração com os dois downloads foi validada localmente; ainda não se afirma que o servidor recebeu a correção. Aprendizado: módulos que mudam assinatura também precisam seguir o carregamento por arquivo adotado pelo core/schema em um processo Streamlit que sobrevive à atualização.

## Atualização de 09/10/2026 — calibração não residencial e rastreabilidade

Fonte de verdade desta atualização. Os critérios anteriores ficam preservados abaixo como histórico.

- [x] Auditar a pesquisa de 09/10/2026, datas, finalidades, áreas e identificadores.
- [x] Separar treino, ajuste temporal e validação posterior por imóvel/anúncio, sem futuros ou duplicados nos comparáveis.
- [x] Comparar K, pesos físico/geográfico, MAD, pisos e recência; registrar precisão, COD, PRD, razões por valor e localização.
- [x] Integrar parâmetros aceitos, mantendo fallback explícito para finalidades sem suporte.
- [x] Retirar a expressão VU ajustado das saídas e incluir endereços, fatores e pesos no PDF e auditoria Excel.
- [x] Validar regressões, executar o aplicativo e revisar exemplos reais de relatório e auditoria.
- [x] Documentar resultados, limitações e entregar as alterações em branch revisável.

Resultados: 38 testes aprovados; fluxo completo com recálculo, PDF e Excel; 652 previsões na validação de promoção e 150 alvos com holdout espacial. Novos perfis aceitos: salas e galpões. O ganho das salas permanece incerto por bootstrap e envolve piora do P90, PRD e erro médio em reais. Limitações de atributos cadastrais históricos, preços pedidos, regressividade e seleção de perfis estão registradas na metodologia de outubro.

Hipótese: comparáveis da mesma finalidade, próximos em atributos e localização, aproximam o aluguel anunciado. Recência pode reduzir defasagem, mas não corrige preço pedido para contratado. Pisos e filtros de cauda podem elevar razões e agravar regressividade. A seleção de candidatos usa somente desenvolvimento; a validação posterior aceita ou rejeita sua promoção, sem retuning. Esse período é validação de promoção, não teste prospectivo intocado. Perfis raros são explicitamente classificados como sem suporte.

## Objetivo

Criar um aplicativo independente, derivado de `estimador_knn_siri`, para estimar aluguéis por KNN usando apenas ofertas válidas, com parâmetros calibrados por finalidade a partir da pesquisa de 04/09/2026 e identidade visual vermelha.

## Etapas

- [x] Mapear o fluxo atual de importação, filtros, cortes, seleção de comparáveis, estimativa e relatório.
- [x] Auditar a planilha de aluguel: esquema, qualidade, cobertura espacial, finalidade, valores e outliers.
- [x] Definir cortes e parâmetros KNN por finalidade sem vazamento de dados.
- [x] Implementar a nova regra de retenção de ofertas e a calibração de aluguéis.
- [x] Substituir identidade VERA por SIRI Aluguéis e aplicar paleta vermelha.
- [x] Atualizar documentação e testes.
- [x] Executar testes, verificações estáticas e validação funcional.
- [x] Criar e publicar um novo repositório no GitHub.

## Critérios de aceitação

- Somente ofertas de aluguel válidas entram no conjunto de comparáveis; registros não-oferta não entram silenciosamente.
- Regras de corte e KNN são explícitas, versionadas e específicas por finalidade quando houver suporte amostral.
- Distâncias preservam a unidade/CRS documentados e não usam informação futura ou do imóvel avaliado na calibração.
- A interface e o relatório não usam a identidade visual VERA.
- Os testes automatizados passam e cobrem os principais comportamentos novos.
