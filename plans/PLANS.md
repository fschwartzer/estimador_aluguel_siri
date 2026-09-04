# Plano de execução — SIRI Aluguéis

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
- [ ] Criar e publicar um novo repositório no GitHub.

## Critérios de aceitação

- Somente ofertas de aluguel válidas entram no conjunto de comparáveis; registros não-oferta não entram silenciosamente.
- Regras de corte e KNN são explícitas, versionadas e específicas por finalidade quando houver suporte amostral.
- Distâncias preservam a unidade/CRS documentados e não usam informação futura ou do imóvel avaliado na calibração.
- A interface e o relatório não usam a identidade visual VERA.
- Os testes automatizados passam e cobrem os principais comportamentos novos.
