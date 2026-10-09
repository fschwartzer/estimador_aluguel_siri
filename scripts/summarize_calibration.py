"""Gera documentação e tabela agregada a partir da execução registrada."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_rentals import ROOT, core


if __name__ == "__main__":
    path = ROOT/"artifacts/calibration/aluguel_parametros_2026_10.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(ROOT/"tmp/predictions.csv")
    rows = []
    intervals = {}
    rng = np.random.default_rng(20261009)
    for purpose, evaluation in report["evaluations"].items():
        current = evaluation.get("test_selected")
        if not current:
            continue
        baseline = evaluation["test_baseline"]
        config = report["purpose_parameters"][core.normalize_text(purpose)]
        rows.append(dict(finalidade=purpose,n=current["n"],mdape_anterior=baseline["mdape"],mdape_atual=current["mdape"],cod=current["cod"],prd=current["prd"],mediana_razoes=current["median_ratio"],k_min=config["min_k"],k_max=config["max_k"],peso_fisico=config["similarity_weight"],peso_geografico=config["location_weight"],recencia_ativa=config["temporal_weight_enabled"],meia_vida_dias=config["temporal_half_life_days"],fator_minimo_recencia=config["temporal_min_factor"],piso_vu=config["unit_value_floor"],ganho_confirmado=evaluation["accepted_for_use"]))
        own = predictions.loc[predictions.purpose.eq(purpose)]
        paired = own.loc[own.model.eq("selecionado")].merge(own.loc[own.model.eq("legado")],on="index",suffixes=("_new","_old"))
        new = abs(paired.predicted_new.to_numpy()/paired.actual_new.to_numpy()-1)
        old = abs(paired.predicted_old.to_numpy()/paired.actual_old.to_numpy()-1)
        indices = rng.integers(0,len(paired),size=(2000,len(paired)))
        change = np.median(new[indices],axis=1)-np.median(old[indices],axis=1)
        intervals[purpose] = dict(mdape_change_ci95_pp=(np.quantile(change,[.025,.975])*100).tolist(),bootstrap_resamples=2000,method="bootstrap pareado de imóveis; intervalo exploratório, sem correção pela seleção de modelos")
        evaluation["uncertainty"] = intervals[purpose]
    report["validation_role"] = "agosto/setembro usado como porta de promoção, sem reajustar candidatos neste período; métricas retrospectivas, não teste prospectivo intocado"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(rows).to_csv(ROOT/"artifacts/calibration/validacao_nao_residencial_2026_10.csv",index=False)
    lines = [
        "# Calibração não residencial — outubro de 2026", "",
        "## Decompor", "",
        "Fonte: `SIRI_pesquisa_aluguel-09.10.2026 10.47.31.xlsx`, cópia binária local identificada pelo SHA-256 no JSON. São 149.842 linhas, de 04/10/2019 a 01/09/2026, todas classificadas como oferta de aluguel. O alvo é aluguel mensal anunciado por m². Nenhum preço contratado está disponível. A coleta não é a data do negócio nem a data da avaliação.", "",
        "A normalização de finalidade e área é a mesma do aplicativo. Somente finalidades não residenciais entram na nova pesquisa. Salas e lojas usam área privativa; galpões usam construída; imóveis comerciais usam construída pela cobertura histórica de treinamento (área privativa em apenas 58 de 6.374 registros totais). Terrenos e glebas usam área de lote e testada. O perfil especializado só é aplicado ao denominador calibrado; outra área mantém o perfil legado.", "",
        "## Resolver", "",
        "Hipótese técnica: proximidade física e geográfica entre imóveis da mesma finalidade aproxima o aluguel anunciado. A recência controla a influência relativa de coletas antigas; não atualiza preços por um índice e não aplica desconto oferta/transação.", "",
        "Identidades são componentes conectados por inscrição SIAT, URL e origem+código do anúncio, sem utilizar preço. Cada janela mantém o registro mais recente por componente. Todas as identidades elegíveis da janela, inclusive as não amostradas, são retiradas do treino. Treinamento contém somente datas anteriores à janela. A normalização robusta, seleção espacial, filtros de cauda e winsorização são refeitos exclusivamente nos comparáveis de treinamento. Alvos de validação não são cortados pelo piso de preço, evitando uma melhora artificial por remoção de aluguéis baixos.", "",
        "Desenvolvimento: março/2026, até 100 imóveis por finalidade; confirmação de desenvolvimento: setembro/2025, até 100. Validação posterior: agosto/setembro/2026, até 200. Amostragem aleatória com semente 20261009. Exigem-se pelo menos 35 alvos e 80 linhas históricas no desenvolvimento para calibrar. Esses mínimos são controles operacionais, não prova de precisão inferencial.", "",
        "Busca em etapas: K 6–15/12–25/20–40, vizinhos efetivos 5/10/16, peso físico 0,20/0,50/0,80, potência 0,35/0,75, MAD 1,50/3,00 e filtro local ligado/desligado. Compara pisos zero/metade/inteiro do legado, teto individual 0,15/0,25 e bônus de edifício 1/2. Nos melhores perfis e no legado, compara recência desligada e meias-vidas 90/180/365/730 dias, com fator mínimo 0,10/0,35/0,65. A busca é finita e escalonada; não há garantia de ótimo global.", "",
        "Score = MdAPE + 0,15×P90 APE + 0,50×|mediana da razão−1| + 0,20×COD/100 + 0,30×|PRD−1|. Na confirmação, o score é normalizado pelo legado de cada período. A recência exige ganho agregado de pelo menos 1% sobre a melhor opção sem recência no desenvolvimento. O candidato só substitui o legado quando a validação posterior tem ≥30 previsões, nenhuma falha, score pelo menos 1% menor e MdAPE não maior. A validação posterior é porta de promoção, portanto suas métricas não devem ser apresentadas como teste final prospectivo intocado. Não se escolhe uma alternativa diferente procurando o melhor resultado posterior.", "",
        "## Verificar", "",
        "Razão = aluguel estimado/aluguel observado. COD = 100×média dos desvios absolutos das razões em torno de sua mediana/mediana. PRD principal = média simples das razões/(soma dos aluguéis totais estimados/soma dos aluguéis totais observados). O PRD de valores unitários também é registrado, com outro significado de ponderação, e pode ser muito sensível a erros de área. As métricas abaixo são da validação temporal posterior, nunca do treino.", "",
        "| Finalidade | n | MdAPE legado | MdAPE aplicado | COD | PRD | Mediana das razões |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['finalidade']} | {row['n']} | {row['mdape_anterior']:.1%} | {row['mdape_atual']:.1%} | {row['cod']:.2f} | {row['prd']:.3f} | {row['mediana_razoes']:.3f} |")
    lines += ["", "Intervalos exploratórios de 95% para a mudança de MdAPE (novo menos legado), em pontos percentuais:", ""]
    for purpose, interval in intervals.items():
        if report["evaluations"][purpose]["accepted_for_use"]:
            lower, upper = interval["mdape_change_ci95_pp"]
            lines.append(f"- {purpose}: {lower:.2f} a {upper:+.2f} p.p.")
    lines += ["", "O intervalo das salas inclui zero: a melhora pontual não demonstra ganho estatístico estável. Confiança moderada no resultado de galpões e baixa a moderada nas salas; os intervalos não corrigem a seleção dos perfis."]
    sala = report["evaluations"].get("SALA COMERCIAL", {})
    if sala.get("accepted_for_use"):
        old, new = sala["test_baseline"], sala["test_selected"]
        lines += ["", f"Nas salas, o P90 APE passa de {old['p90_ape']:.2%} para {new['p90_ape']:.2%}, o PRD de {old['prd']:.3f} para {new['prd']:.3f} e o erro absoluto médio do aluguel total de R$ {old['mean_absolute_error_brl_month']:.2f} para R$ {new['mean_absolute_error_brl_month']:.2f}. O novo perfil melhora o score composto e a mediana do erro, mas piora a cauda e o erro médio em reais; essa troca exige acompanhamento operacional."]
    lines += ["", "Os segmentos mantidos no legado não passaram a porta de promoção. Isso evita aplicar candidatos cuja melhora em desenvolvimento não se sustentou no período posterior."]
    lines += ["", "A comparação aplica os parâmetros legados e candidatos à mesma data de referência do alvo. Ela compara perfis sob uma regra temporal comum; não reproduz todos os detalhes da versão anterior, cuja referência de recência era a última coleta dos comparáveis.", "", "Os atributos cadastrais SIAT da pesquisa não possuem histórico de vigência verificável. O corte temporal impede preços/coletas futuros e identidades compartilhadas no treino, mas não comprova que todo atributo cadastral já era conhecido na data histórica. Há 111 registros brutos com ano de construção posterior à coleta; nos perfis aplicados, nenhum permanece em salas, galpões, lojas ou terrenos após preparo, e dois permanecem no treino de imóveis comerciais (perfil legado). Essa inconsistência e a falta de versionamento cadastral limitam a interpretação do backtesting como simulação histórica estrita."]
    lines += ["", "O JSON inclui candidatos rejeitados, métricas dos dois períodos de desenvolvimento, ablação da recência no candidato, decis de aluguel total, bairros com pelo menos cinco alvos, holdout espacial em blocos de 1 km e intervalos exploratórios por bootstrap pareado (2.000 reamostragens). O holdout remove o bloco do alvo antes de qualquer preparo, sem buffer; 30 imóveis por finalidade. Distâncias: aproximação equiretangular local sobre EPSG:4326, em km. O bônus pleno usa 30 m, com transição até 50 m; coordenadas coincidentes não provam identidade de edifício.", "", "## Sintetizar", "", "| Finalidade | K | Peso físico/geográfico | Recência | Piso (R$/m²/mês) | Situação |", "|---|---|---|---|---:|---|"]
    for row in rows:
        recency = f"{row['meia_vida_dias']:.0f} dias; mínimo {row['fator_minimo_recencia']:.2f}" if row['recencia_ativa'] else "desligada"
        lines.append(f"| {row['finalidade']} | {row['k_min']}–{row['k_max']} | {row['peso_fisico']:.2f}/{row['peso_geografico']:.2f} | {recency} | {row['piso_vu']:.2f} | {'novo perfil confirmado' if row['ganho_confirmado'] else 'legado mantido'} |")
    lines += ["", "Finalidades sem suporte: " + "; ".join(f"{p} ({e.get('tuning_n',0)} alvos no ajuste; {e.get('train_rows',0)} linhas históricas)" for p,e in report['evaluations'].items() if 'test_selected' not in e) + ". Não são vendidas como recalibradas.", "", "O PDF usa VU mensal e VU robusto, com endereço, referência de linha, data da coleta, idade, peso bruto de distância, fator de recência, bônus e peso final. O Excel conserva componentes, endereços, diagnósticos e exclusões. Soma dos pesos = 1; soma de peso final×VU robusto = VU estimado. A coluna interna anterior é preservada no core para compatibilidade com notebooks, sem exposição como VU ajustado nos relatórios.", "", "## Ressalvas", "", "Há overfitting possível pela busca de muitos perfis, amostras limitadas e snapshots correlacionados. A confiança é baixa para segmentos raros. O bootstrap não incorpora toda a incerteza de seleção, dependência espacial ou erros de identificação. Uma validação prospectiva com novas coletas continua necessária.", "", "Viés: preços pedidos podem diferir dos contratados; duplicados sem identidade compartilhada podem persistir; cobertura espacial, padrão, conservação e conflitos de cadastro não são uniformes. Pisos e exclusão da cauda inferior podem elevar razões e agravar regressividade. COD e PRD observados não demonstram uniformidade satisfatória; comparar decis e regiões é indispensável. PRD acima de 1 sugere maior razão relativa em aluguéis de menor valor, mas a interpretação requer dados válidos e controle de heterogeneidade.", "", "O alvo não é transformado em log. Logs de VU são usados somente nos filtros robustos de treinamento; não há retransposição exponencial de previsão nem smearing. Endereços cadastrados podem não ser o endereço efetivo do anúncio. O score de confiança do relatório é heurístico, sem interpretação de probabilidade ou intervalo estatístico.", "", "Reprodução:", "", "```powershell", "python -m pip install -r requirements-dev.txt", "python scripts/inspect_rental_source.py '<pesquisa.xlsx>' tmp/source.pkl --enriched tmp/enriched.pkl", "python scripts/calibrate_rentals.py --cache tmp/enriched.pkl --source '<pesquisa.xlsx>' --limit 100", "python scripts/validate_spatial_rentals.py --cache tmp/enriched.pkl", "python scripts/summarize_calibration.py", "python -m unittest discover -s tests -v", "```", "", "Dados individuais e exemplos de auditoria ficam fora do Git. O SHA-256 identifica a cópia usada; o JSON e a tabela CSV agregada são versionados."]
    (ROOT/"artifacts/calibration/METODOLOGIA_ALUGUEL_2026_10.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(pd.DataFrame(rows)[["finalidade","n","mdape_anterior","mdape_atual","cod","prd","ganho_confirmado"]].to_string(index=False))
