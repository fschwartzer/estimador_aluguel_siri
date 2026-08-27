from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable
import re

import numpy as np
import pandas as pd


MODULE_API_VERSION = "6.12.0"
MODULE_BUILD_ID = "estimador-knn-siri-lite-1.19.0-20260826"


MIN_CONSTRUCTION_YEAR = 1500
MAX_CONSTRUCTION_YEAR = date.today().year


TIPO_ITBI = "guia itbi"
TIPO_OFERTA = "oferta"
TIPO_ALUGUEL = "oferta aluguel"


CALIBRATED_GLOBAL_PARAMETERS: dict[str, float | int | str] = {
    "profile": "global_calibrado_3_meses",
    "min_k": 12,
    "max_k": 25,
    "min_effective_neighbors": 11.0,
    "similarity_weight": 0.45,
    "location_weight": 0.55,
    "max_individual_weight": 0.30,
    "distance_power": 0.35,
    "robust_mad_threshold": 1.25,
}

CALIBRATED_PURPOSE_PARAMETERS: dict[
    str,
    dict[str, float | int | str],
] = {
    "casa / residencia": {
        "profile": "especifico_casa_residencia",
        "min_k": 5,
        "max_k": 40,
        "min_effective_neighbors": 5.0,
        "similarity_weight": 0.85,
        "location_weight": 0.15,
        "max_individual_weight": 0.20,
        "distance_power": 0.50,
        "robust_mad_threshold": 2.00,
    },
    "sala comercial": {
        "profile": "especifico_sala_comercial",
        "min_k": 9,
        "max_k": 20,
        "min_effective_neighbors": 15.0,
        "similarity_weight": 0.60,
        "location_weight": 0.40,
        "max_individual_weight": 0.25,
        "distance_power": 0.65,
        "robust_mad_threshold": 2.50,
    },
    "garagem / vaga": {
        "profile": "conservador_garagem_vaga_v1_12",
        "min_k": 7,
        "max_k": 30,
        "min_effective_neighbors": 5.0,
        "similarity_weight": 2.0 / 3.0,
        "location_weight": 1.0 / 3.0,
        "max_individual_weight": 0.30,
        "distance_power": 1.00,
        "robust_mad_threshold": 2.50,
    },
    "garagem / vaga residencial": {
        "profile": "conservador_garagem_vaga_residencial_v1_12",
        "min_k": 7,
        "max_k": 30,
        "min_effective_neighbors": 5.0,
        "similarity_weight": 2.0 / 3.0,
        "location_weight": 1.0 / 3.0,
        "max_individual_weight": 0.30,
        "distance_power": 1.00,
        "robust_mad_threshold": 2.50,
    },
    "garagem / vaga nao residencial": {
        "profile": "conservador_garagem_vaga_nao_residencial_v1_12",
        "min_k": 7,
        "max_k": 30,
        "min_effective_neighbors": 5.0,
        "similarity_weight": 2.0 / 3.0,
        "location_weight": 1.0 / 3.0,
        "max_individual_weight": 0.30,
        "distance_power": 1.00,
        "robust_mad_threshold": 2.50,
    },
}


@dataclass(frozen=True)
class ColumnMapping:
    tipo_informacao: str
    finalidade_oferta: str
    valor: str
    area_construida: str | None
    area_privativa: str | None
    latitude: str | None
    longitude: str | None
    siat_area_total_lote: str | None
    testada: str | None
    # Mantido ao final e com default para preservar chamadas posicionais
    # existentes em notebooks que usam a interface anterior.
    ano_construcao: str | None = None


@dataclass(frozen=True)
class PreparationResult:
    data: pd.DataFrame
    discount: float
    diagnostics: dict[str, Any]
    excluded_data: pd.DataFrame
    flagged_data: pd.DataFrame


@dataclass(frozen=True)
class EstimateResult:
    estimated_unit_value: float
    estimated_total_value: float
    weighted_std_unit: float
    effective_neighbors: float
    neighbors: pd.DataFrame
    active_features: list[str]
    geographic_scale_km: float
    diagnostics: dict[str, Any]
    local_excluded_data: pd.DataFrame


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    replacements = str.maketrans(
        "áàãâäéèêëíìîïóòõôöúùûüç",
        "aaaaaeeeeiiiiooooouuuuc",
    )
    return " ".join(text.translate(replacements).split())




AREA_REGIME_PRIVATE = "privativa"
AREA_REGIME_TOTAL_BUILT = "total_construida"
AREA_MODE_AUTOMATIC = "automatica"
AREA_MODE_PRIVATE = "privativa"
AREA_MODE_TOTAL_BUILT = "total_construida"


def preferred_area_regime(area_preference: Any) -> str:
    normalized = normalize_text(area_preference)
    if normalized == "construida":
        return AREA_REGIME_TOTAL_BUILT
    return AREA_REGIME_PRIVATE


def area_floor_is_compatible(
    area_preference: Any,
    area_regime: Any,
) -> bool:
    preference = normalize_text(area_preference)
    regime = normalize_text(area_regime)

    if preference == "privativa ou construida":
        return regime in {
            AREA_REGIME_PRIVATE,
            AREA_REGIME_TOTAL_BUILT,
        }
    if preference == "construida":
        return regime == AREA_REGIME_TOTAL_BUILT
    if preference == "privativa":
        return regime == AREA_REGIME_PRIVATE
    return True


def choose_area_regime(
    requested_mode: Any,
    preferred_regime_value: Any,
    private_available: bool,
    total_built_available: bool,
    private_count: int,
    total_built_count: int,
    minimum_required: int,
) -> dict[str, Any]:
    """
    Seleciona um único denominador de área para toda a estimativa.

    No modo automático, o regime preferencial é mantido apenas quando
    alcança o K inicial depois de todos os filtros. Caso contrário, o
    regime alternativo é utilizado se atingir o mesmo mínimo. Quando
    nenhum regime é suficiente, a seleção automática é interrompida.
    """
    requested = normalize_text(requested_mode)
    preferred = normalize_text(preferred_regime_value)
    minimum = max(int(minimum_required), 2)

    availability = {
        AREA_REGIME_PRIVATE: bool(private_available),
        AREA_REGIME_TOTAL_BUILT: bool(total_built_available),
    }
    counts = {
        AREA_REGIME_PRIVATE: max(int(private_count), 0),
        AREA_REGIME_TOTAL_BUILT: max(int(total_built_count), 0),
    }

    aliases = {
        "automatica": AREA_MODE_AUTOMATIC,
        "automatico": AREA_MODE_AUTOMATIC,
        "automatic": AREA_MODE_AUTOMATIC,
        "area privativa": AREA_MODE_PRIVATE,
        "privativa": AREA_MODE_PRIVATE,
        "area total construida": AREA_MODE_TOTAL_BUILT,
        "area total/construida": AREA_MODE_TOTAL_BUILT,
        "total construida": AREA_MODE_TOTAL_BUILT,
        "total_construida": AREA_MODE_TOTAL_BUILT,
    }
    mode = aliases.get(requested, requested)
    if preferred not in availability:
        preferred = AREA_REGIME_PRIVATE
    alternative = (
        AREA_REGIME_TOTAL_BUILT
        if preferred == AREA_REGIME_PRIVATE
        else AREA_REGIME_PRIVATE
    )

    def _result(
        regime: str,
        reason: str,
        automatic: bool,
    ) -> dict[str, Any]:
        count = counts[regime]
        return {
            "regime": regime,
            "reason": reason,
            "automatic": automatic,
            "sample_count": count,
            "minimum_required": minimum,
            "sample_sufficient": count >= minimum,
            "private_count": counts[AREA_REGIME_PRIVATE],
            "total_built_count": counts[AREA_REGIME_TOTAL_BUILT],
        }

    if mode in {AREA_MODE_PRIVATE, AREA_MODE_TOTAL_BUILT}:
        if not availability[mode]:
            label = (
                "área privativa"
                if mode == AREA_MODE_PRIVATE
                else "área total/construída"
            )
            raise ValueError(
                f"O regime solicitado não está disponível: {label}."
            )
        if counts[mode] < 2:
            raise ValueError(
                "O regime solicitado não possui ao menos dois "
                "comparáveis válidos após os filtros."
            )
        return _result(
            mode,
            "regime escolhido manualmente",
            False,
        )

    if mode != AREA_MODE_AUTOMATIC:
        raise ValueError("Modo de seleção da área inválido.")

    for regime, reason in (
        (
            preferred,
            "regime preferencial com amostra suficiente",
        ),
        (
            alternative,
            "regime alternativo selecionado porque o preferencial "
            "não estava disponível ou era insuficiente",
        ),
    ):
        if availability[regime] and counts[regime] >= minimum:
            return _result(regime, reason, True)

    available_descriptions = []
    if availability[AREA_REGIME_PRIVATE]:
        available_descriptions.append(
            f"privativa: {counts[AREA_REGIME_PRIVATE]}"
        )
    if availability[AREA_REGIME_TOTAL_BUILT]:
        available_descriptions.append(
            "total/construída: "
            f"{counts[AREA_REGIME_TOTAL_BUILT]}"
        )
    detail = "; ".join(available_descriptions) or "nenhum regime disponível"
    raise ValueError(
        "Nenhum regime de área atingiu o K inicial após os filtros "
        f"({minimum}). Contagens: {detail}. Selecione manualmente um "
        "regime apenas se aceitar trabalhar com amostra reduzida."
    )


def calibrated_parameters_for_purpose(
    purpose: Any,
) -> dict[str, float | int | str]:
    normalized = normalize_text(purpose)
    selected = CALIBRATED_PURPOSE_PARAMETERS.get(
        normalized,
        CALIBRATED_GLOBAL_PARAMETERS,
    )
    return dict(selected)


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[Rr]\$", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .str.extract(r"([-+]?\d[\d.,]*)", expand=False)
    )
    has_comma = cleaned.str.contains(",", regex=False, na=False)
    has_dot = cleaned.str.contains(".", regex=False, na=False)
    both = has_comma & has_dot

    comma_is_decimal = both & (
        cleaned.str.rfind(",") > cleaned.str.rfind(".")
    )
    dot_is_decimal = both & ~comma_is_decimal
    cleaned.loc[comma_is_decimal] = (
        cleaned.loc[comma_is_decimal]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned.loc[dot_is_decimal] = cleaned.loc[dot_is_decimal].str.replace(
        ",", "", regex=False
    )
    cleaned.loc[has_comma & ~has_dot] = cleaned.loc[
        has_comma & ~has_dot
    ].str.replace(",", ".", regex=False)
    dot_thousands = (
        has_dot
        & ~has_comma
        & cleaned.str.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+", na=False)
    )
    cleaned.loc[dot_thousands] = cleaned.loc[dot_thousands].str.replace(
        ".", "", regex=False
    )
    return pd.to_numeric(cleaned, errors="coerce")


def is_valid_construction_year(value: Any) -> bool:
    """Valida um ano civil inteiro dentro da faixa operacional do cadastro."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(numeric)
        and MIN_CONSTRUCTION_YEAR <= numeric <= MAX_CONSTRUCTION_YEAR
        and np.isclose(numeric, round(numeric), atol=1e-9)
    )


def valid_construction_year_mask(series: pd.Series) -> pd.Series:
    """Retorna a máscara auditável de anos da construção utilizáveis."""
    numeric = to_numeric(series)
    values = numeric.to_numpy(dtype=float)
    valid = (
        np.isfinite(values)
        & (values >= MIN_CONSTRUCTION_YEAR)
        & (values <= MAX_CONSTRUCTION_YEAR)
        & np.isclose(values, np.round(values), atol=1e-9)
    )
    return pd.Series(valid, index=series.index, dtype=bool)


def _positive_values(values: Iterable[float]) -> np.ndarray:
    # Converte também dtypes anuláveis do pandas, evitando que pd.NA
    # interrompa o cálculo do fator de oferta.
    numeric = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    )
    arr = numeric.to_numpy(dtype=float, na_value=np.nan)
    return arr[np.isfinite(arr) & (arr > 0)]


def estimate_offer_discount(
    itbi_unit_values: Iterable[float],
    offer_unit_values: Iterable[float],
    cap: float = 0.20,
    offers_only_discount: float = 0.10,
) -> tuple[float, dict[str, Any]]:
    """
    Define o fator de oferta por duas regras mutuamente exclusivas:

    1. Quando existem apenas ofertas, aplica desconto convencional de 10%.
    2. Quando há pelo menos dois dados de ITBI e dois de oferta, calcula:

           mediana(1 - VU_ITBI / VU_Oferta)

       em quantis pareados, usando ``cap`` somente como teto do desconto
       empiricamente calculado.

    O desconto convencional de 10% não é produzido pela razão entre as
    distribuições e, portanto, não utiliza o teto empírico de 20%.
    """
    itbi = _positive_values(itbi_unit_values)
    offers = _positive_values(offer_unit_values)

    diagnostics: dict[str, Any] = {
        "n_itbi_discount": int(itbi.size),
        "n_offer_discount": int(offers.size),
        "discount_cap": float(cap),
        "offers_only_discount": float(offers_only_discount),
        "discount_was_capped": False,
    }

    # Regra subsidiária: a amostra contém ofertas, mas nenhuma Guia ITBI.
    if itbi.size == 0 and offers.size > 0:
        diagnostics.update(
            {
                "discount_method": "fator convencional para amostra somente de ofertas",
                "discount_source": "offers_only_fallback",
                "raw_discount_median": np.nan,
                "quantiles_used": 0,
                "discount_warning": (
                    f"A amostra contém somente ofertas. Foi aplicado o desconto "
                    f"convencional de {offers_only_discount:.0%}; o teto de "
                    f"{cap:.0%} é reservado ao desconto calculado pela razão "
                    "entre Guias ITBI e ofertas."
                ),
            }
        )
        return float(offers_only_discount), diagnostics

    # Sem ofertas, não há fator de oferta a aplicar.
    if offers.size == 0:
        diagnostics.update(
            {
                "discount_method": "sem ofertas para ajustar",
                "discount_source": "no_offers",
                "raw_discount_median": np.nan,
                "quantiles_used": 0,
            }
        )
        return 0.0, diagnostics

    # Existe algum ITBI, mas não há quantidade suficiente para quantis pareados.
    if itbi.size < 2 or offers.size < 2:
        diagnostics.update(
            {
                "discount_method": "amostra insuficiente para razão empírica",
                "discount_source": "insufficient_mixed_sample",
                "raw_discount_median": np.nan,
                "quantiles_used": 0,
                "discount_warning": (
                    "Desconto igual a zero: para calcular a razão empírica são "
                    "necessários ao menos dois dados de Guia ITBI e dois de "
                    "Oferta. O desconto convencional de 10% é aplicado somente "
                    "quando não existe nenhuma Guia ITBI."
                ),
            }
        )
        return 0.0, diagnostics

    n_quantiles = int(np.clip(min(itbi.size, offers.size), 5, 19))
    quantiles = np.linspace(0.10, 0.90, n_quantiles)
    q_itbi = np.quantile(itbi, quantiles)
    q_offer = np.quantile(offers, quantiles)

    valid = np.isfinite(q_itbi) & np.isfinite(q_offer) & (q_offer > 0)
    raw_discounts = 1.0 - (q_itbi[valid] / q_offer[valid])
    raw_discounts = raw_discounts[np.isfinite(raw_discounts)]

    if raw_discounts.size == 0:
        diagnostics.update(
            {
                "discount_method": "falha na razão empírica",
                "discount_source": "empirical_failure",
                "raw_discount_median": np.nan,
                "quantiles_used": 0,
                "discount_warning": "Não foi possível calcular o desconto.",
            }
        )
        return 0.0, diagnostics

    raw_median = float(np.median(raw_discounts))
    discount = float(np.clip(raw_median, 0.0, cap))
    diagnostics.update(
        {
            "discount_method": "mediana de descontos em quantis pareados",
            "discount_source": "empirical_ratio",
            "raw_discount_median": raw_median,
            "raw_discount_mean_diagnostic": float(np.mean(raw_discounts)),
            "quantiles_used": int(raw_discounts.size),
            "discount_was_capped": bool(raw_median > cap),
        }
    )

    if raw_median > cap:
        diagnostics["discount_warning"] = (
            f"O desconto bruto de {raw_median:.1%} foi limitado ao teto de "
            f"{cap:.0%}."
        )

    return discount, diagnostics


def _parse_registration_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    numeric = pd.to_numeric(series, errors="coerce")
    serial_mask = parsed.isna() & numeric.between(1, 100000)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime(
            numeric.loc[serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return parsed


def _valid_registration_key(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_text).astype("string")
    ignored = {
        "",
        "0",
        "0.0",
        "nan",
        "none",
        "<na>",
        "sem inscricao",
        "sem inscrição",
    }
    return normalized.where(~normalized.isin(ignored), "")


def _normalized_money_key(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return numeric.map(
        lambda value: (
            f"{float(value):.2f}"
            if pd.notna(value) and np.isfinite(value) and float(value) > 0
            else ""
        )
    ).astype("string")


def deduplicate_offers(
    data: pd.DataFrame,
    date_column: str | None,
    identifier_columns: Iterable[str],
    registration_column: str | None = None,
    value_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deduplicação hierárquica.

    1. Prioriza tipo da informação + inscrição SIAT + valor.
       Essa etapa alcança Guias ITBI e ofertas.
    2. Nas ofertas restantes, usa identificadores genuínos de anúncio como
       fallback. Colunas genéricas de origem não devem ser fornecidas.
    """
    diagnostics: dict[str, Any] = {
        "offer_deduplication_enabled": True,
        "offer_duplicates_removed": 0,
        "market_duplicates_removed_primary": 0,
        "market_duplicate_groups_primary": 0,
        "offer_duplicates_removed_fallback": 0,
        "offer_duplicate_groups_fallback": 0,
        "offer_rows_without_identifier": 0,
        "offer_rows_without_valid_date": 0,
        "offer_deduplication_date_column": date_column or "",
        "offer_deduplication_identifier_columns": "",
        "deduplication_primary_columns": "",
    }

    cleaned = data.copy()

    if date_column and date_column in cleaned.columns:
        cleaned["_data_registro_deduplicacao"] = _parse_registration_dates(
            cleaned[date_column]
        )
    else:
        cleaned["_data_registro_deduplicacao"] = pd.NaT
        diagnostics["deduplication_warning"] = (
            "A coluna de data não foi encontrada. Em empates, foi mantida a "
            "última linha do arquivo."
        )

    diagnostics["offer_rows_without_valid_date"] = int(
        cleaned.loc[
            cleaned["_tipo_norm"].eq(TIPO_OFERTA),
            "_data_registro_deduplicacao",
        ].isna().sum()
    )

    # Etapa primária: tipo + inscrição + valor.
    primary_available = (
        registration_column
        and value_column
        and registration_column in cleaned.columns
        and value_column in cleaned.columns
    )

    if primary_available:
        registration_key = _valid_registration_key(
            cleaned[registration_column]
        )
        value_key = _normalized_money_key(cleaned[value_column])
        valid_primary = registration_key.ne("") & value_key.ne("")

        primary_key = pd.Series("", index=cleaned.index, dtype="string")
        primary_key.loc[valid_primary] = (
            cleaned.loc[valid_primary, "_tipo_norm"].astype("string")
            + "::"
            + registration_key.loc[valid_primary]
            + "::"
            + value_key.loc[valid_primary]
        )
        cleaned["_chave_deduplicacao_prioritaria"] = primary_key

        primary_candidates = cleaned.loc[valid_primary].copy()
        if not primary_candidates.empty:
            primary_candidates = primary_candidates.sort_values(
                [
                    "_chave_deduplicacao_prioritaria",
                    "_data_registro_deduplicacao",
                    "_row_excel",
                ],
                ascending=[True, True, True],
                na_position="first",
                kind="mergesort",
            )
            group_sizes = primary_candidates.groupby(
                "_chave_deduplicacao_prioritaria"
            ).size()
            diagnostics["market_duplicate_groups_primary"] = int(
                (group_sizes > 1).sum()
            )

            primary_duplicated = primary_candidates.duplicated(
                subset=["_chave_deduplicacao_prioritaria"],
                keep="last",
            )
            primary_removed = primary_candidates.index[primary_duplicated]
            diagnostics["market_duplicates_removed_primary"] = int(
                len(primary_removed)
            )
            cleaned = cleaned.drop(index=primary_removed).copy()

        diagnostics["deduplication_primary_columns"] = (
            f"_tipo_norm, {registration_column}, {value_column}"
        )
    else:
        diagnostics["deduplication_primary_warning"] = (
            "A deduplicação prioritária por inscrição e valor não foi "
            "aplicada porque uma das colunas não foi encontrada."
        )

    # Etapa de fallback: somente ofertas e somente IDs genuínos.
    identifier_columns = [
        column
        for column in identifier_columns
        if column and column in cleaned.columns
    ]
    diagnostics["offer_deduplication_identifier_columns"] = ", ".join(
        identifier_columns
    )

    if not identifier_columns:
        diagnostics["offer_duplicates_removed"] = int(
            diagnostics["market_duplicates_removed_primary"]
        )
        return cleaned, diagnostics

    offers = cleaned.loc[cleaned["_tipo_norm"].eq(TIPO_OFERTA)].copy()
    if offers.empty:
        diagnostics["offer_duplicates_removed"] = int(
            diagnostics["market_duplicates_removed_primary"]
        )
        return cleaned, diagnostics

    key = pd.Series("", index=offers.index, dtype="string")
    source = pd.Series("", index=offers.index, dtype="string")
    ignored = {
        "",
        "nan",
        "none",
        "<na>",
        "0",
        "transacao",
        "transação",
        "oferta",
        "crawler",
        "auxiliador",
        "malcon",
        "credito real",
        "pmi",
    }

    for column in identifier_columns:
        normalized = offers[column].map(normalize_text).astype("string")
        valid = ~normalized.isin(ignored)
        fill = key.eq("") & valid
        key.loc[fill] = normalize_text(column) + "::" + normalized.loc[fill]
        source.loc[fill] = column

    offers["_chave_oferta_deduplicacao"] = key
    offers["_fonte_chave_oferta"] = source
    without_id = key.eq("")
    diagnostics["offer_rows_without_identifier"] = int(without_id.sum())

    candidates = offers.loc[~without_id].copy()
    if candidates.empty:
        diagnostics["offer_duplicates_removed"] = int(
            diagnostics["market_duplicates_removed_primary"]
        )
        return cleaned, diagnostics

    candidates = candidates.sort_values(
        [
            "_chave_oferta_deduplicacao",
            "_data_registro_deduplicacao",
            "_row_excel",
        ],
        ascending=[True, True, True],
        na_position="first",
        kind="mergesort",
    )
    group_sizes = candidates.groupby("_chave_oferta_deduplicacao").size()
    diagnostics["offer_duplicate_groups_fallback"] = int(
        (group_sizes > 1).sum()
    )

    duplicated = candidates.duplicated(
        subset=["_chave_oferta_deduplicacao"],
        keep="last",
    )
    removed_indices = candidates.index[duplicated]
    diagnostics["offer_duplicates_removed_fallback"] = int(
        len(removed_indices)
    )
    cleaned = cleaned.drop(index=removed_indices).copy()

    kept = candidates.loc[
        ~duplicated,
        [
            "_chave_oferta_deduplicacao",
            "_fonte_chave_oferta",
        ],
    ]
    for column in kept.columns:
        cleaned.loc[kept.index, column] = kept[column]

    diagnostics["offer_duplicates_removed"] = int(
        diagnostics["market_duplicates_removed_primary"]
        + diagnostics["offer_duplicates_removed_fallback"]
    )
    diagnostics["offer_duplicate_groups"] = int(
        diagnostics["market_duplicate_groups_primary"]
        + diagnostics["offer_duplicate_groups_fallback"]
    )
    return cleaned, diagnostics



PURPOSE_UNIT_VALUE_FLOORS: dict[str, float] = {
    'apartamento': 1200.00,
    'cobertura': 900.00,
    'flat / apart-hotel': 1200.00,
    'casa / residencia': 800.00,
    'loja': 700.00,
    'loja em galeria': 1000.00,
    'loja em shopping': 1200.00,
    'sala comercial': 900.00,
    'imovel comercial': 850.00,
    'galpao / deposito': 650.00,
    'terreno': 200.00,
    'gleba': 20.00,
    'construcao em area de gleba': 100.00,
    'garagem / vaga': 150.00,
    'garagem / vaga residencial': 200.00,
    'garagem / vaga nao residencial': 150.00,
    'hotel': 400.00,
    'imovel especial': 400.00,
}

PREFILTER_SYMBOLIC_VALUE_MAX = 1.00
PREFILTER_MIN_ITBI_FOR_DIAGNOSTIC = 8
PREFILTER_MIN_ITBI_FOR_AUTO_EXCLUSION = 15
PREFILTER_ALERT_MODIFIED_Z = 2.5
PREFILTER_EXCLUDE_MODIFIED_Z = 3.5
PREFILTER_IQR_OUTER_MULTIPLIER = 3.0


_NATURE_COLUMN_NAMES = {
    "natureza_transacao",
    "natureza_da_transacao",
    "natureza_transmissao",
    "natureza_da_transmissao",
    "tipo_transacao",
    "tipo_de_transacao",
    "tipo_transmissao",
    "tipo_de_transmissao",
    "descricao_transacao",
    "descricao_da_transacao",
    "descricao_transmissao",
    "descricao_da_transmissao",
    "itbi_natureza",
    "itbi_tipo_transacao",
    "itbi_tipo_transmissao",
    "negocio_juridico",
    "natureza_negocio_juridico",
}

_FRACTION_COLUMN_NAMES = {
    "fracao_transmitida",
    "fracao_ideal",
    "percentual_transmitido",
    "percentual_transmissao",
    "percentual_do_imovel",
    "quota_transmitida",
    "quota_parte",
    "parte_ideal",
    "quinhao",
}

_PROPERTY_COUNT_COLUMN_NAMES = {
    "quantidade_imoveis",
    "quantidade_de_imoveis",
    "qtd_imoveis",
    "numero_imoveis",
    "numero_de_imoveis",
    "imoveis_transmitidos",
}

_NON_MARKET_NATURE_PATTERN = (
    r"\b(?:"
    r"doacao|heranca|inventario|partilha|cessao gratuita|"
    r"transmissao gratuita|usufruto|nua propriedade|"
    r"integralizacao de capital|incorporacao de capital|"
    r"fracao|parte ideal|quota parte|quinhao"
    r")\b"
)


def _canonical_column_name(value: Any) -> str:
    normalized = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return "_".join(tokens)


def _matching_columns(
    data: pd.DataFrame,
    accepted_names: set[str],
) -> list[str]:
    return [
        str(column)
        for column in data.columns
        if _canonical_column_name(column) in accepted_names
    ]


def _append_reason(
    reasons: pd.Series,
    mask: pd.Series | np.ndarray,
    reason: str,
) -> pd.Series:
    boolean_mask = pd.Series(mask, index=reasons.index).fillna(False).astype(bool)
    empty = reasons.eq("")
    reasons.loc[boolean_mask & empty] = reason
    reasons.loc[boolean_mask & ~empty] = (
        reasons.loc[boolean_mask & ~empty] + "; " + reason
    )
    return reasons


def _parse_transmitted_share(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    text = series.astype("string").str.strip()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    fraction_match = text.str.extract(
        r"^\s*(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)\s*$"
    )
    numerator = to_numeric(fraction_match[0])
    denominator = to_numeric(fraction_match[1])
    fraction_valid = (
        numerator.notna()
        & denominator.notna()
        & denominator.gt(0)
    )
    result.loc[fraction_valid] = (
        numerator.loc[fraction_valid] / denominator.loc[fraction_valid]
    )

    percent_text = text.str.contains("%", regex=False, na=False)
    numeric_text = (
        text.str.replace("%", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    numeric = to_numeric(numeric_text)

    canonical_name = _canonical_column_name(column_name)
    percentage_column = "percentual" in canonical_name

    unresolved = result.isna() & numeric.notna()
    as_percent = unresolved & (
        percent_text
        | percentage_column
        | numeric.gt(1.0)
    )
    result.loc[as_percent] = numeric.loc[as_percent] / 100.0

    as_fraction = unresolved & ~as_percent
    result.loc[as_fraction] = numeric.loc[as_fraction]
    return result


def _ordered_control_columns(data: pd.DataFrame) -> pd.DataFrame:
    priority = [
        "_row_excel",
        "_etapa_controle",
        "_motivo_exclusao",
        "_motivo_alerta",
        "_valor_unitario_original",
        "_piso_finalidade_vu",
        "_log_valor_unitario",
        "_escore_robusto_prefiltro",
        "_limite_inferior_vu_prefiltro",
        "_limite_superior_vu_prefiltro",
    ]
    ordered = [column for column in priority if column in data.columns]
    ordered.extend(column for column in data.columns if column not in ordered)
    return data.loc[:, ordered].copy()


def _safe_market_prefilter(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    reference_area_column: str,
    value_kind: str,
    selected_purpose: str,
    floor_purpose: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = data.copy()
    reasons = pd.Series("", index=working.index, dtype="string")

    raw_value = pd.to_numeric(working[mapping.valor], errors="coerce")
    reference_area = pd.to_numeric(
        working[reference_area_column],
        errors="coerce",
    )

    value_kind_norm = normalize_text(value_kind)
    total_value_mode = value_kind_norm == "valor total"

    reasons = _append_reason(
        reasons,
        ~np.isfinite(raw_value) | raw_value.le(0),
        "Valor informado inválido, nulo ou não positivo",
    )
    reasons = _append_reason(
        reasons,
        np.isfinite(raw_value)
        & raw_value.gt(0)
        & raw_value.le(PREFILTER_SYMBOLIC_VALUE_MAX),
        "Valor informado simbólico ou sem representatividade mercadológica",
    )
    reasons = _append_reason(
        reasons,
        ~np.isfinite(reference_area) | reference_area.le(0),
        "Área de referência inválida, nula ou não positiva",
    )

    if total_value_mode:
        working["_valor_unitario_original"] = raw_value / reference_area
    elif value_kind_norm in {
        "valor unitario",
        "valor unitario por m2",
        "valor unitario por m²",
    }:
        working["_valor_unitario_original"] = raw_value
    else:
        raise ValueError("Natureza do valor inválida.")

    numeric_unit = pd.to_numeric(
        working["_valor_unitario_original"],
        errors="coerce",
    )
    working["_valor_unitario_original"] = numeric_unit.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    reasons = _append_reason(
        reasons,
        ~np.isfinite(working["_valor_unitario_original"])
        | working["_valor_unitario_original"].le(0),
        "Valor unitário inválido ou não positivo",
    )

    purpose_floor_name = floor_purpose or selected_purpose
    purpose_norm = normalize_text(purpose_floor_name)
    purpose_floor = float(
        PURPOSE_UNIT_VALUE_FLOORS.get(purpose_norm, 0.0)
    )
    working["_piso_finalidade_vu"] = purpose_floor

    purpose_floor_mask = (
        purpose_floor > 0
        and np.isfinite(purpose_floor)
    )
    if purpose_floor_mask:
        below_purpose_floor = (
            np.isfinite(working["_valor_unitario_original"])
            & working["_valor_unitario_original"].gt(0)
            & working["_valor_unitario_original"].lt(purpose_floor)
        )
        reasons = _append_reason(
            reasons,
            below_purpose_floor,
            (
                "Valor unitário inferior ao piso da finalidade "
                f"({purpose_floor:.2f} R$/m²)"
            ),
        )
    else:
        below_purpose_floor = pd.Series(
            False,
            index=working.index,
        )

    is_itbi = working["_tipo_norm"].eq(TIPO_ITBI)

    nature_columns = _matching_columns(working, _NATURE_COLUMN_NAMES)
    non_market_mask = pd.Series(False, index=working.index)
    for column in nature_columns:
        nature_text = working[column].map(normalize_text)
        non_market_mask |= nature_text.str.contains(
            _NON_MARKET_NATURE_PATTERN,
            regex=True,
            na=False,
        )
    reasons = _append_reason(
        reasons,
        is_itbi & non_market_mask,
        "Natureza da transmissão não mercadológica ou interesse parcial",
    )

    fraction_columns = _matching_columns(working, _FRACTION_COLUMN_NAMES)
    partial_interest_mask = pd.Series(False, index=working.index)
    for column in fraction_columns:
        share = _parse_transmitted_share(working[column], column)
        partial_interest_mask |= share.gt(0) & share.lt(0.999999)
    reasons = _append_reason(
        reasons,
        is_itbi & partial_interest_mask,
        "Transmissão parcial identificada sem integralização segura",
    )

    property_count_columns = _matching_columns(
        working,
        _PROPERTY_COUNT_COLUMN_NAMES,
    )
    multiple_property_mask = pd.Series(False, index=working.index)
    for column in property_count_columns:
        count = to_numeric(working[column])
        multiple_property_mask |= count.gt(1)
    reasons = _append_reason(
        reasons,
        is_itbi & multiple_property_mask,
        "Transmissão conjunta de múltiplos imóveis sem valor individualizado",
    )

    deterministic_mask = reasons.ne("")
    deterministic_excluded = working.loc[deterministic_mask].copy()
    if not deterministic_excluded.empty:
        deterministic_excluded["_etapa_controle"] = "regra determinística"
        deterministic_excluded["_motivo_exclusao"] = reasons.loc[
            deterministic_mask
        ].astype(str)
        deterministic_excluded["_motivo_alerta"] = ""
        deterministic_excluded["_log_valor_unitario"] = np.where(
            deterministic_excluded["_valor_unitario_original"].gt(0),
            np.log(
                deterministic_excluded["_valor_unitario_original"].where(
                    deterministic_excluded["_valor_unitario_original"].gt(0)
                )
            ),
            np.nan,
        )
        deterministic_excluded["_escore_robusto_prefiltro"] = np.nan
        deterministic_excluded["_limite_inferior_vu_prefiltro"] = np.nan
        deterministic_excluded["_limite_superior_vu_prefiltro"] = np.nan

    kept = working.loc[~deterministic_mask].copy()
    kept["_log_valor_unitario"] = np.log(
        kept["_valor_unitario_original"].astype(float)
    )
    kept["_escore_robusto_prefiltro"] = np.nan
    kept["_limite_inferior_vu_prefiltro"] = np.nan
    kept["_limite_superior_vu_prefiltro"] = np.nan

    itbi_index = kept.index[kept["_tipo_norm"].eq(TIPO_ITBI)]
    n_itbi = int(len(itbi_index))

    statistical_method = "não aplicado: menos de 8 Guias ITBI"
    auto_exclusion_enabled = False
    robust_median_log = np.nan
    robust_mad_log = np.nan
    lower_vu = np.nan
    upper_vu = np.nan

    extreme_mask = pd.Series(False, index=kept.index)
    attention_mask = pd.Series(False, index=kept.index)

    if n_itbi >= PREFILTER_MIN_ITBI_FOR_DIAGNOSTIC:
        log_values = kept.loc[itbi_index, "_log_valor_unitario"].astype(float)
        robust_median_log = float(np.median(log_values))
        absolute_deviation = np.abs(log_values - robust_median_log)
        robust_mad_log = float(np.median(absolute_deviation))

        if np.isfinite(robust_mad_log) and robust_mad_log > 1e-12:
            modified_z = (
                0.6745
                * (log_values - robust_median_log)
                / robust_mad_log
            )
            kept.loc[itbi_index, "_escore_robusto_prefiltro"] = modified_z

            lower_log = (
                robust_median_log
                - PREFILTER_EXCLUDE_MODIFIED_Z
                * robust_mad_log
                / 0.6745
            )
            upper_log = (
                robust_median_log
                + PREFILTER_EXCLUDE_MODIFIED_Z
                * robust_mad_log
                / 0.6745
            )
            lower_vu = float(np.exp(lower_log))
            upper_vu = float(np.exp(upper_log))

            absolute_score = modified_z.abs()
            extreme_index = absolute_score.index[
                absolute_score.gt(PREFILTER_EXCLUDE_MODIFIED_Z)
            ]
            attention_index = absolute_score.index[
                absolute_score.gt(PREFILTER_ALERT_MODIFIED_Z)
                & absolute_score.le(PREFILTER_EXCLUDE_MODIFIED_Z)
            ]
            extreme_mask.loc[extreme_index] = True
            attention_mask.loc[attention_index] = True
            statistical_method = (
                "escore Z modificado sobre ln(valor unitário)"
            )
        else:
            q1, q3 = np.quantile(log_values, [0.25, 0.75])
            iqr = float(q3 - q1)
            if np.isfinite(iqr) and iqr > 1e-12:
                lower_log = q1 - PREFILTER_IQR_OUTER_MULTIPLIER * iqr
                upper_log = q3 + PREFILTER_IQR_OUTER_MULTIPLIER * iqr
                lower_vu = float(np.exp(lower_log))
                upper_vu = float(np.exp(upper_log))
                outside = (
                    log_values.lt(lower_log)
                    | log_values.gt(upper_log)
                )
                extreme_mask.loc[outside.index[outside]] = True
                statistical_method = (
                    "cercas externas de 3×IQR sobre ln(valor unitário), "
                    "utilizadas porque o MAD foi zero"
                )
            else:
                statistical_method = (
                    "não aplicado: MAD e IQR iguais a zero"
                )

        kept.loc[itbi_index, "_limite_inferior_vu_prefiltro"] = lower_vu
        kept.loc[itbi_index, "_limite_superior_vu_prefiltro"] = upper_vu
        auto_exclusion_enabled = (
            n_itbi >= PREFILTER_MIN_ITBI_FOR_AUTO_EXCLUSION
        )

    flagged_mask = attention_mask.copy()
    if not auto_exclusion_enabled:
        flagged_mask |= extreme_mask

    flagged = kept.loc[flagged_mask].copy()
    if not flagged.empty:
        flagged["_etapa_controle"] = "alerta robusto"
        flagged["_motivo_exclusao"] = ""
        flagged["_motivo_alerta"] = np.where(
            extreme_mask.loc[flagged.index],
            (
                "Valor unitário extremo identificado; não excluído "
                "automaticamente porque a amostra possui menos de 15 "
                "Guias ITBI"
            ),
            (
                "Valor unitário na faixa de atenção robusta "
                "(2,5 < |M| ≤ 3,5)"
            ),
        )

    statistical_excluded = kept.iloc[0:0].copy()
    if auto_exclusion_enabled and extreme_mask.any():
        statistical_excluded = kept.loc[extreme_mask].copy()
        statistical_excluded["_etapa_controle"] = (
            "filtro robusto automático"
        )
        statistical_excluded["_motivo_exclusao"] = (
            "Valor unitário extremo no pré-filtro robusto"
        )
        statistical_excluded["_motivo_alerta"] = ""
        kept = kept.loc[~extreme_mask].copy()

    excluded_frames = [
        frame
        for frame in (deterministic_excluded, statistical_excluded)
        if not frame.empty
    ]
    if excluded_frames:
        excluded = pd.concat(excluded_frames, ignore_index=True, sort=False)
        excluded = _ordered_control_columns(excluded)
    else:
        excluded = _ordered_control_columns(
            working.iloc[0:0].assign(
                _etapa_controle=pd.Series(dtype="string"),
                _motivo_exclusao=pd.Series(dtype="string"),
                _motivo_alerta=pd.Series(dtype="string"),
                _log_valor_unitario=pd.Series(dtype=float),
                _escore_robusto_prefiltro=pd.Series(dtype=float),
                _limite_inferior_vu_prefiltro=pd.Series(dtype=float),
                _limite_superior_vu_prefiltro=pd.Series(dtype=float),
            )
        )

    if not flagged.empty:
        flagged = _ordered_control_columns(flagged)
    else:
        flagged = _ordered_control_columns(
            working.iloc[0:0].assign(
                _etapa_controle=pd.Series(dtype="string"),
                _motivo_exclusao=pd.Series(dtype="string"),
                _motivo_alerta=pd.Series(dtype="string"),
                _log_valor_unitario=pd.Series(dtype=float),
                _escore_robusto_prefiltro=pd.Series(dtype=float),
                _limite_inferior_vu_prefiltro=pd.Series(dtype=float),
                _limite_superior_vu_prefiltro=pd.Series(dtype=float),
            )
        )

    reason_counts = (
        excluded["_motivo_exclusao"].value_counts().to_dict()
        if not excluded.empty
        else {}
    )

    diagnostics = {
        "prefilter_enabled": True,
        "prefilter_symbolic_value_max": PREFILTER_SYMBOLIC_VALUE_MAX,
        "prefilter_input_rows": int(len(working)),
        "prefilter_deterministic_excluded": int(
            len(deterministic_excluded)
        ),
        "prefilter_statistical_excluded": int(
            len(statistical_excluded)
        ),
        "prefilter_total_excluded": int(len(excluded)),
        "prefilter_flagged": int(len(flagged)),
        "prefilter_itbi_before": int(is_itbi.sum()),
        "prefilter_itbi_after": int(
            kept["_tipo_norm"].eq(TIPO_ITBI).sum()
        ),
        "prefilter_itbi_for_robust_analysis": n_itbi,
        "prefilter_auto_exclusion_enabled": bool(
            auto_exclusion_enabled
        ),
        "prefilter_statistical_method": statistical_method,
        "prefilter_log_median": robust_median_log,
        "prefilter_log_mad": robust_mad_log,
        "prefilter_lower_vu": lower_vu,
        "prefilter_upper_vu": upper_vu,
        "prefilter_alert_modified_z": PREFILTER_ALERT_MODIFIED_Z,
        "prefilter_exclude_modified_z": PREFILTER_EXCLUDE_MODIFIED_Z,
        "prefilter_nature_columns_detected": nature_columns,
        "prefilter_fraction_columns_detected": fraction_columns,
        "prefilter_property_count_columns_detected": (
            property_count_columns
        ),
        "prefilter_exclusion_reasons": reason_counts,
        "purpose_unit_value_floor": purpose_floor,
        "purpose_floor_excluded": int(
            pd.Series(
                below_purpose_floor,
                index=working.index,
            ).fillna(False).sum()
        ),
        "purpose_floor_name": purpose_floor_name,
        "market_segment": selected_purpose,
    }
    return kept, excluded, flagged, diagnostics

def validate_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> None:
    required = [
        mapping.tipo_informacao,
        mapping.finalidade_oferta,
        mapping.valor,
    ]
    optional = [
        mapping.latitude,
        mapping.longitude,
        mapping.area_construida,
        mapping.area_privativa,
        mapping.siat_area_total_lote,
        mapping.testada,
        mapping.ano_construcao,
    ]
    missing = [column for column in required if column not in df.columns]
    missing += [
        column for column in optional if column is not None and column not in df.columns
    ]
    if missing:
        raise ValueError(
            "As seguintes colunas mapeadas não existem: "
            + ", ".join(sorted(set(missing)))
        )


CONFLICT_TYPOLOGICAL_VALUES = ("sim", "moderado")


def _preparation_with_updates(
    result: PreparationResult,
    *,
    diagnostics: dict[str, Any],
    data: pd.DataFrame | None = None,
    excluded_data: pd.DataFrame | None = None,
    flagged_data: pd.DataFrame | None = None,
) -> PreparationResult:
    return PreparationResult(
        data=result.data if data is None else data,
        discount=result.discount,
        diagnostics=diagnostics,
        excluded_data=(
            result.excluded_data
            if excluded_data is None
            else excluded_data
        ),
        flagged_data=(
            result.flagged_data
            if flagged_data is None
            else flagged_data
        ),
    )


def prepare_data(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    selected_purpose: str,
    value_kind: str,
    reference_area_column: str,
    discount_cap: float = 0.20,
    remove_offer_duplicates: bool = True,
    duplicate_date_column: str | None = None,
    duplicate_identifier_columns: Iterable[str] = (),
    duplicate_registration_column: str | None = None,
    duplicate_value_column: str | None = None,
    floor_purpose: str | None = None,
    conflict_column: str | None = None,
    minimum_without_conflict: int | None = None,
    conflict_values: Iterable[str] = CONFLICT_TYPOLOGICAL_VALUES,
) -> PreparationResult:
    validate_mapping(df, mapping)
    if reference_area_column not in df.columns:
        raise ValueError("A coluna de área de referência não existe.")

    source_data = df.copy()
    if "_row_excel" not in source_data.columns:
        source_data["_row_excel"] = np.arange(
            2,
            len(source_data) + 2,
        )

    if (
        conflict_column
        and conflict_column in source_data.columns
        and minimum_without_conflict is not None
        and int(minimum_without_conflict) > 0
    ):
        conflict_norms = {
            normalize_text(value)
            for value in conflict_values
        }
        purpose_mask = source_data[
            mapping.finalidade_oferta
        ].map(normalize_text).eq(normalize_text(selected_purpose))
        type_mask = source_data[
            mapping.tipo_informacao
        ].map(normalize_text).isin([TIPO_ITBI, TIPO_OFERTA])
        conflict_mask = (
            purpose_mask
            & type_mask
            & source_data[conflict_column]
            .map(normalize_text)
            .isin(conflict_norms)
        )
        conflict_rows_available = int(conflict_mask.sum())

        if conflict_rows_available:
            clean_source = source_data.loc[~conflict_mask].copy()
            clean_result: PreparationResult | None = None
            clean_error = ""

            try:
                clean_result = prepare_data(
                    df=clean_source,
                    mapping=mapping,
                    selected_purpose=selected_purpose,
                    value_kind=value_kind,
                    reference_area_column=reference_area_column,
                    discount_cap=discount_cap,
                    remove_offer_duplicates=remove_offer_duplicates,
                    duplicate_date_column=duplicate_date_column,
                    duplicate_identifier_columns=(
                        duplicate_identifier_columns
                    ),
                    duplicate_registration_column=(
                        duplicate_registration_column
                    ),
                    duplicate_value_column=duplicate_value_column,
                    floor_purpose=floor_purpose,
                    conflict_column=None,
                    minimum_without_conflict=None,
                )
            except Exception as exc:
                clean_error = str(exc)

            clean_count = (
                int(len(clean_result.data))
                if clean_result is not None
                else 0
            )
            required = int(minimum_without_conflict)

            if clean_result is not None and clean_count >= required:
                clean_data = clean_result.data.copy()
                clean_data["_uso_conflito_tipologico"] = (
                    "Sem conflito tipológico"
                )

                policy_excluded = source_data.loc[
                    conflict_mask
                ].copy()
                policy_excluded["_etapa_controle"] = (
                    "política de conflito tipológico"
                )
                policy_excluded["_motivo_exclusao"] = (
                    "Conflito tipológico desnecessário: a amostra sem "
                    "conflito atingiu o mínimo configurado"
                )
                policy_excluded["_motivo_alerta"] = ""

                excluded = pd.concat(
                    [
                        clean_result.excluded_data,
                        _ordered_control_columns(policy_excluded),
                    ],
                    ignore_index=True,
                    sort=False,
                )

                diagnostics = {
                    **clean_result.diagnostics,
                    "conflict_policy": (
                        "conflitos excluídos: amostra limpa suficiente"
                    ),
                    "conflict_fallback_used": False,
                    "conflict_minimum_required": required,
                    "conflict_free_prepared_count": clean_count,
                    "conflict_rows_available": (
                        conflict_rows_available
                    ),
                    "conflict_rows_included": 0,
                    "conflict_clean_error": clean_error,
                }
                return _preparation_with_updates(
                    clean_result,
                    diagnostics=diagnostics,
                    data=clean_data,
                    excluded_data=_ordered_control_columns(excluded),
                )

            full_result = prepare_data(
                df=source_data,
                mapping=mapping,
                selected_purpose=selected_purpose,
                value_kind=value_kind,
                reference_area_column=reference_area_column,
                discount_cap=discount_cap,
                remove_offer_duplicates=remove_offer_duplicates,
                duplicate_date_column=duplicate_date_column,
                duplicate_identifier_columns=(
                    duplicate_identifier_columns
                ),
                duplicate_registration_column=(
                    duplicate_registration_column
                ),
                duplicate_value_column=duplicate_value_column,
                floor_purpose=floor_purpose,
                conflict_column=None,
                minimum_without_conflict=None,
            )

            full_data = full_result.data.copy()
            included_mask = full_data[conflict_column].map(
                normalize_text
            ).isin(conflict_norms)
            full_data["_uso_conflito_tipologico"] = np.where(
                included_mask,
                "Incluído por insuficiência da amostra sem conflito",
                "Sem conflito tipológico",
            )
            included_count = int(included_mask.sum())

            conflict_alerts = full_data.loc[included_mask].copy()
            if not conflict_alerts.empty:
                conflict_alerts["_etapa_controle"] = (
                    "política de conflito tipológico"
                )
                conflict_alerts["_motivo_exclusao"] = ""
                conflict_alerts["_motivo_alerta"] = (
                    "Dado com conflito tipológico incluído apenas porque "
                    "a amostra sem conflito era insuficiente"
                )

            flagged = pd.concat(
                [
                    full_result.flagged_data,
                    _ordered_control_columns(conflict_alerts),
                ],
                ignore_index=True,
                sort=False,
            )

            diagnostics = {
                **full_result.diagnostics,
                "conflict_policy": (
                    "contingência: conflitos incluídos por insuficiência"
                ),
                "conflict_fallback_used": True,
                "conflict_minimum_required": required,
                "conflict_free_prepared_count": clean_count,
                "conflict_rows_available": conflict_rows_available,
                "conflict_rows_included": included_count,
                "conflict_clean_error": clean_error,
            }
            return _preparation_with_updates(
                full_result,
                diagnostics=diagnostics,
                data=full_data,
                flagged_data=_ordered_control_columns(flagged),
            )

    data = source_data.copy()
    data["_tipo_norm"] = data[mapping.tipo_informacao].map(normalize_text)
    data["_finalidade_norm"] = data[mapping.finalidade_oferta].map(
        normalize_text
    )

    data = data.loc[
        data["_finalidade_norm"].eq(normalize_text(selected_purpose))
    ].copy()
    data = data.loc[
        data["_tipo_norm"].isin([TIPO_ITBI, TIPO_OFERTA])
    ].copy()

    dedup_diag: dict[str, Any] = {
        "offer_deduplication_enabled": bool(remove_offer_duplicates),
        "offer_duplicates_removed": 0,
        "offer_duplicate_groups": 0,
        "offer_rows_without_identifier": 0,
        "offer_rows_without_valid_date": 0,
    }
    if remove_offer_duplicates:
        data, dedup_diag = deduplicate_offers(
            data,
            duplicate_date_column,
            duplicate_identifier_columns,
            registration_column=duplicate_registration_column,
            value_column=duplicate_value_column,
        )

    numeric_columns = {
        mapping.valor,
        reference_area_column,
    }
    for column in (
        mapping.latitude,
        mapping.longitude,
        mapping.area_construida,
        mapping.area_privativa,
        mapping.siat_area_total_lote,
        mapping.testada,
        mapping.ano_construcao,
    ):
        if column:
            numeric_columns.add(column)

    for column in numeric_columns:
        data[column] = to_numeric(data[column])

    data, excluded_data, flagged_data, prefilter_diag = (
        _safe_market_prefilter(
            data=data,
            mapping=mapping,
            reference_area_column=reference_area_column,
            value_kind=value_kind,
            selected_purpose=selected_purpose,
            floor_purpose=floor_purpose,
        )
    )

    itbi = data.loc[
        data["_tipo_norm"].eq(TIPO_ITBI),
        "_valor_unitario_original",
    ]
    offers = data.loc[
        data["_tipo_norm"].eq(TIPO_OFERTA),
        "_valor_unitario_original",
    ]
    discount, discount_diag = estimate_offer_discount(
        itbi,
        offers,
        discount_cap,
    )

    data["_fator_ajuste"] = np.where(
        data["_tipo_norm"].eq(TIPO_OFERTA),
        1.0 - discount,
        1.0,
    )
    data["_valor_unitario_ajustado"] = (
        data["_valor_unitario_original"] * data["_fator_ajuste"]
    )

    diagnostics = {
        **discount_diag,
        **dedup_diag,
        **prefilter_diag,
        "purpose": selected_purpose,
        "floor_purpose": floor_purpose or selected_purpose,
        "n_filtered": int(len(data)),
        "n_itbi": int(data["_tipo_norm"].eq(TIPO_ITBI).sum()),
        "n_offer": int(data["_tipo_norm"].eq(TIPO_OFERTA).sum()),
        "reference_area_column": reference_area_column,
        "value_kind": value_kind,
        "conflict_policy": "sem conflitos disponíveis",
        "conflict_fallback_used": False,
        "conflict_minimum_required": (
            int(minimum_without_conflict)
            if minimum_without_conflict is not None
            else 0
        ),
        "conflict_free_prepared_count": int(len(data)),
        "conflict_rows_available": 0,
        "conflict_rows_included": 0,
        "conflict_clean_error": "",
    }
    return PreparationResult(
        data=data,
        discount=discount,
        diagnostics=diagnostics,
        excluded_data=excluded_data,
        flagged_data=flagged_data,
    )

def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        mad = float(np.median(np.abs(values - center)))
        scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(abs(center), 1.0)
    return center, scale


def _local_xy_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = 6371.0088
    lat0 = np.radians(target_lat)
    x = radius * np.cos(lat0) * np.radians(longitudes - target_lon)
    y = radius * np.radians(latitudes - target_lat)
    return x, y


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * weights.sum()
    return float(values[np.searchsorted(cumulative, cutoff, side="left")])


def _cap_and_normalize_weights(
    raw_weights: np.ndarray,
    requested_cap: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw_weights, dtype=float)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)
    n = len(raw)
    if n == 0:
        raise ValueError("Não existem pesos para normalizar.")
    if raw.sum() <= 0:
        raw = np.ones(n, dtype=float)

    # A soma dos pesos precisa alcançar 1. Com poucos vizinhos, o teto mínimo
    # matematicamente viável é 1/n.
    effective_cap = max(float(requested_cap), 1.0 / n)
    effective_cap = min(effective_cap, 1.0)

    weights = np.zeros(n, dtype=float)
    free = np.ones(n, dtype=bool)
    remaining = 1.0

    for _ in range(n + 2):
        if not free.any():
            break
        free_raw = raw[free]
        if free_raw.sum() <= 0:
            proposed = np.full(free.sum(), remaining / free.sum())
        else:
            proposed = remaining * free_raw / free_raw.sum()

        over = proposed > effective_cap + 1e-15
        free_indices = np.flatnonzero(free)
        if not over.any():
            weights[free_indices] = proposed
            remaining = 0.0
            break

        capped_indices = free_indices[over]
        weights[capped_indices] = effective_cap
        free[capped_indices] = False
        remaining = 1.0 - weights[~free].sum()

    if free.any() and remaining > 1e-12:
        free_indices = np.flatnonzero(free)
        weights[free_indices] += remaining / len(free_indices)

    weights = np.clip(weights, 0.0, effective_cap)
    weights /= weights.sum()
    return weights, effective_cap


def _robust_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    mad_threshold: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    median = _weighted_quantile(values, weights, 0.50)
    abs_dev = np.abs(values - median)
    mad = _weighted_quantile(abs_dev, weights, 0.50)
    robust_sigma = 1.4826 * mad

    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
        clipped = values.copy()
        lower = float(np.min(values))
        upper = float(np.max(values))
    else:
        lower = float(median - mad_threshold * robust_sigma)
        upper = float(median + mad_threshold * robust_sigma)
        clipped = np.clip(values, lower, upper)

    estimate = float(np.sum(weights * clipped))
    diagnostics = {
        "robust_method": "média ponderada winsorizada por MAD",
        "robust_median": float(median),
        "robust_mad": float(mad),
        "robust_sigma": float(robust_sigma),
        "robust_lower_bound": lower,
        "robust_upper_bound": upper,
        "robust_adjusted_count": int(np.sum(~np.isclose(values, clipped))),
        "robust_adjusted_weight": float(
            np.sum(weights[~np.isclose(values, clipped)])
        ),
    }
    return estimate, clipped, diagnostics


def _resolve_features(
    mapping: ColumnMapping,
    target: dict[str, float | None],
    territorial: bool,
) -> tuple[list[tuple[str, str]], bool]:
    built_valid = any(
        target.get(key) is not None
        and np.isfinite(float(target[key]))
        and float(target[key]) > 0
        for key in ("area_construida", "area_privativa")
    )
    lot_value = target.get("siat_area_total_lote")
    lot_valid = (
        lot_value is not None
        and np.isfinite(float(lot_value))
        and float(lot_value) > 0
    )
    effective_territorial = territorial or (lot_valid and not built_valid)

    pairs = [
        ("area_construida", mapping.area_construida),
        ("area_privativa", mapping.area_privativa),
    ]
    if effective_territorial:
        pairs.extend(
            [
                ("siat_area_total_lote", mapping.siat_area_total_lote),
                ("testada", mapping.testada),
            ]
        )
    else:
        # Em imóveis prediais térreos, o lote é uma característica física
        # adicional. O denominador do valor unitário continua sendo a área
        # construída/privativa selecionada pela interface.
        if lot_valid:
            pairs.append(
                ("siat_area_total_lote", mapping.siat_area_total_lote)
            )
        construction_year = target.get("ano_construcao")
    
        # O ano é uma característica opcional.
        # Só entra no KNN quando existe na base e foi informado para o avaliando.
        if construction_year is not None:
            if not mapping.ano_construcao:
                raise ValueError(
                    "Foi informado o ano da construção para o avaliando, "
                    "mas a planilha não possui uma coluna de ano da construção."
                )
    
            if not is_valid_construction_year(construction_year):
                raise ValueError(
                    "Quando informado, o ano da construção deve ser um inteiro entre "
                    f"{MIN_CONSTRUCTION_YEAR} e {MAX_CONSTRUCTION_YEAR}."
                )
    
            pairs.append(
                ("ano_construcao", mapping.ano_construcao)
            )

    active: list[tuple[str, str]] = []
    for key, column in pairs:
        value = target.get(key)
        if (
            column
            and value is not None
            and np.isfinite(float(value))
            and float(value) > 0
        ):
            active.append((key, column))

    if effective_territorial and not any(
        key == "siat_area_total_lote" for key, _ in active
    ):
        raise ValueError(
            "Para imóvel territorial, informe e mapeie a área total do lote."
        )
    if effective_territorial and not any(
        key == "testada" for key, _ in active
    ):
        raise ValueError(
            "Para imóvel territorial, informe e mapeie a TESTADA."
        )
    if not active:
        raise ValueError(
            "Informe ao menos uma característica de área válida para o avaliando."
        )
    return active, effective_territorial


def _valid_coordinate_pair(latitude: Any, longitude: Any) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(lat)
        and np.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def _valid_candidates(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    active_features: list[tuple[str, str]],
    use_location: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    columns = [
        "_valor_unitario_ajustado",
    ] + [column for _, column in active_features]

    missing = [column for column in columns if column not in result.columns]
    if missing:
        raise ValueError(
            "As seguintes características mapeadas não existem na amostra: "
            + ", ".join(dict.fromkeys(missing))
        )

    valid = np.ones(len(result), dtype=bool)
    for column in dict.fromkeys(columns):
        result[column] = to_numeric(result[column])
        valid &= np.isfinite(result[column])

    valid &= result["_valor_unitario_ajustado"].gt(0).to_numpy()
    for key, column in active_features:
        valid &= result[column].gt(0).to_numpy()
        if key == "ano_construcao":
            valid &= valid_construction_year_mask(result[column]).to_numpy()

    physically_valid = result.loc[valid].copy()
    invalid_location = physically_valid.iloc[0:0].copy()
    if not use_location:
        return physically_valid, invalid_location

    if (
        not mapping.latitude
        or not mapping.longitude
        or mapping.latitude not in physically_valid.columns
        or mapping.longitude not in physically_valid.columns
    ):
        raise ValueError(
            "A localização foi ativada, mas as colunas de latitude e longitude "
            "não estão disponíveis."
        )

    physically_valid[mapping.latitude] = to_numeric(
        physically_valid[mapping.latitude]
    )
    physically_valid[mapping.longitude] = to_numeric(
        physically_valid[mapping.longitude]
    )
    location_valid = (
        physically_valid[mapping.latitude].between(-90, 90)
        & physically_valid[mapping.longitude].between(-180, 180)
    )
    invalid_location = physically_valid.loc[~location_valid].copy()
    return physically_valid.loc[location_valid].copy(), invalid_location


def _distance_profile(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    active_features: list[tuple[str, str]],
    target: dict[str, float | None],
    similarity_weight: float,
    use_location: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_features = len(active_features)
    attr_sq = np.zeros(len(data), dtype=float)

    for key, column in active_features:
        values = data[column].to_numpy(dtype=float)
        center, scale = _robust_center_scale(values)
        delta = (values - float(target[key])) / scale
        attr_sq += np.square(delta) / n_features

    attribute_distance = np.sqrt(attr_sq)
    if not use_location:
        return (
            attribute_distance,
            attribute_distance,
            np.full(len(data), np.nan, dtype=float),
            np.nan,
        )

    lat_target = float(target["latitude"])
    lon_target = float(target["longitude"])
    latitudes = data[mapping.latitude].to_numpy(dtype=float)
    longitudes = data[mapping.longitude].to_numpy(dtype=float)
    x_km, y_km = _local_xy_km(latitudes, longitudes, lat_target, lon_target)
    radial_km = np.sqrt(np.square(x_km) + np.square(y_km))

    positive = radial_km[np.isfinite(radial_km) & (radial_km > 0)]
    geo_scale = float(np.median(positive)) if positive.size else 1.0
    geo_scale = float(np.clip(geo_scale, 0.25, 20.0))

    location_weight = 1.0 - similarity_weight
    composite_sq = (
        similarity_weight * attr_sq
        + location_weight
        * (np.square(x_km / geo_scale) + np.square(y_km / geo_scale))
    )
    return np.sqrt(composite_sq), attribute_distance, radial_km, geo_scale


def _extrapolation_diagnostics(
    data: pd.DataFrame,
    neighbors: pd.DataFrame,
    active_features: list[tuple[str, str]],
    target: dict[str, float | None],
    geo_distances: np.ndarray,
    effective_neighbors: float,
    required_effective_neighbors: float,
    robust_adjusted_weight: float,
) -> dict[str, Any]:
    points = 0
    reasons: list[str] = []
    coverage: dict[str, dict[str, float | bool]] = {}

    for key, column in active_features:
        target_value = float(target[key])
        candidates = data[column].to_numpy(dtype=float)
        selected = neighbors[column].to_numpy(dtype=float)
        nearest_absolute_gap = float(
            np.min(np.abs(candidates - target_value))
        )
        nearest_gap = float(nearest_absolute_gap / target_value)
        candidate_min = float(np.min(candidates))
        candidate_max = float(np.max(candidates))
        outside = target_value < candidate_min or target_value > candidate_max

        coverage[column] = {
            "target": target_value,
            "candidate_min": candidate_min,
            "candidate_max": candidate_max,
            "selected_min": float(np.min(selected)),
            "selected_max": float(np.max(selected)),
            "nearest_absolute_difference": nearest_absolute_gap,
            "nearest_relative_difference": nearest_gap,
            "outside_candidate_range": bool(outside),
        }

        if outside:
            points += 3
            reasons.append(
                f"{column}: valor do avaliando fora da faixa observada na amostra"
            )
        elif key == "ano_construcao" and nearest_absolute_gap > 30:
            points += 3
            reasons.append(
                f"{column}: construção comparável mais próxima difere "
                "mais de 30 anos"
            )
        elif key == "ano_construcao" and nearest_absolute_gap > 20:
            points += 2
            reasons.append(
                f"{column}: construção comparável mais próxima difere "
                "mais de 20 anos"
            )
        elif key == "ano_construcao" and nearest_absolute_gap > 10:
            points += 1
            reasons.append(
                f"{column}: construção comparável mais próxima difere "
                "mais de 10 anos"
            )
        elif nearest_gap > 0.50:
            points += 3
            reasons.append(
                f"{column}: comparável mais próximo difere mais de 50%"
            )
        elif nearest_gap > 0.30:
            points += 2
            reasons.append(
                f"{column}: comparável mais próximo difere mais de 30%"
            )
        elif nearest_gap > 0.15:
            points += 1
            reasons.append(
                f"{column}: comparável mais próximo difere mais de 15%"
            )

    median_geo = float(np.median(geo_distances))
    max_geo = float(np.max(geo_distances))
    if median_geo > 10:
        points += 2
        reasons.append("distância geográfica mediana dos vizinhos superior a 10 km")
    elif median_geo > 5:
        points += 1
        reasons.append("distância geográfica mediana dos vizinhos superior a 5 km")

    if effective_neighbors < required_effective_neighbors:
        points += 2
        reasons.append("número efetivo de vizinhos abaixo do objetivo")

    if robust_adjusted_weight > 0.30:
        points += 1
        reasons.append("mais de 30% do peso foi afetado pelo tratamento robusto")

    if points >= 5:
        level = "alto"
    elif points >= 2:
        level = "moderado"
    else:
        level = "baixo"

    confidence = int(np.clip(100 - 12 * points, 20, 100))
    return {
        "risk_level": level,
        "risk_points": int(points),
        "confidence_score": confidence,
        "risk_reasons": reasons,
        "feature_coverage": coverage,
        "selected_geo_median_km": median_geo,
        "selected_geo_max_km": max_geo,
    }




LOCAL_FILTER_MIN_REFERENCE = 8
LOCAL_FILTER_STRICT_AREA_MIN_RATIO = 0.50
LOCAL_FILTER_STRICT_AREA_MAX_RATIO = 2.00
LOCAL_FILTER_RELAXED_AREA_MIN_RATIO = 1.0 / 3.0
LOCAL_FILTER_RELAXED_AREA_MAX_RATIO = 3.00
LOCAL_FILTER_STRICT_GEO_KM = 5.0
LOCAL_FILTER_RELAXED_GEO_KM = 10.0

LOCAL_FILTER_GLOBAL_PARAMETERS: dict[str, float | int | str] = {
    "profile": "adaptativo_seguro_global",
    "median_fraction_floor": 0.45,
    "lower_modified_z": 3.00,
    "iqr_multiplier": 2.00,
    "gap_min_ratio": 1.70,
    "gap_max_lower_share": 0.30,
    "max_exclusion_share": 0.30,
    "compact_iqr_ratio": 2.20,
    "proximity_weight_power": 0.75,
    "max_reference_count": 60,
}

LOCAL_FILTER_PURPOSE_PARAMETERS: dict[
    str,
    dict[str, float | int | str],
] = {
    "sala comercial": {
        "profile": "adaptativo_seguro_sala_comercial",
        "median_fraction_floor": 0.55,
        "gap_min_ratio": 1.60,
        "compact_iqr_ratio": 2.00,
    },
    "loja": {
        "profile": "adaptativo_seguro_loja",
        "median_fraction_floor": 0.50,
        "gap_min_ratio": 1.65,
    },
    "loja em galeria": {
        "profile": "adaptativo_seguro_loja",
        "median_fraction_floor": 0.50,
        "gap_min_ratio": 1.65,
    },
    "loja em shopping": {
        "profile": "adaptativo_seguro_loja",
        "median_fraction_floor": 0.50,
        "gap_min_ratio": 1.65,
    },
    "imovel comercial": {
        "profile": "adaptativo_seguro_imovel_comercial",
        "median_fraction_floor": 0.50,
        "gap_min_ratio": 1.65,
    },
}


def local_filter_parameters_for_purpose(
    purpose: Any,
) -> dict[str, float | int | str]:
    parameters = dict(LOCAL_FILTER_GLOBAL_PARAMETERS)
    parameters.update(
        LOCAL_FILTER_PURPOSE_PARAMETERS.get(
            normalize_text(purpose),
            {},
        )
    )
    return parameters


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return np.nan

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    position = int(np.searchsorted(cumulative, threshold, side="left"))
    position = min(position, values.size - 1)
    return float(values[position])


def _detect_lower_market_gap(
    values: np.ndarray,
    weights: np.ndarray,
    distances: np.ndarray,
    min_upper_count: int,
    min_ratio: float,
    max_lower_share: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "detected": False,
        "ratio": np.nan,
        "lower_value": np.nan,
        "upper_value": np.nan,
        "cutoff": np.nan,
        "lower_count": 0,
        "upper_count": 0,
        "lower_share": 0.0,
        "lower_weight_share": 0.0,
        "affinity_ratio": np.nan,
    }

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    distances = np.asarray(distances, dtype=float)

    valid = (
        np.isfinite(values)
        & (values > 0)
        & np.isfinite(weights)
        & (weights > 0)
        & np.isfinite(distances)
    )
    values = values[valid]
    weights = weights[valid]
    distances = distances[valid]

    if values.size < max(LOCAL_FILTER_MIN_REFERENCE, min_upper_count + 1):
        return result

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_distances = distances[order]

    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0:
        return result

    best_score = -np.inf
    best_result: dict[str, Any] | None = None

    for boundary in range(sorted_values.size - 1):
        lower_count = boundary + 1
        upper_count = sorted_values.size - lower_count
        lower_share = lower_count / sorted_values.size
        lower_weight_share = float(
            np.sum(sorted_weights[:lower_count]) / total_weight
        )

        if upper_count < min_upper_count:
            continue
        if lower_share > max_lower_share:
            continue
        if lower_weight_share > max_lower_share:
            continue

        lower_value = float(sorted_values[boundary])
        upper_value = float(sorted_values[boundary + 1])
        if lower_value <= 0:
            continue

        ratio = upper_value / lower_value
        if not np.isfinite(ratio) or ratio < min_ratio:
            continue

        lower_distance = float(
            np.median(sorted_distances[:lower_count])
        )
        upper_distance = float(
            np.median(sorted_distances[lower_count:])
        )
        affinity_ratio = upper_distance / max(lower_distance, 1e-9)

        affinity_ok = lower_count <= 2 or affinity_ratio <= 1.35
        if not affinity_ok:
            continue

        score = (
            np.log(ratio)
            * (1.0 - lower_weight_share)
            / max(affinity_ratio, 0.50)
        )
        if score <= best_score:
            continue

        best_score = score
        best_result = {
            "detected": True,
            "ratio": float(ratio),
            "lower_value": lower_value,
            "upper_value": upper_value,
            "cutoff": float(np.sqrt(lower_value * upper_value)),
            "lower_count": int(lower_count),
            "upper_count": int(upper_count),
            "lower_share": float(lower_share),
            "lower_weight_share": float(lower_weight_share),
            "affinity_ratio": float(affinity_ratio),
        }

    return best_result or result


def _local_lower_tail_filter(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    active_features: list[tuple[str, str]],
    target: dict[str, float | None],
    similarity_weight: float,
    use_location: bool,
    min_k: int,
    max_k: int,
    purpose: Any = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Filtro local inferior adaptativo.

    A referência é formada por imóveis física e espacialmente compatíveis.
    Os valores unitários já estão ajustados pelo fator de oferta. O limite
    combina estatística robusta, fração da mediana local ponderada e uma
    ruptura entre grupos de valor, com travas contra exclusão excessiva.
    """
    parameters = local_filter_parameters_for_purpose(purpose)
    diagnostics: dict[str, Any] = {
        "local_low_filter_enabled": True,
        "local_low_filter_method": (
            "mediana local ponderada + MAD/IQR log + ruptura de mercado"
        ),
        "local_low_filter_profile": parameters["profile"],
        "local_low_filter_applied": False,
        "local_low_filter_reference_mode": "não aplicado",
        "local_low_filter_reference_count": 0,
        "local_low_filter_excluded": 0,
        "local_low_filter_exclusion_share": 0.0,
        "local_low_filter_lower_bound": np.nan,
        "local_low_filter_median_vu": np.nan,
        "local_low_filter_weighted_median_vu": np.nan,
        "local_low_filter_median_fraction": float(
            parameters["median_fraction_floor"]
        ),
        "local_low_filter_fraction_bound": np.nan,
        "local_low_filter_mad_bound": np.nan,
        "local_low_filter_iqr_bound": np.nan,
        "local_low_filter_robust_bound": np.nan,
        "local_low_filter_iqr_ratio": np.nan,
        "local_low_filter_compact_distribution": False,
        "local_low_filter_gap_detected": False,
        "local_low_filter_gap_ratio": np.nan,
        "local_low_filter_gap_lower_value": np.nan,
        "local_low_filter_gap_upper_value": np.nan,
        "local_low_filter_gap_cutoff": np.nan,
        "local_low_filter_gap_lower_share": 0.0,
        "local_low_filter_gap_affinity_ratio": np.nan,
        "local_low_filter_cutoff_source": "não aplicado",
        "local_low_filter_bimodal_alert": False,
        "local_low_filter_cancelled_reason": "",
    }

    empty_excluded = data.iloc[0:0].copy()
    if len(data) < max(int(min_k) + 1, LOCAL_FILTER_MIN_REFERENCE):
        diagnostics["local_low_filter_reference_mode"] = (
            "não aplicado: poucos candidatos"
        )
        return data, empty_excluded, diagnostics

    distances, _, geo_distances, _ = _distance_profile(
        data,
        mapping,
        active_features,
        target,
        similarity_weight,
        use_location,
    )

    strict_mask = np.ones(len(data), dtype=bool)
    relaxed_mask = np.ones(len(data), dtype=bool)

    for target_key, column in active_features:
        # Os limites abaixo são razões de área; o ano já participa da
        # distância física padronizada, mas não desse filtro de área.
        if target_key == "ano_construcao":
            continue
        target_value = target.get(target_key)
        if (
            target_value is None
            or not np.isfinite(float(target_value))
            or float(target_value) <= 0
        ):
            continue

        values = data[column].to_numpy(dtype=float)
        ratio = values / float(target_value)
        strict_mask &= (
            np.isfinite(ratio)
            & (ratio >= LOCAL_FILTER_STRICT_AREA_MIN_RATIO)
            & (ratio <= LOCAL_FILTER_STRICT_AREA_MAX_RATIO)
        )
        relaxed_mask &= (
            np.isfinite(ratio)
            & (ratio >= LOCAL_FILTER_RELAXED_AREA_MIN_RATIO)
            & (ratio <= LOCAL_FILTER_RELAXED_AREA_MAX_RATIO)
        )

    if use_location:
        strict_mask &= geo_distances <= LOCAL_FILTER_STRICT_GEO_KM
        relaxed_mask &= geo_distances <= LOCAL_FILTER_RELAXED_GEO_KM

    if int(strict_mask.sum()) >= LOCAL_FILTER_MIN_REFERENCE:
        reference_positions = np.flatnonzero(strict_mask)
        reference_mode = (
            "área entre 50% e 200% e distância até 5 km"
            if use_location
            else "características físicas entre 50% e 200%"
        )
    elif int(relaxed_mask.sum()) >= LOCAL_FILTER_MIN_REFERENCE:
        reference_positions = np.flatnonzero(relaxed_mask)
        reference_mode = (
            "área entre 33% e 300% e distância até 10 km"
            if use_location
            else "características físicas entre 33% e 300%"
        )
    else:
        cohort_size = min(
            len(data),
            max(
                LOCAL_FILTER_MIN_REFERENCE,
                min(
                    int(parameters["max_reference_count"]),
                    max(int(max_k) * 2, 20),
                ),
            ),
        )
        reference_positions = np.argsort(
            distances,
            kind="mergesort",
        )[:cohort_size]
        reference_mode = (
            "candidatos mais próximos por distância composta"
            if use_location
            else "candidatos mais próximos por distância física"
        )

    reference = data.iloc[reference_positions].copy()
    reference_values_all = reference[
        "_valor_unitario_ajustado"
    ].to_numpy(dtype=float)
    reference_distances_all = distances[reference_positions]

    valid_reference = (
        np.isfinite(reference_values_all)
        & (reference_values_all > 0)
        & np.isfinite(reference_distances_all)
    )
    reference_values = reference_values_all[valid_reference]
    reference_distances = reference_distances_all[valid_reference]

    diagnostics["local_low_filter_reference_mode"] = reference_mode
    diagnostics["local_low_filter_reference_count"] = int(
        reference_values.size
    )

    if reference_values.size < LOCAL_FILTER_MIN_REFERENCE:
        return data, empty_excluded, diagnostics

    proximity_power = float(parameters["proximity_weight_power"])
    raw_weights = 1.0 / np.power(
        reference_distances + 0.25,
        proximity_power,
    )
    reference_weights, _ = _cap_and_normalize_weights(
        raw_weights,
        0.25,
    )

    log_values = np.log(reference_values)
    median_log = _weighted_quantile(
        log_values,
        reference_weights,
        0.50,
    )
    median_vu = float(np.exp(median_log))
    abs_deviation = np.abs(log_values - median_log)
    mad_log = _weighted_quantile(
        abs_deviation,
        reference_weights,
        0.50,
    )
    q1 = _weighted_quantile(
        log_values,
        reference_weights,
        0.25,
    )
    q3 = _weighted_quantile(
        log_values,
        reference_weights,
        0.75,
    )
    iqr = float(q3 - q1)
    iqr_ratio = float(np.exp(iqr)) if np.isfinite(iqr) else np.nan
    compact_distribution = bool(
        np.isfinite(iqr_ratio)
        and iqr_ratio <= float(parameters["compact_iqr_ratio"])
    )

    fraction_bound = (
        median_vu * float(parameters["median_fraction_floor"])
    )
    mad_bound = 0.0
    if np.isfinite(mad_log) and mad_log > 1e-12:
        lower_log_mad = (
            median_log
            - float(parameters["lower_modified_z"])
            * mad_log
            / 0.6745
        )
        mad_bound = float(np.exp(lower_log_mad))

    iqr_bound = 0.0
    if np.isfinite(iqr) and iqr > 1e-12:
        iqr_bound = float(
            np.exp(
                q1
                - float(parameters["iqr_multiplier"])
                * iqr
            )
        )

    robust_bound = float(max(mad_bound, iqr_bound, 0.0))
    gap = _detect_lower_market_gap(
        reference_values,
        reference_weights,
        reference_distances,
        min_upper_count=max(int(min_k), LOCAL_FILTER_MIN_REFERENCE),
        min_ratio=float(parameters["gap_min_ratio"]),
        max_lower_share=float(parameters["gap_max_lower_share"]),
    )

    candidate_bounds: list[tuple[str, float]] = []
    if robust_bound > 0:
        candidate_bounds.append(("robusto MAD/IQR", robust_bound))
    if compact_distribution or bool(gap["detected"]):
        candidate_bounds.append(
            ("fração da mediana local", fraction_bound)
        )
    if bool(gap["detected"]):
        candidate_bounds.append(
            ("ruptura entre grupos de valor", float(gap["cutoff"]))
        )

    diagnostics.update(
        {
            "local_low_filter_median_vu": median_vu,
            "local_low_filter_weighted_median_vu": median_vu,
            "local_low_filter_fraction_bound": fraction_bound,
            "local_low_filter_mad_bound": (
                mad_bound if mad_bound > 0 else np.nan
            ),
            "local_low_filter_iqr_bound": (
                iqr_bound if iqr_bound > 0 else np.nan
            ),
            "local_low_filter_robust_bound": (
                robust_bound if robust_bound > 0 else np.nan
            ),
            "local_low_filter_iqr_ratio": iqr_ratio,
            "local_low_filter_compact_distribution": (
                compact_distribution
            ),
            "local_low_filter_gap_detected": bool(gap["detected"]),
            "local_low_filter_gap_ratio": gap["ratio"],
            "local_low_filter_gap_lower_value": gap["lower_value"],
            "local_low_filter_gap_upper_value": gap["upper_value"],
            "local_low_filter_gap_cutoff": gap["cutoff"],
            "local_low_filter_gap_lower_share": gap["lower_share"],
            "local_low_filter_gap_affinity_ratio": (
                gap["affinity_ratio"]
            ),
            "local_low_filter_bimodal_alert": bool(gap["detected"]),
        }
    )

    if not candidate_bounds:
        diagnostics["local_low_filter_cutoff_source"] = (
            "nenhum indício seguro de cauda inferior incompatível"
        )
        return data, empty_excluded, diagnostics

    cutoff_source, lower_bound = max(
        candidate_bounds,
        key=lambda item: item[1],
    )
    lower_bound = float(lower_bound)

    candidate_values = data[
        "_valor_unitario_ajustado"
    ].to_numpy(dtype=float)
    low_mask = (
        np.isfinite(candidate_values)
        & (candidate_values > 0)
        & (candidate_values < lower_bound)
    )

    max_exclusion_share = float(parameters["max_exclusion_share"])
    max_exclusion_count = int(
        np.floor(len(data) * max_exclusion_share)
    )
    low_count = int(low_mask.sum())

    if low_count > max_exclusion_count:
        robust_mask = (
            np.isfinite(candidate_values)
            & (candidate_values > 0)
            & (candidate_values < robust_bound)
        ) if robust_bound > 0 else np.zeros(len(data), dtype=bool)

        robust_count = int(robust_mask.sum())
        if (
            robust_bound > 0
            and robust_count <= max_exclusion_count
            and int((~robust_mask).sum()) >= max(int(min_k), 2)
        ):
            low_mask = robust_mask
            low_count = robust_count
            lower_bound = robust_bound
            cutoff_source = "robusto MAD/IQR após trava de 30%"
        else:
            diagnostics.update(
                {
                    "local_low_filter_lower_bound": lower_bound,
                    "local_low_filter_cutoff_source": cutoff_source,
                    "local_low_filter_cancelled_reason": (
                        "exclusão potencial superior ao limite de 30% "
                        "da amostra; possível submercado legítimo"
                    ),
                }
            )
            return data, empty_excluded, diagnostics

    if int((~low_mask).sum()) < max(int(min_k), 2):
        diagnostics.update(
            {
                "local_low_filter_lower_bound": lower_bound,
                "local_low_filter_cutoff_source": cutoff_source,
                "local_low_filter_cancelled_reason": (
                    "exclusão cancelada para preservar o K inicial"
                ),
            }
        )
        return data, empty_excluded, diagnostics

    diagnostics["local_low_filter_lower_bound"] = lower_bound
    diagnostics["local_low_filter_cutoff_source"] = cutoff_source

    excluded = data.loc[low_mask].copy()
    if excluded.empty:
        return data, excluded, diagnostics

    excluded["_etapa_controle"] = (
        "filtro local adaptativo de cauda inferior"
    )
    excluded["_motivo_exclusao"] = (
        "Valor unitário ajustado incompatível com o submercado local "
        "do avaliando"
    )
    excluded["_motivo_alerta"] = ""
    excluded["_limite_inferior_vu_prefiltro"] = lower_bound
    excluded["_limite_superior_vu_prefiltro"] = np.nan
    excluded["_escore_robusto_prefiltro"] = np.nan
    excluded["_mediana_local_ponderada"] = median_vu
    excluded["_origem_limite_local"] = cutoff_source
    excluded["_razao_ruptura_local"] = gap["ratio"]

    cleaned = data.loc[~low_mask].copy()
    diagnostics.update(
        {
            "local_low_filter_applied": True,
            "local_low_filter_excluded": int(len(excluded)),
            "local_low_filter_exclusion_share": float(
                len(excluded) / len(data)
            ),
        }
    )
    return cleaned, _ordered_control_columns(excluded), diagnostics

def estimate_knn(
    preparation: PreparationResult,
    mapping: ColumnMapping,
    target: dict[str, float | None],
    reference_area_column: str,
    min_k: int = 12,
    max_k: int = 25,
    min_effective_neighbors: float = 11.0,
    similarity_weight: float = 0.45,
    distance_power: float = 0.35,
    max_individual_weight: float = 0.25,
    robust_mad_threshold: float = 1.25,
    territorial: bool = False,
    exclude_indices: Iterable[Any] = (),
) -> EstimateResult:
    if not 0.0 < similarity_weight < 1.0:
        raise ValueError("O peso físico deve estar entre 0 e 1.")
    if min_k < 2:
        raise ValueError("O k mínimo deve ser ao menos 2.")
    if max_k < min_k:
        raise ValueError("O k máximo não pode ser menor que o k mínimo.")
    if min_effective_neighbors < 1:
        raise ValueError("O mínimo de vizinhos efetivos deve ser positivo.")
    if distance_power <= 0:
        raise ValueError("A potência da distância deve ser positiva.")
    if not 0 < max_individual_weight <= 1:
        raise ValueError("O limite de peso deve estar entre 0 e 1.")
    if robust_mad_threshold <= 0:
        raise ValueError("O limiar robusto deve ser positivo.")

    active_features, effective_territorial = _resolve_features(
        mapping, target, territorial
    )

    location_columns_available = bool(
        mapping.latitude
        and mapping.longitude
        and mapping.latitude in preparation.data.columns
        and mapping.longitude in preparation.data.columns
    )
    target_location_valid = _valid_coordinate_pair(
        target.get("latitude"),
        target.get("longitude"),
    )
    location_used = bool(
        location_columns_available and target_location_valid
    )
    effective_similarity_weight = (
        float(similarity_weight) if location_used else 1.0
    )
    
    uses_construction_year = any(
        key == "ano_construcao"
        for key, _ in active_features
    )
    
    invalid_construction_years = preparation.data.iloc[0:0].copy()
    construction_year_excluded_count = 0
    
    if uses_construction_year:
        year_mask = valid_construction_year_mask(
            preparation.data[mapping.ano_construcao]
        )
    
        invalid_construction_years = preparation.data.loc[
            ~year_mask
        ].copy()
    
        construction_year_excluded_count = int(
            len(invalid_construction_years)
        )
    
        if construction_year_excluded_count:
            invalid_construction_years["_etapa_controle"] = (
                "validação do ano da construção"
            )
            invalid_construction_years["_motivo_exclusao"] = (
                "Ano da construção ausente, fracionário, futuro ou fora "
                f"da faixa {MIN_CONSTRUCTION_YEAR}–{MAX_CONSTRUCTION_YEAR}"
            )
            invalid_construction_years["_motivo_alerta"] = ""
            invalid_construction_years = _ordered_control_columns(
                invalid_construction_years
            )
        
    data, invalid_locations = _valid_candidates(
        preparation.data,
        mapping,
        active_features,
        use_location=location_used,
    )
    candidates_after_physical_validation = int(
        len(data) + len(invalid_locations)
    )
    location_invalid_excluded_count = int(len(invalid_locations))
    if location_invalid_excluded_count:
        invalid_locations["_etapa_controle"] = "validação da localização"
        invalid_locations["_motivo_exclusao"] = (
            "Latitude ou longitude ausente, não numérica ou fora dos limites "
            "do EPSG:4326"
        )
        invalid_locations["_motivo_alerta"] = ""
        invalid_locations = _ordered_control_columns(invalid_locations)

    exclusions = set(exclude_indices)
    if exclusions:
        data = data.loc[~data.index.isin(exclusions)].copy()
    if len(data) < 2:
        raise ValueError("Não existem candidatos suficientes para a estimativa.")

    data, local_excluded_data, local_filter_diag = (
        _local_lower_tail_filter(
            data=data,
            mapping=mapping,
            active_features=active_features,
            target=target,
            similarity_weight=effective_similarity_weight,
            use_location=location_used,
            min_k=min_k,
            max_k=max_k,
            purpose=preparation.diagnostics.get("purpose", ""),
        )
    )
    if len(data) < 2:
        raise ValueError(
            "Não existem candidatos suficientes após o filtro local."
        )

    local_excluded_data = _ordered_control_columns(
        pd.concat(
            [
                invalid_construction_years,
                invalid_locations,
                local_excluded_data,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    distances, attr_dist, geo_dist, geo_scale = _distance_profile(
        data,
        mapping,
        active_features,
        target,
        effective_similarity_weight,
        location_used,
    )
    order = np.argsort(distances, kind="mergesort")
    max_available = min(int(max_k), len(data))
    min_available = min(max(int(min_k), 2), max_available)

    selected_k = max_available
    selected_weights: np.ndarray | None = None
    selected_effective = 0.0
    selected_cap = max_individual_weight

    epsilon = 1e-9
    for candidate_k in range(min_available, max_available + 1):
        idx = order[:candidate_k]
        
        # Peso original do KNN
        raw = 1.0 / np.power(
            distances[idx] + epsilon,
            distance_power
        )
        
        building_factor = np.ones(candidate_k, dtype=float)
        if location_used:
            # Distância geográfica em metros e bônus para provável mesmo
            # edifício. Sem localização, nenhum bônus espacial é aplicado.
            geo_m = geo_dist[idx] * 1000.0
            same_building = geo_m <= 30.0
            building_factor[same_building] = 2.0
            transition = (geo_m > 30.0) & (geo_m <= 50.0)
            building_factor[transition] = (
                1.0
                + (50.0 - geo_m[transition])
                / (50.0 - 30.0)
            )
        
        # Aplica o bônus antes da normalização
        raw = raw * building_factor
        
        # Mantém o limite individual existente
        weights, effective_cap = _cap_and_normalize_weights(
            raw,
            max_individual_weight
        )
        effective_n = float(1.0 / np.sum(np.square(weights)))
        selected_k = candidate_k
        selected_weights = weights
        selected_effective = effective_n
        selected_cap = effective_cap
        if effective_n >= min_effective_neighbors:
            break

    if selected_weights is None:
        raise RuntimeError("Falha na seleção adaptativa de vizinhos.")

    selected_positions = order[:selected_k]
    neighbors = data.iloc[selected_positions].copy()
    selected_distances = distances[selected_positions]
    selected_attr = attr_dist[selected_positions]
    selected_geo = geo_dist[selected_positions]

    values = neighbors["_valor_unitario_ajustado"].to_numpy(dtype=float)
    estimated_unit, robust_values, robust_diag = _robust_weighted_mean(
        values, selected_weights, robust_mad_threshold
    )

    mapping_to_target = {
        mapping.area_construida: "area_construida",
        mapping.area_privativa: "area_privativa",
        mapping.siat_area_total_lote: "siat_area_total_lote",
    }
    reference_target = target.get(reference_area_column)
    if reference_target is None:
        reference_key = mapping_to_target.get(reference_area_column)
        reference_target = target.get(reference_key) if reference_key else None
    if (
        reference_target is None
        or not np.isfinite(float(reference_target))
        or float(reference_target) <= 0
    ):
        raise ValueError(
            "Informe a área do avaliando correspondente à área de referência."
        )

    estimated_total = estimated_unit * float(reference_target)
    weighted_variance = float(
        np.sum(selected_weights * np.square(robust_values - estimated_unit))
    )
    weighted_std = float(np.sqrt(max(weighted_variance, 0.0)))

    neighbors["_distancia_caracteristicas"] = selected_attr
    neighbors["_distancia_geografica_km"] = selected_geo
    neighbors["_distancia_composta"] = selected_distances
    neighbors["_peso_knn"] = selected_weights
    neighbors["_valor_unitario_robusto"] = robust_values
    neighbors["_ajuste_robusto"] = values - robust_values
    neighbors["_contribuicao_valor_unitario"] = (
        selected_weights * robust_values
    )

    extrapolation = _extrapolation_diagnostics(
        data,
        neighbors,
        active_features,
        target,
        selected_geo,
        selected_effective,
        min_effective_neighbors,
        float(robust_diag["robust_adjusted_weight"]),
    )

    diagnostics = {
        "k_min_requested": int(min_k),
        "k_max_requested": int(max_k),
        "k_used": int(selected_k),
        "adaptive_k_reached_target": bool(
            selected_effective >= min_effective_neighbors
        ),
        "min_effective_neighbors": float(min_effective_neighbors),
        "effective_neighbors": float(selected_effective),
        "max_individual_weight_requested": float(max_individual_weight),
        "max_individual_weight_effective": float(selected_cap),
        "max_weight_observed": float(np.max(selected_weights)),
        "location_used": location_used,
        "location_columns_available": location_columns_available,
        "target_location_valid": target_location_valid,
        "location_crs": "EPSG:4326" if location_used else "",
        "location_distance_unit": "km" if location_used else "",
        "location_distance_method": (
            "aproximação equiretangular local sobre coordenadas EPSG:4326"
            if location_used
            else "localização ignorada; distância exclusivamente física"
        ),
        "location_invalid_excluded": location_invalid_excluded_count,
        "similarity_weight_requested": float(similarity_weight),
        "similarity_weight": effective_similarity_weight,
        "location_weight": (
            float(1.0 - effective_similarity_weight)
        ),
        "distance_power": float(distance_power),
        "n_candidates": int(len(data)),
        "n_candidates_after_physical_validation": (
            candidates_after_physical_validation
        ),
        "construction_year_used": bool(uses_construction_year),
        "construction_year_column": (
            mapping.ano_construcao
            if uses_construction_year
            else ""
        ),
        "construction_year_target": (
            float(target["ano_construcao"])
            if uses_construction_year
            else np.nan
        ),
        "construction_year_invalid_excluded": (
            construction_year_excluded_count
        ),
        **local_filter_diag,
        "reference_target_area": float(reference_target),
        "effective_territorial": bool(effective_territorial),
        **robust_diag,
        **extrapolation,
    }

    return EstimateResult(
        estimated_unit_value=estimated_unit,
        estimated_total_value=estimated_total,
        weighted_std_unit=weighted_std,
        effective_neighbors=selected_effective,
        neighbors=neighbors,
        active_features=[column for _, column in active_features],
        geographic_scale_km=geo_scale,
        diagnostics=diagnostics,
        local_excluded_data=local_excluded_data,
    )


def _safe_group_value(value: Any) -> str:
    normalized = normalize_text(value)
    return "" if normalized in {"", "nan", "none", "<na>"} else normalized


def backtest_knn(
    preparation: PreparationResult,
    mapping: ColumnMapping,
    reference_area_column: str,
    territorial: bool,
    min_k: int,
    max_k: int,
    min_effective_neighbors: float,
    similarity_weight: float,
    distance_power: float,
    max_individual_weight: float,
    robust_mad_threshold: float,
    sample_size: int = 150,
    group_column: str | None = None,
    evaluation_scope: str = "itbi",
    random_state: int = 42,
) -> BacktestResult:
    data = preparation.data.copy()
    if normalize_text(evaluation_scope) in {"itbi", "guias itbi", "guia itbi"}:
        eligible = data.loc[data["_tipo_norm"].eq(TIPO_ITBI)].copy()
        scope_label = "Guias ITBI"
    else:
        eligible = data.copy()
        scope_label = "Todos os dados ajustados"

    required = [
        mapping.latitude,
        mapping.longitude,
        "_valor_unitario_ajustado",
        reference_area_column,
    ]
    if territorial:
        if not mapping.testada:
            raise ValueError(
                "O backtesting territorial exige uma coluna de TESTADA mapeada."
            )
        required.append(mapping.testada)
    else:
        if not mapping.ano_construcao:
            raise ValueError(
                "O backtesting predial exige a coluna siat_ano mapeada."
            )
        required.append(mapping.ano_construcao)
    for column in required:
        eligible[column] = to_numeric(eligible[column])
    valid = np.ones(len(eligible), dtype=bool)
    for column in required:
        valid &= np.isfinite(eligible[column])
    valid &= eligible["_valor_unitario_ajustado"].gt(0).to_numpy()
    valid &= eligible[reference_area_column].gt(0).to_numpy()
    valid &= eligible[mapping.latitude].between(-90, 90).to_numpy()
    valid &= eligible[mapping.longitude].between(-180, 180).to_numpy()
    if not territorial and mapping.ano_construcao:
        valid &= valid_construction_year_mask(
            eligible[mapping.ano_construcao]
        ).to_numpy()
    eligible = eligible.loc[valid].copy()

    if len(eligible) < 10:
        raise ValueError(
            "O backtesting exige pelo menos 10 observações válidas no conjunto escolhido."
        )

    n_sample = min(max(int(sample_size), 10), len(eligible))
    sampled = eligible.sample(n=n_sample, random_state=random_state)

    group_series = None
    if group_column and group_column in data.columns:
        group_series = data[group_column].map(_safe_group_value)

    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, row in sampled.iterrows():
        target = {
            "area_construida": (
                float(row[mapping.area_construida])
                if mapping.area_construida
                and pd.notna(row.get(mapping.area_construida))
                and float(row[mapping.area_construida]) > 0
                else None
            ),
            "area_privativa": (
                float(row[mapping.area_privativa])
                if mapping.area_privativa
                and pd.notna(row.get(mapping.area_privativa))
                and float(row[mapping.area_privativa]) > 0
                else None
            ),
            "siat_area_total_lote": (
                float(row[mapping.siat_area_total_lote])
                if mapping.siat_area_total_lote
                and pd.notna(row.get(mapping.siat_area_total_lote))
                and float(row[mapping.siat_area_total_lote]) > 0
                else None
            ),
            "testada": (
                float(row[mapping.testada])
                if mapping.testada
                and pd.notna(row.get(mapping.testada))
                and float(row[mapping.testada]) > 0
                else None
            ),
            "ano_construcao": (
                float(row[mapping.ano_construcao])
                if not territorial
                and mapping.ano_construcao
                and is_valid_construction_year(
                    row.get(mapping.ano_construcao)
                )
                else None
            ),
            "latitude": float(row[mapping.latitude]),
            "longitude": float(row[mapping.longitude]),
        }

        reference_key_map = {
            mapping.area_construida: "area_construida",
            mapping.area_privativa: "area_privativa",
            mapping.siat_area_total_lote: "siat_area_total_lote",
        }
        key = reference_key_map.get(reference_area_column)
        if key:
            target[reference_area_column] = target[key]

        exclusions: set[Any] = {index}
        group_value = ""
        if group_series is not None:
            group_value = group_series.loc[index]
            if group_value:
                exclusions.update(group_series.index[group_series.eq(group_value)])

        try:
            estimate = estimate_knn(
                preparation=preparation,
                mapping=mapping,
                target=target,
                reference_area_column=reference_area_column,
                min_k=min_k,
                max_k=max_k,
                min_effective_neighbors=min_effective_neighbors,
                similarity_weight=similarity_weight,
                distance_power=distance_power,
                max_individual_weight=max_individual_weight,
                robust_mad_threshold=robust_mad_threshold,
                territorial=territorial,
                exclude_indices=exclusions,
            )
        except Exception as exc:
            failures.append(str(exc))
            continue

        actual = float(row["_valor_unitario_ajustado"])
        predicted = float(estimate.estimated_unit_value)
        error = predicted - actual
        ape = abs(error) / actual
        records.append(
            {
                "indice_original": index,
                "linha_excel": row.get("_row_excel"),
                "grupo_excluido": group_value,
                "tipo_informacao": row.get(mapping.tipo_informacao),
                "valor_unitario_real": actual,
                "valor_unitario_estimado": predicted,
                "erro_unitario": error,
                "erro_percentual": error / actual,
                "erro_percentual_absoluto": ape,
                "k_utilizado": estimate.diagnostics["k_used"],
                "vizinhos_efetivos": estimate.effective_neighbors,
                "peso_maximo": estimate.diagnostics["max_weight_observed"],
                "risco_extrapolacao": estimate.diagnostics["risk_level"],
                "confianca": estimate.diagnostics["confidence_score"],
            }
        )

    predictions = pd.DataFrame(records)
    if len(predictions) < 5:
        detail = failures[0] if failures else "sem detalhe adicional"
        raise ValueError(
            "O backtesting produziu menos de cinco previsões válidas. "
            f"Primeiro erro: {detail}"
        )

    actual = predictions["valor_unitario_real"].to_numpy(dtype=float)
    predicted = predictions["valor_unitario_estimado"].to_numpy(dtype=float)
    errors = predicted - actual
    ape = np.abs(errors) / actual
    ratios = predicted / actual

    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    median_ratio = float(np.median(ratios))
    cod = (
        float(np.mean(np.abs(ratios - median_ratio)) / median_ratio * 100)
        if median_ratio > 0
        else np.nan
    )
    weighted_mean_ratio = float(np.sum(predicted) / np.sum(actual))
    prd = (
        float(np.mean(ratios) / weighted_mean_ratio)
        if weighted_mean_ratio > 0
        else np.nan
    )

    metrics = {
        "n_tested": int(len(predictions)),
        "n_failed": int(n_sample - len(predictions)),
        "mae_unit": float(np.mean(np.abs(errors))),
        "medae_unit": float(np.median(np.abs(errors))),
        "rmse_unit": float(np.sqrt(np.mean(np.square(errors)))),
        "mdape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "median_bias": float(np.median(errors / actual)),
        "r2": r2,
        "cod": cod,
        "prd": prd,
        "median_ratio": median_ratio,
    }
    diagnostics = {
        "evaluation_scope": scope_label,
        "sample_requested": int(sample_size),
        "sample_used": int(n_sample),
        "group_column": group_column or "",
        "random_state": int(random_state),
        "failure_examples": failures[:5],
    }
    return BacktestResult(
        predictions=predictions,
        metrics=metrics,
        diagnostics=diagnostics,
    )
