from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd


MODULE_API_VERSION = "6.13.0"
MODULE_BUILD_ID = "estimador-aluguel-siri-1.0.0-20260904"


DERIVED_AREA_LOTE = "__area_total_lote_efetiva"
DERIVED_AREA_CONSTRUIDA = "__area_construida_efetiva"
DERIVED_REGIME_AREA = "__regime_area_estimativa"
DERIVED_AREA_PRIVATIVA = "__area_privativa_efetiva"
DERIVED_TESTADA = "__testada_efetiva"
DERIVED_TIPO_INFORMACAO = "__tipo_informacao_efetivo"

DERIVED_FINALIDADE_CRAWLER_INFORMADA = (
    "__finalidade_crawler_informada_normalizada"
)
DERIVED_FINALIDADE_SIAT_NORMALIZADA = (
    "__finalidade_siat_normalizada"
)
DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA = (
    "__finalidade_tipo_crawler_normalizada"
)
DERIVED_FINALIDADE_CRAWLER_NORMALIZADA = (
    "__finalidade_crawler_normalizada"
)
DERIVED_FONTE_NORMALIZACAO = "__fonte_normalizacao"
DERIVED_CONFLITO_TIPOLOGICO = "__conflito_tipologico"
DERIVED_CONFIANCA_NORMALIZACAO = "__confianca_normalizacao"
DERIVED_NATUREZA_USO_NORMALIZADA = "__natureza_uso_normalizada"

# Aliases de compatibilidade. O aplicativo 1.11 usa os nomes acima.
DERIVED_SEGMENTO_SIAT = DERIVED_FINALIDADE_SIAT_NORMALIZADA
DERIVED_SEGMENTO_CRAWLER = (
    DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA
)
DERIVED_SEGMENTO_MERCADO = (
    DERIVED_FINALIDADE_CRAWLER_NORMALIZADA
)
DERIVED_FONTE_CLASSIFICACAO = DERIVED_FONTE_NORMALIZACAO
DERIVED_CONFIANCA_CLASSIFICACAO = (
    DERIVED_CONFIANCA_NORMALIZACAO
)


@dataclass(frozen=True)
class SchemaInfo:
    siri_detected: bool
    added_columns: tuple[str, ...]
    notes: tuple[str, ...]


def _normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    replacements = str.maketrans(
        "áàãâäéèêëíìîïóòõôöúùûüç",
        "aaaaaeeeeiiiiooooouuuuc",
    )
    return " ".join(text.translate(replacements).split())




FINALIDADE_APARTAMENTO = "APARTAMENTO"
FINALIDADE_COBERTURA = "COBERTURA"
FINALIDADE_FLAT = "FLAT / APART-HOTEL"
FINALIDADE_CASA = "CASA / RESIDÊNCIA"
FINALIDADE_LOJA = "LOJA"
FINALIDADE_LOJA_GALERIA = "LOJA EM GALERIA"
FINALIDADE_LOJA_SHOPPING = "LOJA EM SHOPPING"
FINALIDADE_SALA = "SALA COMERCIAL"
FINALIDADE_COMERCIAL = "IMÓVEL COMERCIAL"
FINALIDADE_INDUSTRIAL = "GALPÃO / DEPÓSITO"
FINALIDADE_TERRENO = "TERRENO"
FINALIDADE_GLEBA = "GLEBA"
FINALIDADE_CONSTRUCAO_GLEBA = "CONSTRUÇÃO EM ÁREA DE GLEBA"
FINALIDADE_ESTACIONAMENTO = "GARAGEM / VAGA"
FINALIDADE_ESTACIONAMENTO_RESIDENCIAL = (
    "GARAGEM / VAGA RESIDENCIAL"
)
FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL = (
    "GARAGEM / VAGA NÃO RESIDENCIAL"
)
FINALIDADE_HOTEL = "HOTEL"
FINALIDADE_ESPECIAL = "IMÓVEL ESPECIAL"


FINALIDADES_CRAWLER_CANONICAS = (
    FINALIDADE_APARTAMENTO,
    FINALIDADE_COBERTURA,
    FINALIDADE_FLAT,
    FINALIDADE_CASA,
    FINALIDADE_LOJA,
    FINALIDADE_LOJA_GALERIA,
    FINALIDADE_LOJA_SHOPPING,
    FINALIDADE_SALA,
    FINALIDADE_COMERCIAL,
    FINALIDADE_INDUSTRIAL,
    FINALIDADE_TERRENO,
    FINALIDADE_GLEBA,
    FINALIDADE_CONSTRUCAO_GLEBA,
    FINALIDADE_ESTACIONAMENTO,
    FINALIDADE_ESTACIONAMENTO_RESIDENCIAL,
    FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL,
    FINALIDADE_HOTEL,
    FINALIDADE_ESPECIAL,
)


FINALIDADE_CRAWLER_COLUMN_CANDIDATES = (
    "finalidade_crawler_normalizada",
    "finalidade_crawler",
    "finalidade_oferta",
    "crawler_finalidade",
    "crawler_finalidade_pesquisa",
    "finalidade_pesquisa_crawler",
    "finalidade_pesquisa",
    "pesquisa_finalidade",
    "finalidade",
    "finalidade do imóvel",
    "finalidade do imovel",
    "uso do imóvel",
    "uso do imovel",
    "tipo_imovel_pesquisa",
)


_GENERIC_VALUES = {
    "",
    "imovel",
    "outros",
    "outro",
    "nao informado",
    "nao identificada",
    "nao identificado",
    "desconhecido",
    "empreendimento",
    "lancamento",
    "residencial",
}


def normalize_finalidade_crawler(value: object) -> str:
    text = _normalize(value)
    if not text or text in _GENERIC_VALUES:
        return ""

    # Subtipos antes das famílias gerais.
    if any(
        term in text
        for term in ("apart-hotel", "apart hotel", "flat")
    ):
        return FINALIDADE_FLAT

    if "cobertura" in text and "sala" not in text:
        return FINALIDADE_COBERTURA

    if any(
        term in text
        for term in (
            "apartamento",
            "apto",
            "studio",
            "kitnet",
            "kitinete",
            "loft residencial",
        )
    ):
        return FINALIDADE_APARTAMENTO

    if "loja" in text and "shopping" in text:
        return FINALIDADE_LOJA_SHOPPING

    if "loja" in text and "galeria" in text:
        return FINALIDADE_LOJA_GALERIA

    if any(
        term in text
        for term in (
            "sala comercial",
            "sala de cobertura",
            "conjunto comercial",
            "escritorio",
            "office",
            "consultorio",
        )
    ):
        return FINALIDADE_SALA

    if any(
        term in text
        for term in (
            "loja",
            "ponto comercial",
            "loja terrea",
            "loja de interior",
        )
    ):
        return FINALIDADE_LOJA

    if any(
        term in text
        for term in (
            "unidade de comercio",
            "unidade de servico",
            "unidade comercial",
            "imovel comercial",
            "predio comercial",
            "edificio comercial",
            "comercial",
        )
    ):
        return FINALIDADE_COMERCIAL

    if any(
        term in text
        for term in (
            "deposito",
            "armazem",
            "galpao",
            "pavilhao",
            "industrial",
            "fabrica",
            "centro de distribuicao",
        )
    ):
        return FINALIDADE_INDUSTRIAL

    if "construcao em area projetada de gleba" in text:
        return FINALIDADE_CONSTRUCAO_GLEBA

    if "gleba" in text:
        return FINALIDADE_GLEBA

    if any(
        term in text
        for term in (
            "terreno",
            "lote",
            "area rural",
            "chacara",
            "sitio",
            "fazenda",
        )
    ):
        return FINALIDADE_TERRENO

    if any(
        term in text
        for term in (
            "estacionamento",
            "garagem",
            "box",
            "vaga",
        )
    ):
        # "não residencial" deve ser testado antes de "residencial".
        if any(
            term in text
            for term in (
                "nao residencial",
                "nao residenc",
                "não residencial",
                "não residenc",
                "edificio garagem",
                "edifício garagem",
            )
        ):
            return FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL

        if "residencial" in text or "residenc" in text:
            return FINALIDADE_ESTACIONAMENTO_RESIDENCIAL

        return FINALIDADE_ESTACIONAMENTO

    if any(
        term in text
        for term in ("hotel", "pousada", "motel", "hostel")
    ):
        return FINALIDADE_HOTEL

    if any(
        term in text
        for term in (
            "residencia",
            "casa",
            "sobrado",
            "condominio horizontal",
        )
    ):
        return FINALIDADE_CASA

    if "imovel especial" in text or text == "especial":
        return FINALIDADE_ESPECIAL

    return ""


def classify_siat_segment(value: object) -> str:
    """Alias mantido para compatibilidade com versões anteriores."""
    return normalize_finalidade_crawler(value)


def classify_crawler_segment(value: object) -> str:
    """Alias mantido para compatibilidade com versões anteriores."""
    return normalize_finalidade_crawler(value)


def find_finalidade_crawler_column(
    columns: Iterable[str],
) -> str | None:
    return first_existing(
        columns,
        FINALIDADE_CRAWLER_COLUMN_CANDIDATES,
    )


def reference_area_preference(finalidade: object) -> str:
    normalized = normalize_finalidade_crawler(finalidade)

    if normalized in {
        FINALIDADE_TERRENO,
        FINALIDADE_GLEBA,
    }:
        return "terreno"

    if normalized in {
        FINALIDADE_APARTAMENTO,
        FINALIDADE_COBERTURA,
        FINALIDADE_FLAT,
        FINALIDADE_SALA,
        FINALIDADE_ESTACIONAMENTO,
        FINALIDADE_ESTACIONAMENTO_RESIDENCIAL,
        FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL,
    }:
        return "privativa"

    if normalized in {
        FINALIDADE_LOJA,
        FINALIDADE_LOJA_GALERIA,
        FINALIDADE_LOJA_SHOPPING,
        FINALIDADE_COMERCIAL,
    }:
        return "privativa_ou_construida"

    return "construida"



def natureza_uso_normalizada(finalidade: object) -> str:
    normalized = normalize_finalidade_crawler(finalidade)

    if normalized in {
        FINALIDADE_APARTAMENTO,
        FINALIDADE_COBERTURA,
        FINALIDADE_FLAT,
        FINALIDADE_CASA,
        FINALIDADE_ESTACIONAMENTO_RESIDENCIAL,
    }:
        return "RESIDENCIAL"

    if normalized in {
        FINALIDADE_LOJA,
        FINALIDADE_LOJA_GALERIA,
        FINALIDADE_LOJA_SHOPPING,
        FINALIDADE_SALA,
        FINALIDADE_COMERCIAL,
        FINALIDADE_INDUSTRIAL,
        FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL,
        FINALIDADE_HOTEL,
    }:
        return "NÃO RESIDENCIAL"

    if normalized in {
        FINALIDADE_TERRENO,
        FINALIDADE_GLEBA,
        FINALIDADE_CONSTRUCAO_GLEBA,
    }:
        return "TERRITORIAL"

    return "INDETERMINADA"


def _major_family(finalidade: object) -> str:
    normalized = normalize_finalidade_crawler(finalidade)

    if normalized in {
        FINALIDADE_APARTAMENTO,
        FINALIDADE_COBERTURA,
        FINALIDADE_FLAT,
    }:
        return "RESIDENCIAL_MULTIFAMILIAR"

    if normalized == FINALIDADE_CASA:
        return "RESIDENCIAL_UNIFAMILIAR"

    if normalized in {
        FINALIDADE_LOJA,
        FINALIDADE_LOJA_GALERIA,
        FINALIDADE_LOJA_SHOPPING,
        FINALIDADE_SALA,
        FINALIDADE_COMERCIAL,
    }:
        return "COMERCIAL"

    if normalized == FINALIDADE_INDUSTRIAL:
        return "INDUSTRIAL"

    if normalized in {
        FINALIDADE_TERRENO,
        FINALIDADE_GLEBA,
        FINALIDADE_CONSTRUCAO_GLEBA,
    }:
        return "TERRITORIAL"

    if normalized == FINALIDADE_ESTACIONAMENTO_RESIDENCIAL:
        return "ESTACIONAMENTO_RESIDENCIAL"

    if normalized == FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL:
        return "ESTACIONAMENTO_NAO_RESIDENCIAL"

    if normalized == FINALIDADE_ESTACIONAMENTO:
        return "ESTACIONAMENTO"

    if normalized == FINALIDADE_HOTEL:
        return "HOSPEDAGEM"

    if normalized == FINALIDADE_ESPECIAL:
        return "ESPECIAL"

    return ""


def _comparison_level(
    chosen: object,
    other: object,
) -> str:
    chosen_text = str(chosen or "").strip()
    other_text = str(other or "").strip()

    if not chosen_text or not other_text:
        return "Não aplicável"

    if _normalize(chosen_text) == _normalize(other_text):
        return "Não"

    parking_values = {
        FINALIDADE_ESTACIONAMENTO,
        FINALIDADE_ESTACIONAMENTO_RESIDENCIAL,
        FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL,
    }
    if chosen_text in parking_values and other_text in parking_values:
        # A categoria genérica é compatível com uma categoria específica.
        # Residencial versus não residencial é conflito efetivo.
        if FINALIDADE_ESTACIONAMENTO in {chosen_text, other_text}:
            return "Não"
        return "Sim"

    chosen_family = _major_family(chosen_text)
    other_family = _major_family(other_text)
    if chosen_family and chosen_family == other_family:
        return "Moderado"

    return "Sim"


def _combine_conflicts(levels: Iterable[str]) -> str:
    usable = [
        level
        for level in levels
        if level and level != "Não aplicável"
    ]
    if not usable:
        return "Não aplicável"
    if "Sim" in usable:
        return "Sim"
    if "Moderado" in usable:
        return "Moderado"
    return "Não"


def _derive_normalized_finality(
    data: pd.DataFrame,
    type_column: str | None,
    siat_column: str | None,
    crawler_type_column: str | None,
    finalidade_crawler_column: str | None,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    index = data.index
    blank = pd.Series("", index=index, dtype="string")

    finalidade_informada = (
        data[finalidade_crawler_column]
        .map(normalize_finalidade_crawler)
        .astype("string")
        if (
            finalidade_crawler_column
            and finalidade_crawler_column in data.columns
        )
        else blank.copy()
    )
    finalidade_siat = (
        data[siat_column]
        .map(normalize_finalidade_crawler)
        .astype("string")
        if siat_column and siat_column in data.columns
        else blank.copy()
    )
    finalidade_tipo_crawler = (
        data[crawler_type_column]
        .map(normalize_finalidade_crawler)
        .astype("string")
        if (
            crawler_type_column
            and crawler_type_column in data.columns
        )
        else blank.copy()
    )

    type_norm = (
        data[type_column].map(_normalize)
        if type_column and type_column in data.columns
        else blank.copy()
    )
    offer_mask = type_norm.isin({"oferta", "oferta aluguel"})
    itbi_mask = type_norm.eq("guia itbi")

    normalized = blank.copy()
    source = blank.copy()

    parking_specific = {
        FINALIDADE_ESTACIONAMENTO_RESIDENCIAL,
        FINALIDADE_ESTACIONAMENTO_NAO_RESIDENCIAL,
    }

    # A categoria genérica pode ser refinada pelo SIAT sem mudar a
    # família mercadológica do registro.
    informed_parking_refinement = (
        finalidade_informada.eq(FINALIDADE_ESTACIONAMENTO)
        & finalidade_siat.isin(parking_specific)
    )
    finalidade_informada.loc[informed_parking_refinement] = (
        finalidade_siat.loc[informed_parking_refinement]
    )

    crawler_parking_refinement = (
        finalidade_tipo_crawler.eq(FINALIDADE_ESTACIONAMENTO)
        & finalidade_siat.isin(parking_specific)
    )
    finalidade_tipo_crawler.loc[crawler_parking_refinement] = (
        finalidade_siat.loc[crawler_parking_refinement]
    )

    # A finalidade escolhida na pesquisa é a primeira referência para
    # qualquer registro em que esteja preenchida.
    informed_mask = finalidade_informada.ne("")
    normalized.loc[informed_mask] = finalidade_informada.loc[
        informed_mask
    ]
    source.loc[informed_mask] = "FINALIDADE_CRAWLER"
    source.loc[informed_parking_refinement] = (
        "FINALIDADE_CRAWLER_REFINADA_SIAT"
    )

    # Em ofertas sem a finalidade explícita, usa-se o tipo identificado
    # pelo crawler.
    crawler_offer_mask = (
        ~informed_mask
        & offer_mask
        & finalidade_tipo_crawler.ne("")
    )
    normalized.loc[crawler_offer_mask] = (
        finalidade_tipo_crawler.loc[crawler_offer_mask]
    )
    source.loc[crawler_offer_mask] = "TIPO_CRAWLER"
    source.loc[
        crawler_offer_mask & crawler_parking_refinement
    ] = "TIPO_CRAWLER_REFINADO_SIAT"

    # Guias ITBI são convertidas do SIAT para a mesma taxonomia.
    siat_itbi_mask = (
        ~informed_mask
        & itbi_mask
        & finalidade_siat.ne("")
    )
    normalized.loc[siat_itbi_mask] = finalidade_siat.loc[
        siat_itbi_mask
    ]
    source.loc[siat_itbi_mask] = "SIAT"

    # Ofertas sem classificação de mercado utilizável recorrem ao SIAT.
    siat_offer_mask = (
        ~informed_mask
        & offer_mask
        & finalidade_tipo_crawler.eq("")
        & finalidade_siat.ne("")
    )
    normalized.loc[siat_offer_mask] = finalidade_siat.loc[
        siat_offer_mask
    ]
    source.loc[siat_offer_mask] = "SIAT_FALLBACK"

    # Demais linhas válidas também são normalizadas pelo SIAT.
    other_siat_mask = (
        normalized.eq("")
        & finalidade_siat.ne("")
    )
    normalized.loc[other_siat_mask] = finalidade_siat.loc[
        other_siat_mask
    ]
    source.loc[other_siat_mask] = "SIAT"

    conflict = pd.Series(
        "Não aplicável",
        index=index,
        dtype="string",
    )
    for row_index in index:
        chosen = normalized.loc[row_index]
        conflict.loc[row_index] = _combine_conflicts(
            (
                _comparison_level(
                    chosen,
                    finalidade_siat.loc[row_index],
                ),
                _comparison_level(
                    chosen,
                    finalidade_tipo_crawler.loc[row_index],
                ),
            )
        )

    confidence = pd.Series("Baixa", index=index, dtype="string")

    source_finality = source.isin(
        {
            "FINALIDADE_CRAWLER",
            "FINALIDADE_CRAWLER_REFINADA_SIAT",
        }
    )
    explicit_and_crawler_match = (
        source_finality
        & finalidade_tipo_crawler.ne("")
        & (
            finalidade_informada.map(_normalize)
            == finalidade_tipo_crawler.map(_normalize)
        )
    )
    confidence.loc[explicit_and_crawler_match] = "Alta"
    confidence.loc[
        source_finality & ~explicit_and_crawler_match
    ] = "Média"
    confidence.loc[
        source_finality
        & finalidade_tipo_crawler.ne("")
        & (
            _comparison_level_series(
                finalidade_informada,
                finalidade_tipo_crawler,
            )
            == "Sim"
        )
    ] = "Baixa"

    crawler_source = source.isin(
        {
            "TIPO_CRAWLER",
            "TIPO_CRAWLER_REFINADO_SIAT",
        }
    )
    confidence.loc[
        crawler_source
        & conflict.isin({"Não", "Não aplicável"})
    ] = "Alta"
    confidence.loc[
        crawler_source
        & conflict.isin({"Moderado", "Sim"})
    ] = "Média"

    confidence.loc[source.eq("SIAT") & itbi_mask] = "Alta"
    confidence.loc[source.eq("SIAT") & ~itbi_mask] = "Média"
    confidence.loc[source.eq("SIAT_FALLBACK")] = "Baixa"
    confidence.loc[
        source.isin(
            {
                "FINALIDADE_CRAWLER_REFINADA_SIAT",
                "TIPO_CRAWLER_REFINADO_SIAT",
            }
        )
    ] = "Média"

    return (
        finalidade_informada,
        finalidade_siat,
        finalidade_tipo_crawler,
        normalized,
        source,
        conflict,
        confidence,
    )


def _comparison_level_series(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            _comparison_level(a, b)
            for a, b in zip(left, right)
        ],
        index=left.index,
        dtype="string",
    )


def _normalized_header(value: object) -> str:
    """Normaliza pontuação e unidades usuais sem tornar aliases ambíguos."""
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", _normalize(value))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = [str(column) for column in columns]
    exact_lookup: dict[str, str] = {}
    relaxed_lookup: dict[str, list[str]] = {}
    for column in column_list:
        exact_lookup.setdefault(_normalize(column), column)
        relaxed_lookup.setdefault(_normalized_header(column), []).append(column)

    for candidate in candidates:
        match = exact_lookup.get(_normalize(candidate))
        if match is not None:
            return match

    # Cabeçalhos como "Preço (R$)" e "Área construída (m²)" são
    # reconhecidos somente se a forma simplificada apontar para uma única
    # coluna, evitando escolher silenciosamente entre campos ambíguos.
    for candidate in candidates:
        matches = relaxed_lookup.get(_normalized_header(candidate), [])
        if len(matches) == 1:
            return matches[0]
    return None


def _numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    series = df[column]
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


def _coalesce_positive(df: pd.DataFrame, columns: Iterable[str | None]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column and column in df.columns:
            resolved = column
        elif column:
            resolved = first_existing(df.columns, [column])
        else:
            resolved = None
        values = _numeric(df, resolved)
        values = values.where(np.isfinite(values) & (values > 0))
        result = result.where(result.notna(), values)
    return result


def _combine_by_information_type(
    df: pd.DataFrame,
    type_column: str | None,
    offer_columns: Iterable[str | None],
    itbi_columns: Iterable[str | None],
    fallback_columns: Iterable[str | None],
) -> pd.Series:
    fallback = _coalesce_positive(df, fallback_columns)
    if not type_column or type_column not in df.columns:
        return fallback

    type_norm = df[type_column].map(_normalize)
    offer_mask = type_norm.isin({"oferta", "oferta aluguel"})
    itbi_mask = type_norm.eq("guia itbi")

    result = fallback.copy()
    offer_values = _coalesce_positive(df, offer_columns)
    itbi_values = _coalesce_positive(df, itbi_columns)
    result.loc[offer_mask] = offer_values.loc[offer_mask]
    result.loc[itbi_mask] = itbi_values.loc[itbi_mask]
    return result


def _canonical_information_type(value: object) -> str:
    normalized = _normalize(value)
    if not normalized:
        return ""
    if any(
        term in normalized
        for term in ("aluguel", "locacao", "arrendamento")
    ):
        return "Oferta Aluguel"
    if "itbi" in normalized or normalized in {"transacao", "transação"}:
        return "Guia ITBI"
    if any(
        term in normalized
        for term in ("oferta", "anuncio", "anúncio", "venda", "listing")
    ):
        return "Oferta"
    return str(value).strip()


def enrich_known_schemas(df: pd.DataFrame) -> tuple[pd.DataFrame, SchemaInfo]:
    """Cria colunas efetivas sem alterar as colunas originais do arquivo."""
    data = df.copy()
    columns = [str(column) for column in data.columns]
    data.columns = columns

    source_type_column = first_existing(
        columns,
        [
            "tipo_informacao",
            "tipo_informação",
            "tipo de informação",
            "tipo da informação",
            "tipo da informacao",
            "natureza da informação",
            "natureza da informacao",
            "transaction_type",
            "listing_type",
        ],
    )

    siri_markers = {
        "tipo_informacao",
        "siat_finalidade_descricao",
        "valor_oferta",
        "siat_latitude",
        "siat_longitude",
    }
    siri_detected = siri_markers.issubset(set(columns))

    added: list[str] = []
    notes: list[str] = []

    if source_type_column:
        information_type = (
            data[source_type_column]
            .map(_canonical_information_type)
            .astype("string")
        )
        if not siri_detected:
            information_type = information_type.mask(
                information_type.fillna("").str.strip().eq(""),
                "Oferta Aluguel",
            )
    else:
        # Bases de raspagem contêm anúncios, não Guias ITBI. A hipótese fica
        # materializada em uma coluna derivada e nas notas de auditoria.
        information_type = pd.Series("Oferta Aluguel", index=data.index, dtype="string")

    data[DERIVED_TIPO_INFORMACAO] = information_type
    added.append(DERIVED_TIPO_INFORMACAO)
    type_column = DERIVED_TIPO_INFORMACAO
    if source_type_column:
        notes.append(
            "Tipo da informação normalizado a partir da coluna "
            f"'{source_type_column}'."
        )
    else:
        notes.append(
            "Tipo da informação ausente: os registros foram tratados como "
            "ofertas de aluguel, hipótese compatível com bases de anúncios de locação."
        )


    siat_finality_column = first_existing(
        columns,
        ["siat_finalidade_descricao"],
    )
    crawler_type_column = first_existing(
        columns,
        [
            "crawler_tipo_imovel_normalizado",
            "tipo_imovel",
            "tipo do imóvel",
            "tipo do imovel",
            "tipo",
            "property_type",
            "property type",
            "categoria do imóvel",
            "categoria do imovel",
        ],
    )
    finalidade_crawler_column = find_finalidade_crawler_column(
        columns
    )

    (
        finalidade_crawler_informada,
        finalidade_siat_normalizada,
        finalidade_tipo_crawler_normalizada,
        finalidade_crawler_normalizada,
        fonte_normalizacao,
        conflito_tipologico,
        confianca_normalizacao,
    ) = _derive_normalized_finality(
        data,
        type_column=type_column,
        siat_column=siat_finality_column,
        crawler_type_column=crawler_type_column,
        finalidade_crawler_column=finalidade_crawler_column,
    )

    data[DERIVED_FINALIDADE_CRAWLER_INFORMADA] = (
        finalidade_crawler_informada
    )
    data[DERIVED_FINALIDADE_SIAT_NORMALIZADA] = (
        finalidade_siat_normalizada
    )
    data[DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA] = (
        finalidade_tipo_crawler_normalizada
    )
    data[DERIVED_FINALIDADE_CRAWLER_NORMALIZADA] = (
        finalidade_crawler_normalizada
    )
    data[DERIVED_FONTE_NORMALIZACAO] = fonte_normalizacao
    data[DERIVED_CONFLITO_TIPOLOGICO] = conflito_tipologico
    data[DERIVED_CONFIANCA_NORMALIZACAO] = (
        confianca_normalizacao
    )
    data[DERIVED_NATUREZA_USO_NORMALIZADA] = (
        finalidade_crawler_normalizada
        .map(natureza_uso_normalizada)
        .astype("string")
    )

    added.extend(
        [
            DERIVED_FINALIDADE_CRAWLER_INFORMADA,
            DERIVED_FINALIDADE_SIAT_NORMALIZADA,
            DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA,
            DERIVED_FINALIDADE_CRAWLER_NORMALIZADA,
            DERIVED_FONTE_NORMALIZACAO,
            DERIVED_CONFLITO_TIPOLOGICO,
            DERIVED_CONFIANCA_NORMALIZACAO,
            DERIVED_NATUREZA_USO_NORMALIZADA,
        ]
    )

    if finalidade_crawler_column:
        notes.append(
            "Finalidade da pesquisa identificada na coluna "
            f"'{finalidade_crawler_column}'."
        )
    else:
        notes.append(
            "A planilha não contém coluna explícita de finalidade_crawler; "
            "ofertas usam o tipo normalizado do crawler e, quando ausente, "
            "o SIAT."
        )

    notes.append(
        "Todos os registros foram convertidos para a taxonomia única "
        "finalidade_crawler_normalizada."
    )
    lot_series = _combine_by_information_type(
        data,
        type_column,
        offer_columns=[
            "crawler_area_terreno",
            "siat_area_terreno",
            "area_terreno",
            "area_terreno_m2",
            "area_do_terreno",
            "area_lote",
            "area_do_lote",
            "terreno_m2",
            "lote_m2",
            "land_area",
            "lot_area",
        ],
        itbi_columns=["siat_area_terreno", "crawler_area_terreno"],
        fallback_columns=[
            "siat_area_total_lote",
            "area_total_lote",
            "siat_area_terreno",
            "crawler_area_terreno",
            "area_terreno",
            "area_terreno_m2",
            "area_do_terreno",
            "area_lote",
            "area_do_lote",
            "terreno_m2",
            "lote_m2",
            "land_area",
            "lot_area",
        ],
    )
    if lot_series.notna().any():
        data[DERIVED_AREA_LOTE] = lot_series
        added.append(DERIVED_AREA_LOTE)
        notes.append(
            "Área do lote combinada: prioriza a área anunciada nas ofertas e "
            "a área SIAT nas Guias ITBI."
        )

    built_series = _combine_by_information_type(
        data,
        type_column,
        offer_columns=[
            "crawler_area_construida",
            "area_construida_descrita_m2",
            "area_construida",
            "siat_area_construida",
            "itbacotot",
            "area_total_construida",
            "area_edificada",
            "area_anunciada_m2",
            "area_total_descrita_m2",
            "area_util",
            "area_do_imovel",
            "area_imovel",
            "built_area",
            "property_area",
            "area",
        ],
        itbi_columns=[
            "itbacotot",
            "area_construida",
            "siat_area_construida",
            "crawler_area_construida",
        ],
        fallback_columns=[
            "area_construida_descrita_m2",
            "area_construida",
            "crawler_area_construida",
            "siat_area_construida",
            "itbacotot",
            "area_total_construida",
            "area_edificada",
            "area_anunciada_m2",
            "area_total_descrita_m2",
            "area_do_imovel",
            "area_imovel",
            "built_area",
            "property_area",
            "area",
        ],
    )
    if built_series.notna().any():
        data[DERIVED_AREA_CONSTRUIDA] = built_series
        added.append(DERIVED_AREA_CONSTRUIDA)
        notes.append(
            "Área construída combinada conforme a origem do registro."
        )

    private_series = _combine_by_information_type(
        data,
        type_column,
        offer_columns=[
            "crawler_area_privativa",
            "area_privativa_descrita_m2",
            "area_privativa",
            "itbacopriv",
            "area_anunciada_m2",
            "area_util",
            "area_do_imovel",
            "area_imovel",
            "property_area",
            "area",
        ],
        itbi_columns=["itbacopriv", "area_privativa", "crawler_area_privativa"],
        fallback_columns=[
            "area_privativa_descrita_m2",
            "area_privativa",
            "crawler_area_privativa",
            "itbacopriv",
            "area_anunciada_m2",
            "area_util",
            "area_do_imovel",
            "area_imovel",
            "property_area",
            "area",
        ],
    )
    if private_series.notna().any():
        data[DERIVED_AREA_PRIVATIVA] = private_series
        added.append(DERIVED_AREA_PRIVATIVA)
        notes.append(
            "Área privativa combinada conforme a origem do registro."
        )

    frontage_series = _combine_by_information_type(
        data,
        type_column,
        offer_columns=[
            "anuncio_testada",
            "siat_testada_terreno",
            "testada",
            "testada_terreno",
        ],
        itbi_columns=[
            "siat_testada_terreno",
            "testada",
            "testada_terreno",
            "anuncio_testada",
        ],
        fallback_columns=[
            "testada",
            "testada_terreno",
            "siat_testada_terreno",
            "anuncio_testada",
        ],
    )
    if frontage_series.notna().any():
        data[DERIVED_TESTADA] = frontage_series
        added.append(DERIVED_TESTADA)
        notes.append(
            "Testada combinada: prioriza a testada do anúncio nas ofertas e "
            "a testada SIAT nas Guias ITBI."
        )

    return data, SchemaInfo(
        siri_detected=siri_detected,
        added_columns=tuple(added),
        notes=tuple(notes),
    )


def friendly_column_name(column: str) -> str:
    labels = {
        "siat_ano": "Ano da construção",
        DERIVED_AREA_LOTE: "Área total do lote — combinada automaticamente",
        DERIVED_AREA_CONSTRUIDA: (
            "Área total/construída — combinada automaticamente"
        ),
        DERIVED_REGIME_AREA: "Regime de área da estimativa",
        DERIVED_AREA_PRIVATIVA: "Área privativa — combinada automaticamente",
        DERIVED_TESTADA: "Testada — combinada automaticamente",
        DERIVED_TIPO_INFORMACAO: "Tipo da informação — efetivo",
        DERIVED_FINALIDADE_CRAWLER_INFORMADA: (
            "Finalidade da pesquisa — normalizada"
        ),
        DERIVED_FINALIDADE_SIAT_NORMALIZADA: (
            "Finalidade SIAT — normalizada"
        ),
        DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA: (
            "Tipo identificado pelo crawler — normalizado"
        ),
        DERIVED_FINALIDADE_CRAWLER_NORMALIZADA: (
            "Finalidade crawler utilizada pelo estimador"
        ),
        DERIVED_FONTE_NORMALIZACAO: (
            "Fonte da finalidade normalizada"
        ),
        DERIVED_CONFLITO_TIPOLOGICO: (
            "Conflito entre classificações de origem"
        ),
        DERIVED_CONFIANCA_NORMALIZACAO: (
            "Confiança da finalidade normalizada"
        ),
        DERIVED_NATUREZA_USO_NORMALIZADA: (
            "Natureza de uso normalizada"
        ),
    }
    return labels.get(column, column)
