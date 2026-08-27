from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import sys
import types

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from geocodificador_porto_alegre import (
    geocode_porto_alegre_address,
    load_street_axis_index,
    parse_street_and_number,
)
from vera_pdf_report import (
    build_inference_report_pdf,
    calculate_comparable_cod,
)

APP_NAME = "estimador_knn_siri"
APP_EDITION = "LITE 1.19.0"
CORE_VERSION = "6.12.0"

# Parâmetros internos: não ficam expostos ao usuário da edição LITE.
MIN_K = 12
MAX_K = 25
MIN_EFFECTIVE_NEIGHBORS = 11.0
SIMILARITY_WEIGHT = 0.45
DISTANCE_POWER = 0.35
MAX_INDIVIDUAL_WEIGHT = 0.25
ROBUST_MAD_THRESHOLD = 1.25
DISCOUNT_CAP = 0.20


st.set_page_config(
    page_title="Estimador KNN SIRI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODULE_BUILD_ID = "estimador-knn-siri-lite-1.19.0-20260826"
CORE_MODULE_FILE = "estimador_knn_core_v6120.py"
SCHEMA_MODULE_FILE = "estimador_knn_schema_v6120.py"


def _load_exact_source_module(
    filename: str,
    internal_name: str,
):
    """
    Lê, compila e executa diretamente o arquivo ao lado do app.py.

    Não utiliza o mecanismo normal de importação, não consulta outros
    diretórios do Python e não reutiliza módulos antigos do sys.modules.
    """
    module_path = Path(__file__).resolve().parent / filename
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Arquivo interno não encontrado: {module_path}"
        )

    source_text = module_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()[:12]

    module = types.ModuleType(internal_name)
    module.__file__ = str(module_path)
    module.__package__ = ""
    module.__loader__ = None

    # Necessário para dataclasses e, ao mesmo tempo, elimina qualquer
    # objeto anterior que tenha usado o mesmo nome interno.
    sys.modules.pop(internal_name, None)
    sys.modules[internal_name] = module

    compiled = compile(
        source_text,
        str(module_path),
        "exec",
        dont_inherit=True,
    )
    exec(compiled, module.__dict__)
    return module, module_path, source_hash


try:
    _knn, _knn_path, _knn_hash = _load_exact_source_module(
        CORE_MODULE_FILE,
        "_estimador_knn_core_runtime_v6120",
    )
    _schema, _schema_path, _schema_hash = _load_exact_source_module(
        SCHEMA_MODULE_FILE,
        "_estimador_knn_schema_runtime_v6120",
    )
except Exception as exc:
    st.error(
        "Os módulos exclusivos do aplicativo não puderam ser carregados."
    )
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


_required_knn = {
    "ColumnMapping",
    "EstimateResult",
    "PreparationResult",
    "estimate_knn",
    "normalize_text",
    "prepare_data",
    "CALIBRATED_GLOBAL_PARAMETERS",
    "CALIBRATED_PURPOSE_PARAMETERS",
    "calibrated_parameters_for_purpose",
    "AREA_REGIME_PRIVATE",
    "AREA_REGIME_TOTAL_BUILT",
    "choose_area_regime",
    "preferred_area_regime",
    "area_floor_is_compatible",
    "MIN_CONSTRUCTION_YEAR",
    "MAX_CONSTRUCTION_YEAR",
    "valid_construction_year_mask",
}
_required_schema = {
    "DERIVED_AREA_CONSTRUIDA",
    "DERIVED_AREA_LOTE",
    "DERIVED_AREA_PRIVATIVA",
    "DERIVED_REGIME_AREA",
    "DERIVED_TESTADA",
    "DERIVED_TIPO_INFORMACAO",
    "DERIVED_FINALIDADE_CRAWLER_INFORMADA",
    "DERIVED_FINALIDADE_SIAT_NORMALIZADA",
    "DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA",
    "DERIVED_FINALIDADE_CRAWLER_NORMALIZADA",
    "DERIVED_FONTE_NORMALIZACAO",
    "DERIVED_CONFLITO_TIPOLOGICO",
    "DERIVED_CONFIANCA_NORMALIZACAO",
    "DERIVED_NATUREZA_USO_NORMALIZADA",
    "find_finalidade_crawler_column",
    "normalize_finalidade_crawler",
    "natureza_uso_normalizada",
    "reference_area_preference",
    "enrich_known_schemas",
    "first_existing",
    "friendly_column_name",
}

_missing_knn = sorted(name for name in _required_knn if not hasattr(_knn, name))
_missing_schema = sorted(
    name for name in _required_schema if not hasattr(_schema, name)
)
_knn_version = getattr(_knn, "MODULE_API_VERSION", "anterior")
_schema_version = getattr(_schema, "MODULE_API_VERSION", "anterior")
_knn_build = getattr(_knn, "MODULE_BUILD_ID", "anterior")
_schema_build = getattr(_schema, "MODULE_BUILD_ID", "anterior")

if (
    _missing_knn
    or _missing_schema
    or _knn_version != CORE_VERSION
    or _schema_version != CORE_VERSION
    or _knn_build != MODULE_BUILD_ID
    or _schema_build != MODULE_BUILD_ID
):
    st.error("Os módulos exclusivos publicados não correspondem a esta edição.")
    st.code(
        "\n".join(
            [
                f"Núcleo carregado: {_knn_path}",
                f"Versão/build do núcleo: {_knn_version} / {_knn_build}",
                f"SHA do núcleo: {_knn_hash}",
                f"Schema carregado: {_schema_path}",
                f"Versão/build do schema: {_schema_version} / {_schema_build}",
                f"SHA do schema: {_schema_hash}",
                "Itens ausentes no KNN: "
                + (", ".join(_missing_knn) if _missing_knn else "nenhum"),
                "Itens ausentes no schema: "
                + (", ".join(_missing_schema) if _missing_schema else "nenhum"),
            ]
        )
    )
    st.stop()

ColumnMapping = _knn.ColumnMapping
EstimateResult = _knn.EstimateResult
PreparationResult = _knn.PreparationResult
estimate_knn = _knn.estimate_knn
normalize_text = _knn.normalize_text
prepare_data = _knn.prepare_data
CALIBRATED_GLOBAL_PARAMETERS = _knn.CALIBRATED_GLOBAL_PARAMETERS
CALIBRATED_PURPOSE_PARAMETERS = _knn.CALIBRATED_PURPOSE_PARAMETERS
calibrated_parameters_for_purpose = (
    _knn.calibrated_parameters_for_purpose
)
AREA_REGIME_PRIVATE = _knn.AREA_REGIME_PRIVATE
AREA_REGIME_TOTAL_BUILT = _knn.AREA_REGIME_TOTAL_BUILT
choose_area_regime = _knn.choose_area_regime
preferred_area_regime = _knn.preferred_area_regime
area_floor_is_compatible = _knn.area_floor_is_compatible
MIN_CONSTRUCTION_YEAR = _knn.MIN_CONSTRUCTION_YEAR
MAX_CONSTRUCTION_YEAR = _knn.MAX_CONSTRUCTION_YEAR
valid_construction_year_mask = _knn.valid_construction_year_mask

DERIVED_AREA_CONSTRUIDA = _schema.DERIVED_AREA_CONSTRUIDA
DERIVED_AREA_LOTE = _schema.DERIVED_AREA_LOTE
DERIVED_AREA_PRIVATIVA = _schema.DERIVED_AREA_PRIVATIVA
DERIVED_REGIME_AREA = _schema.DERIVED_REGIME_AREA
DERIVED_TESTADA = _schema.DERIVED_TESTADA
DERIVED_TIPO_INFORMACAO = _schema.DERIVED_TIPO_INFORMACAO
DERIVED_FINALIDADE_CRAWLER_INFORMADA = (
    _schema.DERIVED_FINALIDADE_CRAWLER_INFORMADA
)
DERIVED_FINALIDADE_SIAT_NORMALIZADA = (
    _schema.DERIVED_FINALIDADE_SIAT_NORMALIZADA
)
DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA = (
    _schema.DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA
)
DERIVED_FINALIDADE_CRAWLER_NORMALIZADA = (
    _schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA
)
DERIVED_FONTE_NORMALIZACAO = _schema.DERIVED_FONTE_NORMALIZACAO
DERIVED_CONFLITO_TIPOLOGICO = _schema.DERIVED_CONFLITO_TIPOLOGICO
DERIVED_CONFIANCA_NORMALIZACAO = (
    _schema.DERIVED_CONFIANCA_NORMALIZACAO
)
DERIVED_NATUREZA_USO_NORMALIZADA = (
    _schema.DERIVED_NATUREZA_USO_NORMALIZADA
)
find_finalidade_crawler_column = (
    _schema.find_finalidade_crawler_column
)
normalize_finalidade_crawler = (
    _schema.normalize_finalidade_crawler
)
natureza_uso_normalizada = _schema.natureza_uso_normalizada
reference_area_preference = _schema.reference_area_preference
enrich_known_schemas = _schema.enrich_known_schemas
first_existing = _schema.first_existing
friendly_column_name = _schema.friendly_column_name


CUSTOM_CSS = """
<style>
:root {
    --ink: #172033;
    --muted: #667085;
    --line: #E4E9F0;
    --canvas: #F6F8FA;
    --navy: #173B57;
    --teal: #0E7C7B;
    --soft-teal: #E9F5F4;
    --soft-blue: #EEF4F8;
    --amber: #A15C00;
    --red: #B42318;
}
.stApp {
    background:
        radial-gradient(circle at 92% 4%, rgba(14,124,123,.09), transparent 23rem),
        var(--canvas);
    color: var(--ink);
}
.block-container {
    max-width: 1220px;
    padding-top: 1.65rem;
    padding-bottom: 4rem;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.2rem;
    border: 1px solid rgba(23,59,87,.12);
    border-radius: 24px;
    background: linear-gradient(135deg, #FFFFFF 0%, #EDF7F6 100%);
    box-shadow: 0 18px 45px rgba(23,59,87,.075);
    margin-bottom: 1.25rem;
}
.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -100px;
    top: -145px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(14,124,123,.18), rgba(23,59,87,.02));
}
.badge {
    display: inline-flex;
    padding: .32rem .68rem;
    border-radius: 999px;
    background: rgba(14,124,123,.10);
    color: #096968;
    font-size: .74rem;
    font-weight: 800;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.hero h1 {
    max-width: 820px;
    margin: .82rem 0 .65rem;
    font-size: clamp(2rem, 4vw, 3.15rem);
    line-height: 1.03;
    letter-spacing: -.045em;
    color: var(--ink);
}
.hero p {
    max-width: 820px;
    margin: 0;
    color: var(--muted);
    line-height: 1.65;
    font-size: 1.01rem;
}
.step {
    display: flex;
    align-items: center;
    gap: .65rem;
    margin: 1.25rem 0 .65rem;
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.85rem;
    height: 1.85rem;
    border-radius: 50%;
    background: var(--navy);
    color: #fff;
    font-weight: 800;
    font-size: .86rem;
}
.step-title {
    font-size: 1.16rem;
    font-weight: 800;
    letter-spacing: -.015em;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,.88);
    box-shadow: 0 10px 28px rgba(23,59,87,.035);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,.95);
    border: 1px solid var(--line);
    padding: 1rem 1.05rem;
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(23,59,87,.04);
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    color: var(--ink);
    letter-spacing: -.03em;
}
.stButton > button, .stDownloadButton > button {
    min-height: 2.9rem;
    border-radius: 12px;
    border: 0;
    font-weight: 800;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--navy), var(--teal));
    box-shadow: 0 10px 22px rgba(14,124,123,.18);
}
.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
    padding: .35rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(255,255,255,.75);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: .45rem .9rem;
}
.stTabs [aria-selected="true"] {
    color: var(--teal);
    background: var(--soft-teal);
}
.risk-card {
    min-height: 132px;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-left-width: 5px;
    border-radius: 16px;
    background: #fff;
}
.risk-card.low { border-left-color: var(--teal); }
.risk-card.moderate { border-left-color: #D97706; }
.risk-card.high { border-left-color: var(--red); }
.risk-label {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.risk-value {
    margin: .25rem 0;
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 850;
}
.risk-text {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.45;
}
.helper {
    padding: .85rem 1rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(238,244,248,.75);
    color: var(--muted);
    font-size: .9rem;
    line-height: 1.5;
}
.small-note {
    color: var(--muted);
    font-size: .84rem;
}
div[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 14px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def money_br(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    text = f"{value:,.{decimals}f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def number_br(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def percent_br(value: float, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value * 100:.{decimals}f}%".replace(".", ",")


@st.cache_data(show_spinner=False, max_entries=8)
def inference_report_pdf(
    *,
    estimated_unit_value: float,
    estimated_total_value: float,
    confidence_score: float,
    neighbors: pd.DataFrame,
    purpose: str,
    area_regime_label: str,
    target: dict[str, object],
    latitude_column: str | None,
    longitude_column: str | None,
    type_column: str | None,
    reference_area_column: str | None,
) -> bytes:
    """Gera e memoriza o PDF associado a uma estimativa concluída."""
    return build_inference_report_pdf(
        estimated_unit_value=estimated_unit_value,
        estimated_total_value=estimated_total_value,
        confidence_score=confidence_score,
        neighbors=neighbors,
        purpose=purpose,
        area_regime_label=area_regime_label,
        target=target,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        type_column=type_column,
        reference_area_column=reference_area_column,
        logo_path=Path(__file__).resolve().parent / "static" / "vera_header.png",
    )


def step_header(number: int, title: str) -> None:
    st.markdown(
        f"""
        <div class="step">
            <span class="step-number">{number}</span>
            <span class="step-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def purpose_suggests_territorial(purpose: str) -> bool:
    normalized = normalize_text(purpose)
    return any(
        term in normalized
        for term in (
            "terreno",
            "gleba",
            "lote",
            "sitio",
            "fazenda",
            "area rural",
            "chacara",
        )
    )


def choose_existing(columns: list[str], candidates: list[str]) -> str | None:
    return first_existing(columns, candidates)


def build_mapping(df: pd.DataFrame) -> tuple[ColumnMapping | None, list[str]]:
    columns = [str(column) for column in df.columns]

    tipo = choose_existing(
        columns,
        [
            DERIVED_TIPO_INFORMACAO,
            "tipo_informacao",
            "tipo_informação",
            "tipo de informação",
            "tipo da informação",
            "tipo da informacao",
        ],
    )
    finalidade = choose_existing(
        columns,
        [
            DERIVED_FINALIDADE_CRAWLER_NORMALIZADA,
            "finalidade_crawler_normalizada",
            "finalidade_crawler",
            "crawler_finalidade",
            "siat_finalidade_descricao",
        ],
    )
    valor = choose_existing(
        columns,
        [
            "valor_oferta",
            "valor_oferta_rs",
            "valor",
            "valor_total",
            "preco",
            "preço",
            "valor do imóvel",
            "valor do imovel",
            "preço do imóvel",
            "preco do imovel",
            "valor_unitario",
            "valor_unitário",
            "preco_venda",
            "preço de venda",
            "preco_anuncio",
            "valor_anuncio",
            "price",
            "price_brl",
        ],
    )
    latitude = choose_existing(
        columns,
        [
            "__latitude_geocodificada",
            "siat_latitude",
            "latitude",
            "lat",
        ],
    )
    longitude = choose_existing(
        columns,
        [
            "__longitude_geocodificada",
            "siat_longitude",
            "longitude",
            "lon",
            "lng",
        ],
    )

    area_lote = (
        DERIVED_AREA_LOTE
        if DERIVED_AREA_LOTE in columns
        else choose_existing(
            columns,
            [
                "siat_area_total_lote",
                "siat_area_terreno",
                "crawler_area_terreno",
                "area_total_lote",
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
    )
    area_construida = (
        DERIVED_AREA_CONSTRUIDA
        if DERIVED_AREA_CONSTRUIDA in columns
        else choose_existing(
            columns,
            [
                "area_construida",
                "area_construida_descrita_m2",
                "crawler_area_construida",
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
        )
    )
    area_privativa = (
        DERIVED_AREA_PRIVATIVA
        if DERIVED_AREA_PRIVATIVA in columns
        else choose_existing(
            columns,
            [
                "area_privativa",
                "area_privativa_descrita_m2",
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
    )
    testada = (
        DERIVED_TESTADA
        if DERIVED_TESTADA in columns
        else choose_existing(
            columns,
            [
                "testada",
                "testada_terreno",
                "siat_testada_terreno",
                "anuncio_testada",
            ],
        )
    )
    ano_construcao = choose_existing(
        columns,
        [
            "siat_ano",
            "ano_construcao",
            "ano_da_construcao",
            "ano_construção",
        ],
    )

    required = {
        "tipo de informação": tipo,
        "finalidade": finalidade,
        "valor": valor,
    }
    missing = [label for label, column in required.items() if not column]
    if missing:
        return None, missing

    mapping = ColumnMapping(
        tipo_informacao=tipo,
        finalidade_oferta=finalidade,
        valor=valor,
        area_construida=area_construida,
        area_privativa=area_privativa,
        latitude=latitude,
        longitude=longitude,
        siat_area_total_lote=area_lote,
        testada=testada,
        ano_construcao=ano_construcao,
    )
    return mapping, []


def detect_value_kind(value_column: str) -> str:
    normalized = normalize_text(value_column)
    if "unitario" in normalized or "unitário" in normalized:
        return "Valor unitário por m²"
    return "Valor total"


def usable_prepared_count(
    preparation: PreparationResult,
    mapping: ColumnMapping,
    territorial: bool,
    use_construction_year: bool = False,
    use_land_area: bool = False,
) -> int:
    """
    Conta candidatos utilizáveis pelo regime.

    O ano da construção só restringe a amostra quando estiver
    efetivamente sendo utilizado como característica do KNN.
    """
    valid = pd.Series(True, index=preparation.data.index)

    if use_land_area:
        if (
            not mapping.siat_area_total_lote
            or mapping.siat_area_total_lote not in preparation.data.columns
        ):
            return 0
        land_area = pd.to_numeric(
            preparation.data[mapping.siat_area_total_lote],
            errors="coerce",
        )
        valid &= np.isfinite(land_area) & land_area.gt(0)

    if territorial or not use_construction_year:
        return int(valid.sum())

    if (
        not mapping.ano_construcao
        or mapping.ano_construcao not in preparation.data.columns
    ):
        return 0

    valid &= valid_construction_year_mask(
        preparation.data[mapping.ano_construcao]
    )
    return int(valid.sum())


def classify_discount_alert(
    applied_discount: float,
    raw_discount: float | None,
    discount_source: str,
) -> dict[str, str]:
    """
    Classifica somente a comunicação do desconto, sem alterar o cálculo.
    """
    applied = float(applied_discount)
    raw = (
        float(raw_discount)
        if raw_discount is not None and np.isfinite(float(raw_discount))
        else np.nan
    )

    if discount_source == "offers_only_fallback":
        return {
            "level": "info",
            "title": "Fator convencional aplicado",
            "message": (
                "A amostra contém somente ofertas. Foi aplicado desconto de "
                f"{percent_br(applied)}. O teto de 20% não foi acionado, pois "
                "ele funciona exclusivamente como freio da razão empírica entre "
                "Guias ITBI e ofertas."
            ),
            "band": "10% convencional — somente ofertas",
        }

    if discount_source == "no_offers":
        return {
            "level": "success",
            "title": "Sem ofertas para ajustar",
            "message": (
                "A amostra efetiva não contém ofertas; nenhuma redução foi aplicada."
            ),
            "band": "sem ofertas",
        }

    if discount_source in {"insufficient_mixed_sample", "empirical_failure"}:
        return {
            "level": "warning",
            "title": "Razão empírica não calculada",
            "message": (
                "Existem Guias ITBI e ofertas, mas a composição não permite "
                "calcular adequadamente os quantis pareados. O desconto permaneceu "
                "em zero; o fator convencional de 10% é reservado à situação em "
                "que a amostra contém somente ofertas."
            ),
            "band": "amostra mista insuficiente",
        }

    if np.isfinite(raw) and raw > 0.20:
        return {
            "level": "error",
            "title": "Razão empírica acima do freio",
            "message": (
                f"A mediana calculada foi de {percent_br(raw)}. O freio de 20% "
                f"limitou o desconto aplicado a {percent_br(applied)}. Revise a "
                "homogeneidade entre Guias ITBI e ofertas."
            ),
            "band": "razão empírica acima de 20%",
        }

    if applied > 0.15:
        return {
            "level": "warning",
            "title": "Elasticidade elevada",
            "message": (
                f"O desconto empírico de {percent_br(applied)} está na faixa "
                "acima de 15% até 20%."
            ),
            "band": "acima de 15% até 20%",
        }

    if applied > 0.10:
        return {
            "level": "info",
            "title": "Elasticidade relevante",
            "message": (
                f"O desconto empírico de {percent_br(applied)} está na faixa "
                "acima de 10% até 15%."
            ),
            "band": "acima de 10% até 15%",
        }

    if applied > 0:
        return {
            "level": "success",
            "title": "Elasticidade usual ou moderada",
            "message": (
                f"O desconto empírico de {percent_br(applied)} está na faixa "
                "de até 10%."
            ),
            "band": "até 10%",
        }

    return {
        "level": "info",
        "title": "Sem desconto empírico positivo",
        "message": (
            "A razão entre Guias ITBI e ofertas não indicou superestimativa "
            "positiva das ofertas."
        ),
        "band": "0% empírico",
    }


def classify_sample_composition(
    n_itbi: int,
    n_offers: int,
    discount_source: str,
) -> dict[str, str]:
    """
    Classifica a suficiência operacional apenas para emissão de alertas.

    As faixas não bloqueiam o cálculo e não alteram o fator de oferta.
    """
    n_itbi = int(n_itbi)
    n_offers = int(n_offers)

    if discount_source == "offers_only_fallback":
        return {
            "level": "info",
            "title": "Amostra composta somente por ofertas",
            "message": (
                f"A amostra efetiva possui {n_offers} Oferta(s) e nenhuma Guia "
                "ITBI. Foi aplicado o fator convencional de 0,90, equivalente "
                "a desconto de 10%."
            ),
            "band": "somente ofertas — fator 0,90",
        }

    if n_offers == 0:
        return {
            "level": "success",
            "title": "Amostra sem ofertas",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guia(s) ITBI e nenhuma "
                "oferta. Não existe preço ofertado a ser ajustado."
            ),
            "band": "somente Guias ITBI",
        }

    minimum = min(n_itbi, n_offers)

    if n_itbi < 2 or n_offers < 2:
        return {
            "level": "error",
            "title": "Amostra mista insuficiente",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guia(s) ITBI e "
                f"{n_offers} Oferta(s). São necessários pelo menos dois dados "
                "de cada grupo para calcular a mediana por quantis pareados."
            ),
            "band": "amostra mista insuficiente",
        }

    if minimum < 10:
        return {
            "level": "warning",
            "title": "Amostra reduzida",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guias ITBI e "
                f"{n_offers} Ofertas. Um dos grupos tem menos de 10 dados, "
                "o que torna o desconto mais sensível à composição da amostra."
            ),
            "band": "menos de 10 em algum grupo",
        }

    if minimum < 15:
        return {
            "level": "info",
            "title": "Amostra restrita",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guias ITBI e "
                f"{n_offers} Ofertas. Há entre 10 e 14 dados no menor grupo; "
                "o resultado é utilizável, mas deve ser interpretado com cautela."
            ),
            "band": "10 a 14 no menor grupo",
        }

    imbalance = max(n_itbi, n_offers) / minimum
    if imbalance > 5:
        return {
            "level": "warning",
            "title": "Amostra numerosa, porém muito desequilibrada",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guias ITBI e "
                f"{n_offers} Ofertas. Ambos os grupos superam 15 dados, mas o "
                f"maior grupo é {imbalance:.1f} vezes o menor. Verifique se as "
                "distribuições permanecem comparáveis."
            ),
            "band": "15 ou mais, desequilíbrio superior a 5:1",
        }

    if imbalance > 3:
        return {
            "level": "info",
            "title": "Amostra suficiente, com desequilíbrio entre grupos",
            "message": (
                f"A amostra efetiva possui {n_itbi} Guias ITBI e "
                f"{n_offers} Ofertas. A quantidade mínima foi atendida, embora "
                f"o maior grupo seja {imbalance:.1f} vezes o menor."
            ),
            "band": "15 ou mais, desequilíbrio superior a 3:1",
        }

    return {
        "level": "success",
        "title": "Composição amostral adequada",
        "message": (
            f"A amostra efetiva possui {n_itbi} Guias ITBI e "
            f"{n_offers} Ofertas, com pelo menos 15 dados em cada grupo."
        ),
        "band": "15 ou mais em ambos os grupos",
    }


def render_quality_alert(alert: dict[str, str]) -> None:
    title = alert["title"]
    message = alert["message"]
    content = f"**{title}.** {message}"

    if alert["level"] == "error":
        st.error(content)
    elif alert["level"] == "warning":
        st.warning(content)
    elif alert["level"] == "success":
        st.success(content)
    else:
        st.info(content)


def risk_card(level: str, confidence: int, reasons: list[str]) -> None:
    css_level = {
        "baixo": "low",
        "moderado": "moderate",
        "alto": "high",
    }.get(level, "moderate")
    reason = reasons[0] if reasons else "Boa aderência aos dados observados."
    st.markdown(
        f"""
        <div class="risk-card {css_level}">
            <div class="risk-label">Confiabilidade</div>
            <div class="risk-value">{confidence}/100 · risco {level}</div>
            <div class="risk-text">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def unique_preserve_order(values: list[str | None]) -> list[str]:
    """Remove repetições preservando a primeira ocorrência e a ordem."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def make_unique_column_names(columns) -> list[str]:
    """Garante nomes únicos mesmo após renomeações ou cabeçalhos repetidos."""
    counts: dict[str, int] = {}
    result: list[str] = []

    for raw_name in columns:
        name = str(raw_name)
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        result.append(name if occurrence == 0 else f"{name}_{occurrence + 1}")

    return result



def first_named_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Retorna a primeira coluna com o nome solicitado.

    A proteção é necessária para planilhas que eventualmente contenham
    cabeçalhos repetidos.
    """
    matches = np.flatnonzero(np.asarray(df.columns, dtype=str) == str(column_name))
    if matches.size == 0:
        return pd.Series(index=df.index, dtype=float)
    return df.iloc[:, int(matches[0])]


def coordinate_to_numeric(series: pd.Series) -> pd.Series:
    """
    Converte coordenadas numéricas e textos com vírgula decimal.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


GEOCODED_LATITUDE = "__latitude_geocodificada"
GEOCODED_LONGITUDE = "__longitude_geocodificada"
GEOCODE_QUERY = "__endereco_geocodificacao"
GEOCODE_METHOD = "__metodo_geocodificacao"
GEOCODE_MATCHED_STREET = "__logradouro_correspondido"
GEOCODE_SIMILARITY_SCORE = "__score_similaridade_logradouro"
GEOCODE_CDLOG = "__cdlog_encontrado"
GEOCODE_FAILURE_REASON = "__motivo_falha_geocodificacao"


def detect_address_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Reconhece endereço completo ou seus componentes mais usuais."""
    columns = [str(column) for column in df.columns]
    return {
        "full": choose_existing(
            columns,
            [
                "endereco_completo",
                "endereço completo",
                "endereco",
                "endereço",
                "address",
                "localizacao",
                "localização",
            ],
        ),
        "street": choose_existing(
            columns,
            [
                "logradouro",
                "rua",
                "avenida",
                "street",
                "endereco_logradouro",
            ],
        ),
        "number": choose_existing(
            columns,
            ["numero", "número", "numero_endereco", "number"],
        ),
        "neighborhood": choose_existing(
            columns,
            [
                "bairro",
                "bairro_portal",
                "bairro_descricao",
                "neighborhood",
                "district",
            ],
        ),
        "city": choose_existing(
            columns,
            ["cidade", "municipio", "município", "city"],
        ),
        "state": choose_existing(
            columns,
            ["uf", "estado", "state"],
        ),
        "postal_code": choose_existing(
            columns,
            ["cep", "codigo_postal", "código postal", "postal_code"],
        ),
    }


def has_usable_address_columns(mapping: dict[str, str | None]) -> bool:
    return bool(mapping.get("full") or mapping.get("street"))


def address_component_values(
    df: pd.DataFrame,
    column: str,
    component: str,
) -> pd.Series:
    """Formata componentes sem produzir números de porta como ``136.0``."""
    values = df[column].astype("string").fillna("").str.strip()
    if component != "number":
        return values

    numeric_values = pd.to_numeric(values, errors="coerce")
    integer_mask = (
        numeric_values.notna()
        & np.isclose(
            numeric_values,
            numeric_values.round(),
            atol=1e-9,
        )
    )
    values.loc[integer_mask] = (
        numeric_values.loc[integer_mask]
        .round()
        .astype("Int64")
        .astype("string")
    )
    return values


def build_address_queries(
    df: pd.DataFrame,
    address_mapping: dict[str, str | None],
    locality_context: str = "",
) -> pd.Series:
    """Monta consultas de geocodificação sem alterar a planilha original."""
    full_column = address_mapping.get("full")
    if full_column:
        queries = df[full_column].astype("string").fillna("").str.strip()
        for key in ("number", "neighborhood", "city", "state", "postal_code"):
            column = address_mapping.get(key)
            if not column or column == full_column:
                continue
            values = address_component_values(df, column, key)
            combined: list[str] = []
            for query, value in zip(queries, values):
                if not value:
                    combined.append(query)
                    continue
                if key == "number" and parse_street_and_number(query)[1] is not None:
                    combined.append(query)
                    continue
                if value.casefold() in query.casefold():
                    combined.append(query)
                    continue
                combined.append(", ".join(part for part in (query, value) if part))
            queries = pd.Series(combined, index=df.index, dtype="string")
    else:
        queries = pd.Series("", index=df.index, dtype="string")
        for key in (
            "street",
            "number",
            "neighborhood",
            "city",
            "state",
            "postal_code",
        ):
            column = address_mapping.get(key)
            if not column:
                continue
            values = address_component_values(df, column, key)
            queries = pd.Series(
                [
                    ", ".join(part for part in (left, right) if part)
                    for left, right in zip(queries, values)
                ],
                index=df.index,
                dtype="string",
            )

    context = str(locality_context or "").strip()
    if context:
        queries = queries.map(
            lambda value: (
                f"{value}, {context}" if value and context.casefold() not in value.casefold()
                else value
            )
        )
    return queries.str.strip(" ,")


@st.cache_resource(show_spinner=False)
def cached_street_axis_index():
    return load_street_axis_index()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 30)
def geocode_queries(
    queries: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Usa primeiro o eixo de Porto Alegre e depois o fallback Nominatim."""
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim

    try:
        street_axis_index = cached_street_axis_index()
    except Exception:
        street_axis_index = None

    geocoder = Nominatim(
        user_agent="vera-estimador-knn-siri/1.18",
        timeout=10,
    )
    geocode = RateLimiter(
        geocoder.geocode,
        min_delay_seconds=1.05,
        max_retries=1,
        error_wait_seconds=2.0,
        swallow_exceptions=True,
    )
    result: dict[str, dict[str, object]] = {}
    for query in queries:
        street, number = parse_street_and_number(query)
        local_result = None
        normalized_query = normalize_text(query)
        query_parts = [part.strip() for part in query.split(",") if part.strip()]
        local_axis_applicable = (
            "porto alegre" in normalized_query or len(query_parts) <= 2
        )
        if (
            street_axis_index is not None
            and number is not None
            and local_axis_applicable
        ):
            local_result = geocode_porto_alegre_address(
                street_axis_index,
                street,
                number,
            )
        if local_result is not None and local_result.latitude is not None:
            result[query] = {
                "latitude": local_result.latitude,
                "longitude": local_result.longitude,
                "method": local_result.method,
                "matched_street": local_result.matched_street,
                "similarity_score": local_result.similarity_score,
                "cdlog": local_result.cdlog,
                "failure_reason": "",
            }
            continue

        location = geocode(query, exactly_one=True, language="pt-BR")
        if location is None:
            result[query] = {
                "latitude": None,
                "longitude": None,
                "method": "não localizado",
                "matched_street": (
                    local_result.matched_street if local_result else ""
                ),
                "similarity_score": (
                    local_result.similarity_score if local_result else 0.0
                ),
                "cdlog": local_result.cdlog if local_result else None,
                "failure_reason": (
                    local_result.failure_reason
                    if local_result
                    else "Endereço não localizado"
                ),
            }
        else:
            result[query] = {
                "latitude": float(location.latitude),
                "longitude": float(location.longitude),
                "method": "nominatim",
                "matched_street": (
                    local_result.matched_street if local_result else ""
                ),
                "similarity_score": (
                    local_result.similarity_score if local_result else 0.0
                ),
                "cdlog": local_result.cdlog if local_result else None,
                "failure_reason": "",
            }
    return result


def add_geocoded_coordinates(
    df: pd.DataFrame,
    queries: pd.Series,
    maximum_unique_queries: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Acrescenta coordenadas derivadas e métricas de cobertura."""
    data = df.copy()
    normalized_queries = queries.astype("string").fillna("").str.strip()
    unique_queries = list(dict.fromkeys(q for q in normalized_queries if q))
    selected_queries = tuple(unique_queries[: max(int(maximum_unique_queries), 1)])
    coordinates = geocode_queries(selected_queries)

    data[GEOCODE_QUERY] = normalized_queries
    def geocode_value(query: str, field: str, default: object = "") -> object:
        return coordinates.get(query, {}).get(field, default)

    data[GEOCODED_LATITUDE] = normalized_queries.map(
        lambda query: geocode_value(query, "latitude", None)
    )
    data[GEOCODED_LONGITUDE] = normalized_queries.map(
        lambda query: geocode_value(query, "longitude", None)
    )
    data[GEOCODE_METHOD] = normalized_queries.map(
        lambda query: geocode_value(query, "method")
    )
    data[GEOCODE_MATCHED_STREET] = normalized_queries.map(
        lambda query: geocode_value(query, "matched_street")
    )
    data[GEOCODE_SIMILARITY_SCORE] = normalized_queries.map(
        lambda query: geocode_value(query, "similarity_score", 0.0)
    )
    data[GEOCODE_CDLOG] = normalized_queries.map(
        lambda query: geocode_value(query, "cdlog", None)
    )
    data[GEOCODE_FAILURE_REASON] = normalized_queries.map(
        lambda query: geocode_value(query, "failure_reason")
    )
    geocoded_count = int(
        (
            data[GEOCODED_LATITUDE].notna()
            & data[GEOCODED_LONGITUDE].notna()
        ).sum()
    )
    method_counts = data[GEOCODE_METHOD].value_counts().to_dict()
    return data, {
        "rows": int(len(data)),
        "unique_queries": int(len(unique_queries)),
        "attempted_queries": int(len(selected_queries)),
        "geocoded_rows": geocoded_count,
        "axis_rows": int(method_counts.get("eixo_poa", 0)),
        "nominatim_rows": int(method_counts.get("nominatim", 0)),
        "failed_rows": int(method_counts.get("não localizado", 0)),
        "not_attempted_queries": int(
            max(len(unique_queries) - len(selected_queries), 0)
        ),
    }


def valid_coordinate_count(
    df: pd.DataFrame,
    latitude_column: str | None,
    longitude_column: str | None,
) -> int:
    if not latitude_column or not longitude_column:
        return 0
    latitude = coordinate_to_numeric(
        first_named_series(df, latitude_column)
    )
    longitude = coordinate_to_numeric(
        first_named_series(df, longitude_column)
    )
    return int(
        (
            latitude.between(-90, 90)
            & longitude.between(-180, 180)
        ).sum()
    )


def calculate_map_zoom(points: pd.DataFrame) -> float:
    """
    Estima um zoom legível a partir da dispersão espacial dos pontos.
    """
    if points.empty or len(points) == 1:
        return 15.0

    lat_span = float(points["latitude"].max() - points["latitude"].min())
    lon_span = float(points["longitude"].max() - points["longitude"].min())
    span = max(lat_span, lon_span)

    if span <= 0.002:
        return 15.5
    if span <= 0.005:
        return 14.5
    if span <= 0.010:
        return 13.8
    if span <= 0.030:
        return 12.5
    if span <= 0.080:
        return 11.3
    if span <= 0.200:
        return 10.2
    if span <= 0.500:
        return 9.0
    return 7.8


def build_map_data(
    neighbors: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    target_latitude: float,
    target_longitude: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Prepara pontos válidos para o mapa e informa quantos comparáveis foram
    descartados por coordenadas ausentes ou fora dos limites geográficos.
    """
    latitude = coordinate_to_numeric(
        first_named_series(neighbors, latitude_column)
    )
    longitude = coordinate_to_numeric(
        first_named_series(neighbors, longitude_column)
    )

    comparable_points = pd.DataFrame(
        {
            "latitude": latitude,
            "longitude": longitude,
        },
        index=neighbors.index,
    )

    if "_peso_knn" in neighbors.columns:
        comparable_points["peso"] = pd.to_numeric(
            first_named_series(neighbors, "_peso_knn"),
            errors="coerce",
        )
    else:
        comparable_points["peso"] = np.nan

    if "_distancia_geografica_km" in neighbors.columns:
        comparable_points["distancia_km"] = pd.to_numeric(
            first_named_series(neighbors, "_distancia_geografica_km"),
            errors="coerce",
        )
    else:
        comparable_points["distancia_km"] = np.nan

    comparable_points["tipo_ponto"] = "Comparável"
    comparable_points["ordem"] = np.arange(1, len(comparable_points) + 1)
    comparable_points["peso_texto"] = comparable_points["peso"].map(
        lambda value: (
            f"{value * 100:.2f}%"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["distancia_texto"] = comparable_points[
        "distancia_km"
    ].map(
        lambda value: (
            f"{value:.3f} km"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )

    valid_comparable = (
        comparable_points["latitude"].between(-90, 90)
        & comparable_points["longitude"].between(-180, 180)
        & comparable_points["latitude"].notna()
        & comparable_points["longitude"].notna()
    )
    discarded = int((~valid_comparable).sum())
    comparable_points = comparable_points.loc[valid_comparable].copy()

    target_points = pd.DataFrame(
        [
            {
                "latitude": pd.to_numeric(
                    pd.Series([target_latitude]), errors="coerce"
                ).iloc[0],
                "longitude": pd.to_numeric(
                    pd.Series([target_longitude]), errors="coerce"
                ).iloc[0],
                "tipo_ponto": "Imóvel avaliando",
                "ordem": 0,
                "peso": np.nan,
                "peso_texto": "—",
                "distancia_km": 0.0,
                "distancia_texto": "0,000 km",
            }
        ]
    )

    valid_target = (
        target_points["latitude"].between(-90, 90)
        & target_points["longitude"].between(-180, 180)
        & target_points["latitude"].notna()
        & target_points["longitude"].notna()
    )
    target_points = target_points.loc[valid_target].copy()

    return comparable_points, target_points, discarded


def render_comparables_map(
    neighbors: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    target_latitude: float,
    target_longitude: float,
    type_column: str | None = None,
    value_column: str | None = None,
    reference_area_column: str | None = None,
    testada_column: str | None = None,
) -> None:
    """
    Exibe avaliando e comparáveis em mapa Plotly/MapLibre interativo.
    """
    comparable_points, target_points, discarded = build_map_data(
        neighbors=neighbors,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )

    if target_points.empty:
        st.warning(
            "O mapa não pôde ser exibido porque as coordenadas do imóvel "
            "avaliando são inválidas."
        )
        return

    if discarded:
        st.warning(
            f"{discarded} comparável(is) não foi(ram) incluído(s) no mapa "
            "porque possuíam latitude ou longitude inválida."
        )

    if comparable_points.empty:
        st.warning(
            "Não existem comparáveis com coordenadas válidas para representar."
        )
        return

    comparable_points = comparable_points.copy()

    def values_from_neighbors(
        column_name: str | None,
        *,
        numeric: bool = False,
    ) -> pd.Series:
        if not column_name or column_name not in neighbors.columns:
            return pd.Series(
                [np.nan if numeric else "—"] * len(comparable_points),
                index=comparable_points.index,
            )

        source = first_named_series(neighbors, column_name).reset_index(drop=True)
        positions = (
            pd.to_numeric(comparable_points["ordem"], errors="coerce")
            .fillna(1)
            .astype(int)
            .sub(1)
            .clip(lower=0, upper=max(len(source) - 1, 0))
        )
        selected = source.iloc[positions.to_numpy()].reset_index(drop=True)
        selected.index = comparable_points.index

        if numeric:
            return pd.to_numeric(selected, errors="coerce")
        return selected.astype("string").fillna("—")

    comparable_points["tipo_informacao"] = values_from_neighbors(type_column)
    comparable_points["valor_total_original"] = values_from_neighbors(
        value_column,
        numeric=True,
    )
    comparable_points["area_referencia"] = values_from_neighbors(
        reference_area_column,
        numeric=True,
    )
    comparable_points["testada"] = values_from_neighbors(
        testada_column,
        numeric=True,
    )
    comparable_points["valor_unitario_original"] = values_from_neighbors(
        "_valor_unitario_original",
        numeric=True,
    )
    comparable_points["valor_unitario_ajustado"] = values_from_neighbors(
        "_valor_unitario_ajustado",
        numeric=True,
    )
    comparable_points["valor_unitario_robusto"] = values_from_neighbors(
        "_valor_unitario_robusto",
        numeric=True,
    )
    comparable_points["linha_excel"] = values_from_neighbors(
        "_row_excel",
        numeric=True,
    )

    comparable_points = comparable_points.sort_values(
        "peso",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    comparable_points["posicao_no_mapa"] = np.arange(
        1,
        len(comparable_points) + 1,
    )

    weights = pd.to_numeric(
        comparable_points["peso"],
        errors="coerce",
    ).fillna(0.0)

    if float(weights.max()) > float(weights.min()):
        marker_sizes = 15 + 16 * (
            (weights - weights.min()) / (weights.max() - weights.min())
        )
    else:
        marker_sizes = pd.Series(
            np.full(len(comparable_points), 21.0),
            index=comparable_points.index,
        )

    comparable_points["peso_formatado"] = weights.map(
        lambda value: f"{value * 100:.2f}%".replace(".", ",")
    )
    comparable_points["distancia_formatada"] = comparable_points[
        "distancia_km"
    ].map(
        lambda value: (
            f"{value:.3f} km".replace(".", ",")
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["vu_ajustado_formatado"] = comparable_points[
        "valor_unitario_ajustado"
    ].map(
        lambda value: (
            money_br(value) + "/m²"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["vu_original_formatado"] = comparable_points[
        "valor_unitario_original"
    ].map(
        lambda value: (
            money_br(value) + "/m²"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["area_formatada"] = comparable_points[
        "area_referencia"
    ].map(
        lambda value: (
            number_br(value) + " m²"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["testada_formatada"] = comparable_points["testada"].map(
        lambda value: (
            number_br(value) + " m"
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )
    comparable_points["linha_excel_formatada"] = comparable_points[
        "linha_excel"
    ].map(
        lambda value: (
            str(int(value))
            if pd.notna(value) and np.isfinite(value)
            else "—"
        )
    )

    all_points = pd.concat(
        [
            target_points[["latitude", "longitude"]],
            comparable_points[["latitude", "longitude"]],
        ],
        ignore_index=True,
    )

    center_latitude = float(all_points["latitude"].median())
    center_longitude = float(all_points["longitude"].median())
    zoom = calculate_map_zoom(all_points)

    style_options = {
        "Ruas — OpenStreetMap": "open-street-map",
        "Claro — Carto Positron": "carto-positron",
        "Sem mapa-base": "white-bg",
    }
    style_label = st.radio(
        "Mapa-base",
        list(style_options),
        horizontal=True,
        key="estimador_knn_siri_map_style",
        help=(
            "Use 'Sem mapa-base' caso a rede ou o navegador bloqueie os "
            "tiles externos. A navegação e os pontos continuarão interativos."
        ),
    )
    map_style = style_options[style_label]

    line_lats: list[float | None] = []
    line_lons: list[float | None] = []
    target_lat = float(target_points.iloc[0]["latitude"])
    target_lon = float(target_points.iloc[0]["longitude"])

    for row in comparable_points.itertuples(index=False):
        line_lats.extend([target_lat, float(row.latitude), None])
        line_lons.extend([target_lon, float(row.longitude), None])

    figure = go.Figure()

    figure.add_trace(
        go.Scattermap(
            lat=line_lats,
            lon=line_lons,
            mode="lines",
            line={
                "width": 1.2,
                "color": "rgba(23, 59, 87, 0.28)",
            },
            hoverinfo="skip",
            showlegend=False,
            name="Ligações",
        )
    )

    comparable_customdata = np.column_stack(
        [
            comparable_points["posicao_no_mapa"].astype(str),
            comparable_points["tipo_informacao"].astype(str),
            comparable_points["peso_formatado"].astype(str),
            comparable_points["distancia_formatada"].astype(str),
            comparable_points["vu_ajustado_formatado"].astype(str),
            comparable_points["vu_original_formatado"].astype(str),
            comparable_points["area_formatada"].astype(str),
            comparable_points["testada_formatada"].astype(str),
            comparable_points["linha_excel_formatada"].astype(str),
            comparable_points["latitude"].map(lambda x: f"{x:.7f}"),
            comparable_points["longitude"].map(lambda x: f"{x:.7f}"),
        ]
    )

    figure.add_trace(
        go.Scattermap(
            lat=comparable_points["latitude"],
            lon=comparable_points["longitude"],
            mode="markers+text",
            text=comparable_points["posicao_no_mapa"].astype(str),
            textposition="top center",
            textfont={
                "size": 13,
                "color": "#172033",
            },
            marker={
                "size": marker_sizes,
                "color": "#0E7C7B",
                "opacity": 0.86,
                "symbol": "circle",
                "allowoverlap": True,
            },
            customdata=comparable_customdata,
            hovertemplate=(
                "<b>Comparável %{customdata[0]}</b><br>"
                "Tipo: %{customdata[1]}<br>"
                "Peso no KNN: %{customdata[2]}<br>"
                "Distância: %{customdata[3]}<br>"
                "VU ajustado: %{customdata[4]}<br>"
                "VU original: %{customdata[5]}<br>"
                "Área de referência: %{customdata[6]}<br>"
                "Testada: %{customdata[7]}<br>"
                "Linha do Excel: %{customdata[8]}<br>"
                "Latitude: %{customdata[9]}<br>"
                "Longitude: %{customdata[10]}"
                "<extra></extra>"
            ),
            name="Comparáveis",
        )
    )

    figure.add_trace(
        go.Scattermap(
            lat=[target_lat],
            lon=[target_lon],
            mode="markers+text",
            text=["A"],
            textposition="middle center",
            textfont={
                "size": 13,
                "color": "white",
            },
            marker={
                "size": 29,
                "color": "#B42318",
                "opacity": 0.96,
                "symbol": "circle",
                "allowoverlap": True,
            },
            customdata=np.array(
                [[f"{target_lat:.7f}", f"{target_lon:.7f}"]]
            ),
            hovertemplate=(
                "<b>Imóvel avaliando</b><br>"
                "Latitude: %{customdata[0]}<br>"
                "Longitude: %{customdata[1]}"
                "<extra></extra>"
            ),
            name="Imóvel avaliando",
        )
    )

    figure.update_layout(
        height=590,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        map={
            "style": map_style,
            "center": {
                "lat": center_latitude,
                "lon": center_longitude,
            },
            "zoom": zoom,
        },
        dragmode="pan",
        hovermode="closest",
        uirevision=f"knn-siri-{style_label}",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.88)",
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
        key="estimador_knn_siri_plotly_map",
    )

    st.caption(
        "Arraste para navegar, use a roda do mouse para aproximar e passe o "
        "cursor sobre os pontos para consultar os dados. O tamanho do marcador "
        "representa o peso do comparável no KNN."
    )

    with st.expander("Identificação dos pontos do mapa"):
        map_display = comparable_points[
            [
                "posicao_no_mapa",
                "tipo_informacao",
                "peso",
                "distancia_km",
                "valor_unitario_ajustado",
                "area_referencia",
                "testada",
                "linha_excel",
                "latitude",
                "longitude",
            ]
        ].copy()
        map_display["peso_percentual"] = (
            pd.to_numeric(map_display["peso"], errors="coerce") * 100
        )
        map_display = map_display.drop(columns=["peso"])

        st.dataframe(
            map_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "posicao_no_mapa": "Ponto",
                "tipo_informacao": "Tipo",
                "peso_percentual": st.column_config.NumberColumn(
                    "Peso",
                    format="%.2f%%",
                ),
                "distancia_km": st.column_config.NumberColumn(
                    "Distância",
                    format="%.3f km",
                ),
                "valor_unitario_ajustado": st.column_config.NumberColumn(
                    "VU ajustado",
                    format="R$ %.2f",
                ),
                "area_referencia": st.column_config.NumberColumn(
                    "Área",
                    format="%.2f m²",
                ),
                "testada": st.column_config.NumberColumn(
                    "Testada",
                    format="%.2f m",
                ),
                "linha_excel": st.column_config.NumberColumn(
                    "Linha do Excel",
                    format="%d",
                ),
                "latitude": st.column_config.NumberColumn(format="%.7f"),
                "longitude": st.column_config.NumberColumn(format="%.7f"),
            },
        )

def dataframe_to_excel(
    neighbors: pd.DataFrame,
    diagnostics: dict,
    excluded_data: pd.DataFrame,
    flagged_data: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    diagnostics_df = pd.DataFrame(
        [
            {"indicador": key, "valor": str(value)}
            for key, value in diagnostics.items()
            if key not in {"feature_coge", "risk_reasons"}
        ]
    )

    excluded_export = excluded_data.copy()
    flagged_export = flagged_data.copy()
    excluded_export.columns = make_unique_column_names(
        excluded_export.columns
    )
    flagged_export.columns = make_unique_column_names(
        flagged_export.columns
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        neighbors.to_excel(writer, sheet_name="Comparaveis", index=False)
        diagnostics_df.to_excel(writer, sheet_name="Diagnosticos", index=False)
        excluded_export.to_excel(
            writer,
            sheet_name="Dados_excluidos",
            index=False,
        )
        flagged_export.to_excel(
            writer,
            sheet_name="Dados_alertados",
            index=False,
        )

        worksheet = writer.sheets["Comparaveis"]
        headers = {cell.value: cell.column for cell in worksheet[1]}

        for column_name in ("peso_knn",):
            column_number = headers.get(column_name)
            if column_number is None:
                continue
            for row_number in range(2, worksheet.max_row + 1):
                worksheet.cell(
                    row=row_number,
                    column=column_number,
                ).number_format = "0.00%"

        for column_name in (
            "valor_unitario_original",
            "valor_unitario_ajustado",
            "valor_unitario_robusto",
            "contribuicao_valor_unitario",
        ):
            column_number = headers.get(column_name)
            if column_number is None:
                continue
            for row_number in range(2, worksheet.max_row + 1):
                worksheet.cell(
                    row=row_number,
                    column=column_number,
                ).number_format = 'R$ #,##0.00'

        for sheet_name in ("Dados_excluidos", "Dados_alertados"):
            control_sheet = writer.sheets[sheet_name]
            control_headers = {
                cell.value: cell.column for cell in control_sheet[1]
            }
            for column_name in (
                "_valor_unitario_original",
                "_limite_inferior_vu_prefiltro",
                "_limite_superior_vu_prefiltro",
            ):
                column_number = control_headers.get(column_name)
                if column_number is None:
                    continue
                for row_number in range(2, control_sheet.max_row + 1):
                    control_sheet.cell(
                        row=row_number,
                        column=column_number,
                    ).number_format = 'R$ #,##0.00'

    output.seek(0)
    return output.getvalue()


hero_html = f"""
<div class="hero">
<span class="badge">VERA · {APP_EDITION}</span>

<img
src="app/static/vera_header.png"
alt="VERA — Valor Estimado por Referências Amostrais"
style="
display:block;
width:100%;
max-width:720px;
height:auto;
margin:0.8rem auto 0.4rem auto;
"
/>

<p>
Envie uma planilha do SIRI ou uma base de anúncios, informe as características do imóvel e obtenha uma estimativa com tratamento automático de ofertas, duplicidades, valores extremos e extrapolação.
</p>
</div>
"""

st.html(hero_html)

step_header(1, "Carregue a planilha")

with st.container(border=True):
    uploaded = st.file_uploader(
        "Arquivo Excel",
        type=["xlsx", "xlsm", "xls"],
        help="O arquivo é processado em memória durante esta sessão.",
    )

if uploaded is None:
    st.markdown(
        """
        <div class="helper">
            Use uma planilha exportada do SIRI ou uma base de anúncios com
            finalidade, valor e características de área. Coordenadas são
            opcionais quando houver endereço ou quando a localização for ignorada.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.sha1(file_bytes).hexdigest()

try:
    excel_file = pd.ExcelFile(BytesIO(file_bytes))
except Exception as exc:
    st.error(f"Não foi possível abrir o Excel: {exc}")
    st.stop()

if len(excel_file.sheet_names) == 1:
    sheet = excel_file.sheet_names[0]
else:
    sheet = st.selectbox("Planilha a utilizar", excel_file.sheet_names)

file_signature = f"{file_hash}:{sheet}"
if st.session_state.get("_lite_file_signature") != file_signature:
    st.session_state["_lite_file_signature"] = file_signature
    st.session_state.pop("lite_result", None)
    st.session_state.pop("_lite_last_context", None)

try:
    original_df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
except Exception as exc:
    st.error(f"Não foi possível ler a planilha: {exc}")
    st.stop()

if original_df.empty:
    st.error("A planilha selecionada está vazia.")
    st.stop()

original_df.columns = [str(column) for column in original_df.columns]
df, schema_info = enrich_known_schemas(original_df)
mapping, missing_fields = build_mapping(df)

if mapping is None:
    st.error(
        "A planilha não possui todos os campos necessários para a edição LITE."
    )
    st.write("Campos não identificados: " + ", ".join(missing_fields) + ".")
    st.caption(
        "Use cabeçalhos descritivos para finalidade, valor e área. Exemplos: "
        "Tipo, Preço (R$), Área construída (m²) e Área do terreno (m²)."
    )
    st.stop()

address_mapping = detect_address_columns(original_df)
sample_coordinate_count = valid_coordinate_count(
    df,
    mapping.latitude,
    mapping.longitude,
)
sample_geocoding_diagnostics: dict[str, int] = {}
geocoding_context = ""

if sample_coordinate_count < 2:
    st.warning(
        "A planilha não possui ao menos dois registros com coordenadas "
        "válidas. A VERA pode geocodificar endereços reconhecidos ou executar "
        "o KNN apenas com as características físicas."
    )
    location_options = ["Ignorar localização"]
    if has_usable_address_columns(address_mapping):
        location_options.insert(0, "Geocodificar endereços")

    sample_location_mode = st.radio(
        "Localização da amostra",
        location_options,
        horizontal=True,
        help=(
            "Em Porto Alegre, a VERA tenta primeiro a correspondência do "
            "logradouro e a interpolação do número no eixo viário oficial. "
            "Depois usa geocodificação online como fallback."
        ),
    )

    if sample_location_mode == "Geocodificar endereços":
        locality_context = st.text_input(
            "Complemento geográfico dos endereços (opcional)",
            placeholder="Ex.: Porto Alegre, RS, Brasil",
            help=(
                "Útil quando a planilha contém somente logradouro e número. "
                "Não repita cidade ou estado se eles já constarem nas colunas."
            ),
        )
        geocoding_context = locality_context
        address_queries = build_address_queries(
            original_df,
            address_mapping,
            locality_context,
        )
        unique_address_count = int(
            address_queries[address_queries.ne("")].nunique()
        )
        maximum_queries = st.number_input(
            "Máximo de endereços únicos a geocodificar",
            min_value=1,
            max_value=max(unique_address_count, 1),
            value=min(max(unique_address_count, 1), 100),
            step=1,
            help=(
                "A interpolação no eixo viário de Porto Alegre é local e rápida. "
                "Somente os fallbacks online respeitam intervalo entre "
                "requisições. Os resultados ficam em cache por 30 dias."
            ),
        )
        if unique_address_count:
            with st.spinner(
                "Geocodificando endereços. Em bases maiores, esta etapa pode demorar..."
            ):
                geocoded_df, sample_geocoding_diagnostics = (
                    add_geocoded_coordinates(
                        original_df,
                        address_queries,
                        int(maximum_queries),
                    )
                )
            if mapping.latitude and mapping.longitude:
                original_latitude = coordinate_to_numeric(
                    first_named_series(df, mapping.latitude)
                )
                original_longitude = coordinate_to_numeric(
                    first_named_series(df, mapping.longitude)
                )
                valid_original_location = (
                    original_latitude.between(-90, 90)
                    & original_longitude.between(-180, 180)
                )
                geocoded_df.loc[
                    valid_original_location,
                    GEOCODED_LATITUDE,
                ] = original_latitude.loc[valid_original_location]
                geocoded_df.loc[
                    valid_original_location,
                    GEOCODED_LONGITUDE,
                ] = original_longitude.loc[valid_original_location]
                geocoded_df.loc[
                    valid_original_location,
                    GEOCODE_METHOD,
                ] = "coordenadas da planilha"
                sample_geocoding_diagnostics["geocoded_rows"] = int(
                    (
                        geocoded_df[GEOCODED_LATITUDE].notna()
                        & geocoded_df[GEOCODED_LONGITUDE].notna()
                    ).sum()
                )
            final_method_counts = (
                geocoded_df[GEOCODE_METHOD].value_counts().to_dict()
            )
            sample_geocoding_diagnostics.update(
                {
                    "axis_rows": int(final_method_counts.get("eixo_poa", 0)),
                    "nominatim_rows": int(
                        final_method_counts.get("nominatim", 0)
                    ),
                    "original_coordinate_rows": int(
                        final_method_counts.get(
                            "coordenadas da planilha",
                            0,
                        )
                    ),
                    "failed_rows": int(
                        final_method_counts.get("não localizado", 0)
                    ),
                }
            )
            df, schema_info = enrich_known_schemas(geocoded_df)
            mapping, missing_fields = build_mapping(df)
            sample_coordinate_count = valid_coordinate_count(
                df,
                mapping.latitude,
                mapping.longitude,
            )
            st.caption(
                "Geocodificação: "
                f"**{sample_geocoding_diagnostics['geocoded_rows']}** de "
                f"**{sample_geocoding_diagnostics['rows']}** registros com "
                "coordenadas obtidas — "
                f"**{sample_geocoding_diagnostics['axis_rows']}** pelo eixo "
                "viário de Porto Alegre e "
                f"**{sample_geocoding_diagnostics['nominatim_rows']}** pelo "
                "fallback online e "
                f"**{sample_geocoding_diagnostics.get('original_coordinate_rows', 0)}** "
                "já existentes na planilha."
            )
            if sample_geocoding_diagnostics["not_attempted_queries"]:
                st.warning(
                    "O limite informado deixou "
                    f"{sample_geocoding_diagnostics['not_attempted_queries']} "
                    "endereço(s) único(s) sem tentativa de geocodificação."
                )
            failed_geocoding = geocoded_df.loc[
                geocoded_df[GEOCODED_LATITUDE].isna()
                | geocoded_df[GEOCODED_LONGITUDE].isna(),
                [
                    GEOCODE_QUERY,
                    GEOCODE_MATCHED_STREET,
                    GEOCODE_SIMILARITY_SCORE,
                    GEOCODE_CDLOG,
                    GEOCODE_FAILURE_REASON,
                ],
            ].copy()
            if not failed_geocoding.empty:
                with st.expander(
                    "Endereços não geocodificados",
                    expanded=False,
                ):
                    st.dataframe(
                        failed_geocoding,
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.warning("Nenhum endereço não vazio foi encontrado na planilha.")

    if sample_coordinate_count < 2:
        st.info(
            "A localização será ignorada: o peso físico será renormalizado para "
            "100% e nenhum dado será descartado apenas por falta de coordenadas."
        )
else:
    sample_location_mode = "Coordenadas da planilha"
    st.caption(
        f"Coordenadas válidas reconhecidas em **{sample_coordinate_count}** "
        "registro(s)."
    )

if schema_info.siri_detected:
    st.success(
        f"Planilha reconhecida: {len(df):,} registros.".replace(",", ".")
    )
else:
    st.info(
        "A estrutura principal foi reconhecida automaticamente, embora o "
        "arquivo não contenha todos os marcadores do padrão SIRI completo."
    )

if schema_info.notes:
    with st.expander("Hipóteses e regras do reconhecimento da planilha"):
        for schema_note in schema_info.notes:
            st.markdown(f"- {schema_note}")

siat_purpose_column = choose_existing(
    [str(column) for column in df.columns],
    ["siat_finalidade_descricao", "finalidade_oferta", "finalidade"],
)
finalidade_crawler_source_column = find_finalidade_crawler_column(
    [str(column) for column in original_df.columns]
)

normalized_purpose_series = (
    df[DERIVED_FINALIDADE_CRAWLER_NORMALIZADA]
    .dropna()
    .astype(str)
    .str.strip()
)
purposes = sorted(
    value
    for value in normalized_purpose_series.unique()
    if value
)
if not purposes:
    st.error(
        "Nenhum registro pôde ser enquadrado na taxonomia "
        "finalidade_crawler."
    )
    st.stop()

step_header(2, "Informe o imóvel")

selected_purpose = st.selectbox(
    "Finalidade da estimativa",
    purposes,
)
selected_use_nature = natureza_uso_normalizada(selected_purpose)
st.caption(
    f"Natureza de uso: **{selected_use_nature}**."
)
area_preference = reference_area_preference(selected_purpose)
territorial = area_preference == "terreno"

area_preference_labels = {
    "terreno": "área total do lote",
    "privativa": (
        "área privativa, com área total/construída como alternativa"
    ),
    "privativa_ou_construida": (
        "área privativa ou área total/construída"
    ),
    "construida": (
        "área total/construída, com área privativa como alternativa"
    ),
}
st.caption(
    "Todos os comparáveis serão selecionados pela finalidade crawler "
    f"normalizada **{selected_purpose}**. Área de referência prioritária: "
    f"**{area_preference_labels.get(area_preference, area_preference)}**."
)

selected_knn_parameters = calibrated_parameters_for_purpose(
    selected_purpose
)
st.caption(
    "Perfil KNN: "
    f"**{selected_knn_parameters['profile']}** · "
    f"K {int(selected_knn_parameters['min_k'])}–"
    f"{int(selected_knn_parameters['max_k'])} · "
    f"vizinhos efetivos mínimos "
    f"{float(selected_knn_parameters['min_effective_neighbors']):.0f} · "
    f"peso físico "
    f"{float(selected_knn_parameters['similarity_weight']):.0%} · "
    f"peso geográfico "
    f"{float(selected_knn_parameters['location_weight']):.0%}."
)

context_key = (
    f"{file_signature}:{selected_purpose}:{area_preference}"
)
if st.session_state.get("_lite_last_context") != context_key:
    st.session_state["_lite_last_context"] = context_key
    st.session_state.pop("lite_result", None)

st.info(
    "Tratamento automático: "
    + (
        "imóvel territorial."
        if territorial
        else "unidade construída."
    )
)

purpose_mask = (
    df[DERIVED_FINALIDADE_CRAWLER_NORMALIZADA]
    .map(normalize_text)
    .eq(normalize_text(selected_purpose))
)
purpose_types = df.loc[purpose_mask, mapping.tipo_informacao].map(normalize_text)

itbi_count = int(purpose_types.eq("guia itbi").sum())
sale_offer_count = int(purpose_types.eq("oferta").sum())
rental_count = int(
    purpose_types.str.contains(
        r"aluguel|locacao|arrendamento",
        regex=True,
        na=False,
    ).sum()
)
usable_count = itbi_count + sale_offer_count

s1, s2, s3, s4 = st.columns(4)
s1.metric("Dados utilizáveis", usable_count)
s2.metric("Guias ITBI", itbi_count)
s3.metric("Ofertas de venda", sale_offer_count)
s4.metric(
    "Aluguéis excluídos",
    rental_count,
    delta="não entram no KNN",
    delta_color="off",
)

st.caption(
    "Somente Guias ITBI e registros classificados como Oferta de venda "
    "seguem para o cálculo. Ofertas de aluguel são excluídas antes da "
    "deduplicação, do fator de oferta e da seleção dos comparáveis."
)

selected_rows_mask = purpose_mask
explicit_finality_count = int(
    (
        selected_rows_mask
        & df[DERIVED_FONTE_NORMALIZACAO].eq(
            "FINALIDADE_CRAWLER"
        )
    ).sum()
)
crawler_type_count = int(
    (
        selected_rows_mask
        & df[DERIVED_FONTE_NORMALIZACAO].eq("TIPO_CRAWLER")
    ).sum()
)
siat_count = int(
    (
        selected_rows_mask
        & df[DERIVED_FONTE_NORMALIZACAO].eq("SIAT")
    ).sum()
)
siat_fallback_count = int(
    (
        selected_rows_mask
        & df[DERIVED_FONTE_NORMALIZACAO].eq(
            "SIAT_FALLBACK"
        )
    ).sum()
)
typological_conflict_count = int(
    (
        selected_rows_mask
        & df[DERIVED_CONFLITO_TIPOLOGICO].isin(
            ["Sim", "Moderado"]
        )
    ).sum()
)
clean_usable_count = int(
    (
        selected_rows_mask
        & ~df[DERIVED_CONFLITO_TIPOLOGICO].isin(
            ["Sim", "Moderado"]
        )
        & df[mapping.tipo_informacao]
        .map(normalize_text)
        .isin(["guia itbi", "oferta"])
    ).sum()
)
st.caption(
    f"Finalidade da pesquisa: **{explicit_finality_count}** · "
    f"tipo do crawler: **{crawler_type_count}** · "
    f"SIAT: **{siat_count}** · "
    f"fallback ao SIAT: **{siat_fallback_count}** · "
    f"sem conflito: **{clean_usable_count}** · "
    f"com conflito: **{typological_conflict_count}**."
)
st.caption(
    "Registros com conflito tipológico são usados somente quando a "
    "amostra sem conflito não atinge o K inicial configurado."
)

if territorial:
    if not mapping.siat_area_total_lote:
        st.error("A planilha não possui uma área de terreno utilizável.")
        st.stop()
    if not mapping.testada:
        st.error("A planilha não possui uma coluna de TESTADA utilizável.")
        st.stop()
else:
    if not mapping.area_privativa and not mapping.area_construida:
        st.error(
            "A planilha não possui área privativa nem área construída utilizável."
        )
        st.stop()

    if not mapping.ano_construcao:
        st.warning(
            "A planilha não possui informação sobre o ano da construção. "
            "A estimativa será realizada sem essa característica."
        )

with st.form("lite_property_form"):
    with st.container(border=True):
        if territorial:
            c1, c2 = st.columns(2)
            with c1:
                target_area_lote = st.number_input(
                    "Área total do lote (m²)",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                )
            with c2:
                target_testada = st.number_input(
                    "Testada (m)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    help="Comprimento da frente principal do terreno.",
                )
            target_area_privativa = 0.0
            target_area_construida = 0.0
            target_ano_construcao = None
            use_land_area = True
        else:
            available_area_modes = ["Automática"]
            if mapping.area_privativa:
                available_area_modes.append("Área privativa")
            if mapping.area_construida:
                available_area_modes.append("Área total/construída")

            requested_area_mode = st.radio(
                "Base de área",
                available_area_modes,
                horizontal=True,
                help=(
                    "A estimativa utiliza um único denominador. No modo "
                    "automático, a base preferencial só é mantida quando "
                    "atinge o K inicial após todos os filtros."
                ),
            )

            c1, c2 = st.columns(2)
            with c1:
                target_area_privativa = (
                    st.number_input(
                        (
                            "Área privativa (m²) — referência principal"
                            if area_preference == "privativa"
                            else "Área privativa (m²)"
                        ),
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        help=(
                            "Para apartamentos, salas e lojas, esta costuma "
                            "ser a principal área de referência."
                        ),
                    )
                    if mapping.area_privativa
                    else 0.0
                )
            with c2:
                target_area_construida = (
                    st.number_input(
                        (
                            (
                                "Área total/construída (m²) — "
                                "referência principal"
                            )
                            if area_preference == "construida"
                            else "Área total/construída (m²)"
                        ),
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                    )
                    if mapping.area_construida
                    else 0.0
                )
            if mapping.ano_construcao:
                target_ano_construcao = st.number_input(
                    "Ano da construção (opcional)",
                    min_value=MIN_CONSTRUCTION_YEAR,
                    max_value=MAX_CONSTRUCTION_YEAR,
                    value=None,
                    step=1,
                    format="%d",
                    placeholder="Ex.: 1998",
                    help=(
                        "Quando informado, o ano da construção também é utilizado "
                        "na comparação física entre os imóveis. Se deixado em branco, "
                        "a estimativa será realizada sem essa característica."
                    ),
                )
            else:
                target_ano_construcao = None
                st.caption(
                    "Ano da construção não disponível nesta planilha. "
                    "Essa característica não será utilizada no KNN."
                )
            if mapping.siat_area_total_lote:
                land_area_values = pd.to_numeric(
                    df[mapping.siat_area_total_lote],
                    errors="coerce",
                )
                land_comparable_mask = (
                    purpose_mask
                    & df[mapping.tipo_informacao]
                    .map(normalize_text)
                    .isin(["guia itbi", "oferta"])
                    & np.isfinite(land_area_values)
                    & land_area_values.gt(0)
                )
                land_comparable_count = int(land_comparable_mask.sum())
                minimum_land_sample = int(
                    selected_knn_parameters["min_k"]
                )
                land_feature_supported = land_comparable_count >= 2
                default_use_land = (
                    normalize_text(selected_purpose)
                    in {"casa / residencia", "galpao / deposito"}
                    and land_comparable_count >= minimum_land_sample
                )
                use_land_area = st.checkbox(
                    "Imóvel térreo: considerar também a área do terreno",
                    value=default_use_land,
                    disabled=not land_feature_supported,
                    help=(
                        "Ative quando o lote compõe parcela relevante do valor, "
                        "como em casas, lojas térreas e galpões. A área do terreno "
                        "entra como característica física adicional, mas não muda "
                        "o denominador usado para calcular o valor unitário. São "
                        "necessários ao menos dois comparáveis com essa área."
                    ),
                )
                st.caption(
                    "Cobertura da área do terreno neste segmento: "
                    f"**{land_comparable_count}** comparável(is) válido(s); "
                    f"K inicial do perfil: **{minimum_land_sample}**."
                )
                if not land_feature_supported:
                    st.warning(
                        "A área do terreno não pode ser ativada nesta amostra: "
                        "há menos de dois comparáveis com valor preenchido. "
                        "Nenhum terreno será imputado."
                    )
                target_area_lote = (
                    st.number_input(
                        "Área do terreno (m²)",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                    )
                    if use_land_area
                    else 0.0
                )
            else:
                use_land_area = False
                target_area_lote = 0.0
                st.caption(
                    "Área do terreno não disponível na planilha; essa "
                    "característica física não poderá ser utilizada."
                )
            target_testada = 0.0

        if territorial:
            requested_area_mode = "Área total do lote"

        if sample_coordinate_count >= 2:
            target_location_options = [
                "Endereço",
                "Coordenadas",
                "Ignorar localização",
            ]
            target_location_mode = st.radio(
                "Localização do imóvel avaliando",
                target_location_options,
                horizontal=True,
            )
            if target_location_mode == "Coordenadas":
                c3, c4 = st.columns(2)
                with c3:
                    target_latitude = st.number_input(
                        "Latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=-30.0300000,
                        format="%.7f",
                    )
                with c4:
                    target_longitude = st.number_input(
                        "Longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=-51.2300000,
                        format="%.7f",
                    )
                target_address = ""
            elif target_location_mode == "Endereço":
                target_address = st.text_input(
                    "Endereço do imóvel avaliando",
                    placeholder="Ex.: Av. Borges de Medeiros, 2244, Porto Alegre, RS",
                )
                target_latitude = None
                target_longitude = None
            else:
                target_address = ""
                target_latitude = None
                target_longitude = None
                st.caption(
                    "O peso geográfico será zerado e o componente físico "
                    "será renormalizado para 100%."
                )
        else:
            target_location_mode = "Ignorar localização"
            target_address = ""
            target_latitude = None
            target_longitude = None
            st.caption(
                "Localização desativada: a comparação usará somente as "
                "características físicas disponíveis."
            )

        calculate = st.form_submit_button(
            "Calcular estimativa",
            type="primary",
            use_container_width=True,
        )

if calculate:
    try:
        if territorial:
            if target_area_lote <= 0:
                raise ValueError("Informe a área total do lote.")
            if target_testada <= 0:
                raise ValueError("Informe a testada do terreno.")

        if (
            not territorial
            and target_area_privativa <= 0
            and target_area_construida <= 0
        ):
            raise ValueError(
                "Informe a área privativa ou a área total/construída."
            )
        if not territorial and use_land_area and target_area_lote <= 0:
            raise ValueError(
                "Informe a área do terreno ou desative a opção de imóvel térreo."
            )

        if target_location_mode == "Endereço":
            target_query = str(target_address or "").strip()
            if geocoding_context and geocoding_context.casefold() not in target_query.casefold():
                target_query = f"{target_query}, {geocoding_context}".strip(" ,")
            if not target_query:
                raise ValueError("Informe o endereço do imóvel avaliando.")
            with st.spinner("Geocodificando o imóvel avaliando..."):
                target_geocode = geocode_queries((target_query,)).get(
                    target_query,
                    {},
                )
            target_latitude = target_geocode.get("latitude")
            target_longitude = target_geocode.get("longitude")
            target_geocode_method = str(
                target_geocode.get("method", "não localizado")
            )
            if target_latitude is None or target_longitude is None:
                raise ValueError(
                    "Não foi possível geocodificar o endereço do imóvel avaliando. "
                    "Revise-o ou informe as coordenadas manualmente."
                )
        elif target_location_mode == "Coordenadas":
            target_geocode_method = "coordenadas informadas"
        else:
            target_geocode_method = "localização ignorada"
        use_construction_year = (
            not territorial
            and mapping.ano_construcao is not None
            and target_ano_construcao is not None
        )

        original_columns = [str(column) for column in original_df.columns]
        duplicate_date_column = choose_existing(
            original_columns,
            [
                "data_encaminhamento",
                "data_registro",
                "data_coleta",
                "data_anuncio",
                "data_criacao_anuncio",
                "data_extracao",
                "data",
            ],
        )
        duplicate_registration_column = choose_existing(
            original_columns,
            ["siat_inscricao"],
        )
        duplicate_value_column = mapping.valor

        duplicate_identifier_columns = tuple(
            column
            for column in (
                "anuncio_website",
                "imobiliaria_codigo_anuncio",
                "idf_registro",
                "id_anuncio",
                "id_vivareal",
                "codigo_anuncio",
                "url_anuncio",
            )
            if column in original_columns
        )

        minimum_area_sample = int(
            selected_knn_parameters["min_k"]
        )
        preferred_regime = preferred_area_regime(area_preference)

        candidate_specs = {}
        if territorial:
            candidate_specs["terreno"] = {
                "column": mapping.siat_area_total_lote,
                "target_value": target_area_lote,
                "floor_purpose": selected_purpose,
                "floor_compatible": True,
                "label": "Área total do lote",
            }
        else:
            if (
                mapping.area_privativa
                and target_area_privativa > 0
            ):
                private_floor_compatible = area_floor_is_compatible(
                    area_preference,
                    AREA_REGIME_PRIVATE,
                )
                candidate_specs[AREA_REGIME_PRIVATE] = {
                    "column": mapping.area_privativa,
                    "target_value": target_area_privativa,
                    "floor_purpose": (
                        selected_purpose
                        if private_floor_compatible
                        else None
                    ),
                    "floor_compatible": private_floor_compatible,
                    "label": "Área privativa",
                }

            if (
                mapping.area_construida
                and target_area_construida > 0
            ):
                total_floor_compatible = area_floor_is_compatible(
                    area_preference,
                    AREA_REGIME_TOTAL_BUILT,
                )
                candidate_specs[AREA_REGIME_TOTAL_BUILT] = {
                    "column": mapping.area_construida,
                    "target_value": target_area_construida,
                    "floor_purpose": (
                        selected_purpose
                        if total_floor_compatible
                        else None
                    ),
                    "floor_compatible": total_floor_compatible,
                    "label": "Área total/construída",
                }

        preparation_candidates = {}
        preparation_errors = {}
        for regime_name, spec in candidate_specs.items():
            try:
                prepared_candidate = prepare_data(
                    df=df,
                    mapping=mapping,
                    selected_purpose=selected_purpose,
                    floor_purpose=spec["floor_purpose"],
                    value_kind=detect_value_kind(mapping.valor),
                    reference_area_column=spec["column"],
                    discount_cap=DISCOUNT_CAP,
                    remove_offer_duplicates=True,
                    duplicate_date_column=duplicate_date_column,
                    duplicate_identifier_columns=(
                        duplicate_identifier_columns
                    ),
                    duplicate_registration_column=(
                        duplicate_registration_column
                    ),
                    duplicate_value_column=duplicate_value_column,
                    conflict_column=DERIVED_CONFLITO_TIPOLOGICO,
                    minimum_without_conflict=minimum_area_sample,
                )
                preparation_candidates[regime_name] = (
                    prepared_candidate
                )
            except Exception as exc:
                preparation_errors[regime_name] = str(exc)

        if territorial:
            if "terreno" not in preparation_candidates:
                raise ValueError(
                    preparation_errors.get(
                        "terreno",
                        "Não foi possível preparar a amostra territorial.",
                    )
                )
            selected_area_regime = "terreno"
            area_selection = {
                "reason": "regime territorial obrigatório",
                "automatic": True,
                "sample_count": usable_prepared_count(
                    preparation_candidates["terreno"],
                    mapping,
                    territorial=True,
                ),
                "sample_sufficient": (
                    usable_prepared_count(
                        preparation_candidates["terreno"],
                        mapping,
                        territorial=True,
                    )
                    >= minimum_area_sample
                ),
                "private_count": 0,
                "total_built_count": 0,
            }
        else:
            private_count = (
                usable_prepared_count(
                    preparation_candidates[AREA_REGIME_PRIVATE],
                    mapping,
                    territorial=False,
                    use_construction_year=use_construction_year,
                    use_land_area=use_land_area,
                )
                if AREA_REGIME_PRIVATE in preparation_candidates
                else 0
            )
            
            total_built_count = (
                usable_prepared_count(
                    preparation_candidates[AREA_REGIME_TOTAL_BUILT],
                    mapping,
                    territorial=False,
                    use_construction_year=use_construction_year,
                    use_land_area=use_land_area,
                )
                if AREA_REGIME_TOTAL_BUILT in preparation_candidates
                else 0
            )

            area_selection = choose_area_regime(
                requested_mode=requested_area_mode,
                preferred_regime_value=preferred_regime,
                private_available=(
                    AREA_REGIME_PRIVATE
                    in preparation_candidates
                ),
                total_built_available=(
                    AREA_REGIME_TOTAL_BUILT
                    in preparation_candidates
                ),
                private_count=private_count,
                total_built_count=total_built_count,
                minimum_required=minimum_area_sample,
            )
            selected_area_regime = area_selection["regime"]

        preparation = preparation_candidates[
            selected_area_regime
        ]
        selected_area_spec = candidate_specs[
            selected_area_regime
        ]
        reference_area_column = selected_area_spec["column"]

        # A área do regime selecionado e, quando informados, o ano da
        # construção e a área do terreno compõem a distância física.
        target = {
            "area_construida": (
                target_area_construida
                if selected_area_regime
                == AREA_REGIME_TOTAL_BUILT
                else None
            ),
            "area_privativa": (
                target_area_privativa
                if selected_area_regime
                == AREA_REGIME_PRIVATE
                else None
            ),
            "siat_area_total_lote": (
                target_area_lote
                if territorial or use_land_area
                else None
            ),
            "testada": target_testada if territorial else None,
            "ano_construcao": (
                float(target_ano_construcao)
                if not territorial
                and target_ano_construcao is not None
                else None
            ),
            "latitude": target_latitude,
            "longitude": target_longitude,
        }
        target[reference_area_column] = selected_area_spec[
            "target_value"
        ]

        preparation.data[DERIVED_REGIME_AREA] = (
            selected_area_regime
        )
        preparation.diagnostics.update(
            {
                "area_mode_requested": requested_area_mode,
                "area_regime_selected": selected_area_regime,
                "area_regime_label": selected_area_spec["label"],
                "area_regime_reason": area_selection["reason"],
                "area_regime_automatic": area_selection["automatic"],
                "area_regime_sample_count": (
                    area_selection["sample_count"]
                ),
                "area_regime_sample_sufficient": (
                    area_selection["sample_sufficient"]
                ),
                "area_private_prepared_count": (
                    area_selection["private_count"]
                ),
                "area_total_built_prepared_count": (
                    area_selection["total_built_count"]
                ),
                "area_floor_compatible": (
                    selected_area_spec["floor_compatible"]
                ),
                "area_floor_applied": (
                    selected_area_spec["floor_purpose"]
                    is not None
                ),
                "area_floor_purpose": (
                    selected_area_spec["floor_purpose"] or ""
                ),
                "land_area_used_as_physical_feature": bool(
                    not territorial and use_land_area
                ),
                "land_area_target": (
                    float(target_area_lote)
                    if not territorial and use_land_area
                    else np.nan
                ),
                "area_preparation_errors": (
                    " | ".join(
                        f"{key}: {value}"
                        for key, value in sorted(
                            preparation_errors.items()
                        )
                    )
                ),
            }
        )

        rental_rows_after_filter = int(
            preparation.data["_tipo_norm"].str.contains(
                r"aluguel|locacao|arrendamento",
                regex=True,
                na=False,
            ).sum()
        )
        if rental_rows_after_filter:
            raise RuntimeError(
                "Falha de integridade: foram encontrados registros de aluguel "
                "após o filtro da amostra."
            )

        estimate = estimate_knn(
            preparation=preparation,
            mapping=mapping,
            target=target,
            reference_area_column=reference_area_column,
            min_k=int(selected_knn_parameters["min_k"]),
            max_k=int(selected_knn_parameters["max_k"]),
            min_effective_neighbors=float(
                selected_knn_parameters[
                    "min_effective_neighbors"
                ]
            ),
            similarity_weight=float(
                selected_knn_parameters["similarity_weight"]
            ),
            distance_power=float(
                selected_knn_parameters["distance_power"]
            ),
            max_individual_weight=float(
                selected_knn_parameters[
                    "max_individual_weight"
                ]
            ),
            robust_mad_threshold=float(
                selected_knn_parameters[
                    "robust_mad_threshold"
                ]
            ),
            territorial=territorial,
        )

        st.session_state["lite_result"] = {
            "preparation": preparation,
            "estimate": estimate,
            "mapping": mapping,
            "purpose": selected_purpose,
            "finalidade_crawler_normalizada": selected_purpose,
            "market_segment": selected_purpose,
            "territorial": territorial,
            "reference_area_column": reference_area_column,
            "area_regime": selected_area_regime,
            "area_regime_label": selected_area_spec["label"],
            "area_regime_reason": area_selection["reason"],
            "area_mode_requested": requested_area_mode,
            "area_floor_compatible": (
                selected_area_spec["floor_compatible"]
            ),
            "area_floor_applied": (
                selected_area_spec["floor_purpose"] is not None
            ),
            "land_area_used": bool(not territorial and use_land_area),
            "schema_notes": tuple(schema_info.notes),
            "sample_location_mode": sample_location_mode,
            "sample_geocoding_diagnostics": dict(
                sample_geocoding_diagnostics
            ),
            "target_location_mode": target_location_mode,
            "target_geocode_method": target_geocode_method,
            "area_regime_counts": {
                "privativa": area_selection["private_count"],
                "total_construida": (
                    area_selection["total_built_count"]
                ),
            },
            "target": target,
            "duplicate_date_column": duplicate_date_column,
            "knn_parameters": dict(selected_knn_parameters),
        }
    except Exception as exc:
        st.error(str(exc))

run = st.session_state.get("lite_result")
if run is None:
    st.markdown(
        """
        <div class="helper">
            O aplicativo escolherá automaticamente os comparáveis e reduzirá
            a influência de anúncios repetidos, imóveis isolados e valores
            unitários extremos.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

preparation: PreparationResult = run["preparation"]
estimate: EstimateResult = run["estimate"]
mapping = run["mapping"]

all_excluded_data = pd.concat(
    [
        preparation.excluded_data,
        estimate.local_excluded_data,
    ],
    ignore_index=True,
    sort=False,
)
all_excluded_data.columns = make_unique_column_names(
    all_excluded_data.columns
)

step_header(3, "Veja o resultado")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Valor total estimado", money_br(estimate.estimated_total_value))
m2.metric(
    "Valor unitário estimado",
    f"{money_br(estimate.estimated_unit_value)}/m²",
)
m3.metric(
    "Comparáveis utilizados",
    str(estimate.diagnostics["k_used"]),
    delta=f"{estimate.effective_neighbors:.1f} efetivos",
    delta_color="off",
)
m4.metric(
    "Desconto das ofertas",
    percent_br(preparation.discount),
    delta="limitado a 20%",
    delta_color="off",
)

area_counts = run.get("area_regime_counts", {})
st.info(
    f"Base de área selecionada: **{run['area_regime_label']}**. "
    f"Motivo: {run['area_regime_reason']}. "
    f"Amostras preparadas — privativa: "
    f"**{area_counts.get('privativa', 0)}**; "
    f"total/construída: "
    f"**{area_counts.get('total_construida', 0)}**."
)

if run.get("land_area_used"):
    st.info(
        "Imóvel térreo: a **área do terreno** foi incluída como característica "
        "física adicional do KNN. O valor unitário continua calculado pela base "
        f"selecionada ({run['area_regime_label'].lower()})."
    )

if estimate.diagnostics.get("location_used", False):
    st.caption(
        "Localização utilizada no KNN · peso geográfico efetivo: "
        f"**{estimate.diagnostics['location_weight']:.0%}** · origem do "
        "avaliando: "
        f"**{run.get('target_geocode_method', 'não informada')}**."
    )
else:
    st.warning(
        "A estimativa foi executada sem componente geográfico "
        f"({estimate.diagnostics.get('location_fallback_reason', 'indisponível')}). "
        "O peso físico foi renormalizado para 100%."
    )

construction_year_excluded = int(
    estimate.diagnostics.get(
        "construction_year_invalid_excluded",
        0,
    )
)
if construction_year_excluded:
    st.warning(
        f"Foram excluídos **{construction_year_excluded}** candidatos com "
        "ano da construção ausente ou inválido. Esses registros constam "
        "na aba de dados excluídos da exportação."
    )

location_invalid_excluded = int(
    estimate.diagnostics.get("location_invalid_excluded", 0)
)
if location_invalid_excluded:
    st.warning(
        f"Foram excluídos **{location_invalid_excluded}** candidatos sem "
        "coordenadas válidas porque a localização participou desta estimativa. "
        "Eles constam na aba de dados excluídos da exportação."
    )

if not run.get("area_floor_compatible", True):
    st.warning(
        "O piso absoluto por finalidade foi desativado nesta estimativa, "
        "pois ele não foi calibrado para a base alternativa de área. "
        "Os filtros relativos e robustos permanecem ativos."
    )
elif not preparation.diagnostics.get(
    "area_regime_sample_sufficient",
    True,
):
    st.warning(
        "O regime escolhido manualmente ficou abaixo do K inicial. "
        "A estimativa utilizará a amostra disponível e registrará essa "
        "limitação no diagnóstico."
    )

conflict_fallback_used = bool(
    preparation.diagnostics.get(
        "conflict_fallback_used",
        False,
    )
)
conflict_rows_available = int(
    preparation.diagnostics.get(
        "conflict_rows_available",
        0,
    )
)
conflict_rows_included = int(
    preparation.diagnostics.get(
        "conflict_rows_included",
        0,
    )
)
conflict_free_count = int(
    preparation.diagnostics.get(
        "conflict_free_prepared_count",
        0,
    )
)
conflict_minimum = int(
    preparation.diagnostics.get(
        "conflict_minimum_required",
        0,
    )
)

if conflict_fallback_used:
    st.warning(
        "A amostra sem conflito ficou com "
        f"**{conflict_free_count}** dados, abaixo do mínimo de "
        f"**{conflict_minimum}**. Foram admitidos "
        f"**{conflict_rows_included}** dados conflitantes como "
        "contingência."
    )
elif conflict_rows_available:
    st.info(
        f"Foram desconsiderados **{conflict_rows_available}** dados com "
        "conflito tipológico, pois a amostra sem conflito foi suficiente."
    )

raw_discount = preparation.diagnostics.get(
    "raw_discount_median",
    np.nan,
)
discount_source = preparation.diagnostics.get(
    "discount_source",
    "empirical_ratio",
)
n_itbi_effective = int(preparation.diagnostics.get("n_itbi", 0))
n_offer_effective = int(preparation.diagnostics.get("n_offer", 0))

discount_alert = classify_discount_alert(
    preparation.discount,
    raw_discount,
    discount_source,
)
sample_alert = classify_sample_composition(
    n_itbi_effective,
    n_offer_effective,
    discount_source,
)

with st.container(border=True):
    st.markdown("#### Qualidade do fator de oferta")
    qa1, qa2 = st.columns(2)
    with qa1:
        render_quality_alert(discount_alert)
    with qa2:
        render_quality_alert(sample_alert)
    st.caption(
        "Regra aplicada: somente ofertas recebem desconto convencional de 10%. "
        "Quando existem Guias ITBI e ofertas suficientes, utiliza-se a mediana "
        "dos quantis pareados, com 20% apenas como freio do resultado empírico."
    )

tabs = st.tabs(
    [
        "Resumo",
        (
            "Comparáveis e mapa"
            if estimate.diagnostics.get("location_used", False)
            else "Comparáveis"
        ),
    ]
)

with tabs[0]:
    left, right = st.columns([1.45, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Interpretação")
            dispersion = (
                estimate.weighted_std_unit / estimate.estimated_unit_value
                if estimate.estimated_unit_value > 0
                else np.nan
            )
            st.write(
                f"A estimativa foi formada por **{estimate.diagnostics['k_used']} "
                f"comparáveis**, equivalentes a **{estimate.effective_neighbors:.2f} "
                "vizinhos efetivos** após o controle de concentração."
            )
            st.write(
                f"A dispersão entre os comparáveis ficou em "
                f"**{percent_br(dispersion)}** do valor unitário estimado."
            )
            st.write(
                f"O maior peso individual foi de "
                f"**{percent_br(estimate.diagnostics['max_weight_observed'])}**."
            )
            st.progress(estimate.diagnostics["confidence_score"] / 100)
            st.caption(
                "A pontuação considera semelhança física, "
                + (
                    "localização, "
                    if estimate.diagnostics.get("location_used", False)
                    else "a indisponibilidade de localização, "
                )
                + "concentração dos pesos e necessidade de tratamento robusto."
            )

    with right:
        risk_card(
            estimate.diagnostics["risk_level"],
            estimate.diagnostics["confidence_score"],
            estimate.diagnostics["risk_reasons"],
        )

    risk_reasons = estimate.diagnostics["risk_reasons"]
    if risk_reasons:
        with st.expander("Pontos de atenção"):
            for reason in risk_reasons:
                st.warning(reason)

    prefilter_itbi_before = int(
        preparation.diagnostics.get("prefilter_itbi_before", 0)
    )
    prefilter_itbi_after = int(
        preparation.diagnostics.get("prefilter_itbi_after", 0)
    )
    prefilter_deterministic = int(
        preparation.diagnostics.get(
            "prefilter_deterministic_excluded",
            0,
        )
    )
    prefilter_statistical = int(
        preparation.diagnostics.get(
            "prefilter_statistical_excluded",
            0,
        )
    )
    prefilter_flagged = int(
        preparation.diagnostics.get("prefilter_flagged", 0)
    )
    prefilter_total_excluded = int(
        preparation.diagnostics.get("prefilter_total_excluded", 0)
    )
    local_low_excluded = int(
        estimate.diagnostics.get("local_low_filter_excluded", 0)
    )
    total_excluded = prefilter_total_excluded + local_low_excluded

    with st.container(border=True):
        st.markdown("#### Controle prévio da amostra")
        pf1, pf2, pf3, pf4 = st.columns(4)
        pf1.metric("Guias ITBI recebidas", prefilter_itbi_before)
        pf2.metric(
            "Dados excluídos",
            total_excluded,
            delta=(
                f"{prefilter_deterministic} cadastrais · "
                f"{prefilter_statistical} robustos · "
                f"{local_low_excluded} locais"
            ),
            delta_color="off",
        )
        pf3.metric("Dados em alerta", prefilter_flagged)
        pf4.metric("Guias ITBI utilizadas", prefilter_itbi_after)

        purpose_floor = float(
            preparation.diagnostics.get(
                "purpose_unit_value_floor",
                0.0,
            )
            or 0.0
        )
        purpose_floor_excluded = int(
            preparation.diagnostics.get(
                "purpose_floor_excluded",
                0,
            )
        )
        if purpose_floor > 0:
            st.caption(
                f"Piso da finalidade: {money_br(purpose_floor)}/m². "
                f"Registros excluídos por esse piso: "
                f"{purpose_floor_excluded}."
            )

        method = preparation.diagnostics.get(
            "prefilter_statistical_method",
            "não informado",
        )
        lower_vu = preparation.diagnostics.get(
            "prefilter_lower_vu",
            np.nan,
        )
        upper_vu = preparation.diagnostics.get(
            "prefilter_upper_vu",
            np.nan,
        )

        if np.isfinite(lower_vu) and np.isfinite(upper_vu):
            st.caption(
                f"Filtro estatístico: {method}. Faixa robusta automática "
                f"observada: {money_br(lower_vu)}/m² a "
                f"{money_br(upper_vu)}/m²."
            )
        else:
            st.caption(f"Filtro estatístico: {method}.")

        if total_excluded:
            st.warning(
                "Os registros excluídos não participaram da etapa em que "
                "foram rejeitados. O filtro local inferior ocorre antes da "
                "seleção final dos comparáveis."
            )
        else:
            st.success(
                "Nenhum registro precisou ser excluído pelo controle prévio."
            )

        if prefilter_flagged:
            st.info(
                "Os registros em alerta foram mantidos na amostra. Eles "
                "podem ser consultados e auditados abaixo."
            )

        local_lower_bound = estimate.diagnostics.get(
            "local_low_filter_lower_bound",
            np.nan,
        )
        local_reference_mode = estimate.diagnostics.get(
            "local_low_filter_reference_mode",
            "não aplicado",
        )
        local_profile = estimate.diagnostics.get(
            "local_low_filter_profile",
            "não informado",
        )
        local_cutoff_source = estimate.diagnostics.get(
            "local_low_filter_cutoff_source",
            "não informado",
        )
        local_median = estimate.diagnostics.get(
            "local_low_filter_weighted_median_vu",
            np.nan,
        )
        local_fraction = estimate.diagnostics.get(
            "local_low_filter_median_fraction",
            np.nan,
        )
        local_gap_detected = bool(
            estimate.diagnostics.get(
                "local_low_filter_gap_detected",
                False,
            )
        )
        local_gap_ratio = estimate.diagnostics.get(
            "local_low_filter_gap_ratio",
            np.nan,
        )
        local_gap_lower = estimate.diagnostics.get(
            "local_low_filter_gap_lower_value",
            np.nan,
        )
        local_gap_upper = estimate.diagnostics.get(
            "local_low_filter_gap_upper_value",
            np.nan,
        )
        local_cancelled_reason = estimate.diagnostics.get(
            "local_low_filter_cancelled_reason",
            "",
        )

        if np.isfinite(local_median):
            fraction_text = (
                f"{float(local_fraction):.0%}"
                if np.isfinite(local_fraction)
                else "não informada"
            )
            st.caption(
                f"Filtro local **{local_profile}**: "
                f"{local_reference_mode}. Mediana local ponderada: "
                f"{money_br(local_median)}/m²; fração de segurança: "
                f"{fraction_text}."
            )

        if local_gap_detected:
            st.info(
                "Foi identificada possível ruptura entre grupos de valor: "
                f"{money_br(local_gap_lower)}/m² → "
                f"{money_br(local_gap_upper)}/m² "
                f"(razão {float(local_gap_ratio):.2f})."
            )

        if np.isfinite(local_lower_bound):
            if int(local_low_excluded) > 0:
                st.caption(
                    f"Limite local automático: "
                    f"{money_br(local_lower_bound)}/m², definido por "
                    f"**{local_cutoff_source}**. Valores ajustados abaixo "
                    "desse limite foram rejeitados."
                )
            else:
                st.caption(
                    f"Limite local analisado: "
                    f"{money_br(local_lower_bound)}/m², definido por "
                    f"**{local_cutoff_source}**."
                )

        if local_cancelled_reason:
            st.warning(
                "O filtro local não realizou a exclusão: "
                f"{local_cancelled_reason}."
            )

        if (
            not all_excluded_data.empty
            or not preparation.flagged_data.empty
        ):
            with st.expander(
                "Consultar dados excluídos e alertados",
                expanded=False,
            ):
                if not all_excluded_data.empty:
                    st.markdown("##### Dados excluídos")
                    st.dataframe(
                        all_excluded_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "_valor_unitario_original": (
                                st.column_config.NumberColumn(
                                    "Valor unitário",
                                    format="R$ %.2f",
                                )
                            ),
                            "_escore_robusto_prefiltro": (
                                st.column_config.NumberColumn(
                                    "Escore robusto",
                                    format="%.3f",
                                )
                            ),
                            "_limite_inferior_vu_prefiltro": (
                                st.column_config.NumberColumn(
                                    "Limite inferior",
                                    format="R$ %.2f",
                                )
                            ),
                            "_limite_superior_vu_prefiltro": (
                                st.column_config.NumberColumn(
                                    "Limite superior",
                                    format="R$ %.2f",
                                )
                            ),
                        },
                    )

                if not preparation.flagged_data.empty:
                    st.markdown("##### Dados mantidos com alerta")
                    st.dataframe(
                        preparation.flagged_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "_valor_unitario_original": (
                                st.column_config.NumberColumn(
                                    "Valor unitário",
                                    format="R$ %.2f",
                                )
                            ),
                            "_escore_robusto_prefiltro": (
                                st.column_config.NumberColumn(
                                    "Escore robusto",
                                    format="%.3f",
                                )
                            ),
                        },
                    )

    duplicates_removed = int(
        preparation.diagnostics.get("offer_duplicates_removed", 0)
    )
    primary_duplicates = int(
        preparation.diagnostics.get(
            "market_duplicates_removed_primary",
            0,
        )
    )
    fallback_duplicates = int(
        preparation.diagnostics.get(
            "offer_duplicates_removed_fallback",
            0,
        )
    )
    st.info(
        f"Foram removidos **{duplicates_removed}** registros repetidos: "
        f"**{primary_duplicates}** pela chave prioritária "
        "`tipo + siat_inscricao + valor_oferta` e "
        f"**{fallback_duplicates}** por identificadores de anúncio."
    )

    discount_warning = preparation.diagnostics.get("discount_warning")
    if (
        discount_warning
        and discount_source not in {"offers_only_fallback", "empirical_ratio"}
    ):
        st.warning(discount_warning)

with tabs[1]:
    neighbors = estimate.neighbors.copy()

    relevant_columns = unique_preserve_order(
        [
            "_row_excel",
            mapping.tipo_informacao,
            finalidade_crawler_source_column,
            siat_purpose_column,
            "crawler_tipo_imovel_normalizado",
            DERIVED_FINALIDADE_CRAWLER_INFORMADA,
            DERIVED_FINALIDADE_SIAT_NORMALIZADA,
            DERIVED_FINALIDADE_TIPO_CRAWLER_NORMALIZADA,
            DERIVED_FINALIDADE_CRAWLER_NORMALIZADA,
            DERIVED_FONTE_NORMALIZACAO,
            DERIVED_CONFLITO_TIPOLOGICO,
            DERIVED_CONFIANCA_NORMALIZACAO,
            DERIVED_NATUREZA_USO_NORMALIZADA,
            DERIVED_REGIME_AREA,
            "_uso_conflito_tipologico",
            mapping.valor,
            run["reference_area_column"],
            mapping.area_construida,
            mapping.area_privativa,
            mapping.siat_area_total_lote,
            mapping.testada,
            mapping.ano_construcao,
            mapping.latitude,
            mapping.longitude,
            GEOCODE_QUERY,
            GEOCODE_METHOD,
            GEOCODE_MATCHED_STREET,
            GEOCODE_SIMILARITY_SCORE,
            GEOCODE_CDLOG,
            GEOCODE_FAILURE_REASON,
            "_valor_unitario_original",
            "_valor_unitario_ajustado",
            "_valor_unitario_robusto",
            "_distancia_geografica_km",
            "_peso_knn",
            "_contribuicao_valor_unitario",
        ]
    )
    relevant_columns = [
        column for column in relevant_columns if column in neighbors.columns
    ]

    rename = {
        "_row_excel": "linha_excel",
        "_valor_unitario_original": "valor_unitario_original",
        "_valor_unitario_ajustado": "valor_unitario_ajustado",
        "_valor_unitario_robusto": "valor_unitario_robusto",
        "_distancia_geografica_km": "distancia_geografica_km",
        "_peso_knn": "peso_knn",
        "_contribuicao_valor_unitario": "contribuicao_valor_unitario",
    }

    neighbors_export = (
        neighbors.loc[:, relevant_columns]
        .rename(columns=rename)
        .sort_values("peso_knn", ascending=False)
    )

    # Proteção adicional contra cabeçalhos repetidos no arquivo de origem
    # ou colisões produzidas após a renomeação.
    neighbors_export.columns = make_unique_column_names(
        neighbors_export.columns
    )

    neighbors_display = neighbors_export.copy()
    weight_column = next(
        (
            column
            for column in neighbors_display.columns
            if column == "peso_knn" or column.startswith("peso_knn_")
        ),
        None,
    )
    if weight_column is not None:
        neighbors_display["peso_percentual"] = (
            pd.to_numeric(
                neighbors_display[weight_column],
                errors="coerce",
            )
            * 100
        )
        neighbors_display = neighbors_display.drop(columns=[weight_column])

    st.dataframe(
        neighbors_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "peso_percentual": st.column_config.NumberColumn(
                "Peso",
                format="%.2f%%",
            ),
            "distancia_geografica_km": st.column_config.NumberColumn(
                "Distância",
                format="%.3f km",
            ),
            "valor_unitario_original": st.column_config.NumberColumn(
                format="R$ %.2f",
            ),
            "valor_unitario_ajustado": st.column_config.NumberColumn(
                format="R$ %.2f",
            ),
            "valor_unitario_robusto": st.column_config.NumberColumn(
                format="R$ %.2f",
            ),
            "contribuicao_valor_unitario": st.column_config.NumberColumn(
                format="R$ %.2f",
            ),
        },
    )

    if estimate.diagnostics.get("location_used", False):
        st.markdown("#### Localização dos comparáveis")
        render_comparables_map(
            neighbors=neighbors,
            latitude_column=mapping.latitude,
            longitude_column=mapping.longitude,
            target_latitude=run["target"]["latitude"],
            target_longitude=run["target"]["longitude"],
            type_column=mapping.tipo_informacao,
            value_column=mapping.valor,
            reference_area_column=run["reference_area_column"],
            testada_column=mapping.testada,
        )
    else:
        st.info(
            "Mapa indisponível nesta execução porque a localização não "
            "participou do cálculo."
        )

diagnostics = {
    "aplicativo": APP_NAME,
    "edicao": APP_EDITION,
    "finalidade_crawler_normalizada": (
        run["finalidade_crawler_normalizada"]
    ),
    "taxonomia_utilizada": "finalidade_crawler_normalizada",
    "perfil_parametros_knn": run["knn_parameters"]["profile"],
    "k_inicial_configurado": run["knn_parameters"]["min_k"],
    "k_maximo_configurado": run["knn_parameters"]["max_k"],
    "vizinhos_efetivos_minimos_configurados": (
        run["knn_parameters"]["min_effective_neighbors"]
    ),
    "peso_fisico_configurado": (
        run["knn_parameters"]["similarity_weight"]
    ),
    "peso_geografico_configurado": (
        run["knn_parameters"]["location_weight"]
    ),
    "peso_maximo_individual_configurado": (
        run["knn_parameters"]["max_individual_weight"]
    ),
    "potencia_distancia_configurada": (
        run["knn_parameters"]["distance_power"]
    ),
    "limiar_robusto_configurado": (
        run["knn_parameters"]["robust_mad_threshold"]
    ),
    "valor_total_estimado": estimate.estimated_total_value,
    "valor_unitario_estimado": estimate.estimated_unit_value,
    "cod_comparaveis": calculate_comparable_cod(estimate.neighbors),
    "definicao_cod_comparaveis": (
        "desvio absoluto médio dos valores unitários ajustados em torno da "
        "mediana; diagnóstico local, não COD de backtesting"
    ),
    "area_referencia": run["reference_area_column"],
    "ano_construcao_avaliando": (
        run["target"].get("ano_construcao")
    ),
    "area_terreno_como_atributo_fisico": run.get(
        "land_area_used",
        False,
    ),
    "area_terreno_avaliando": run["target"].get(
        "siat_area_total_lote"
    ),
    "modo_localizacao_amostra": run.get("sample_location_mode", ""),
    "modo_localizacao_avaliando": run.get("target_location_mode", ""),
    "metodo_geocodificacao_avaliando": run.get(
        "target_geocode_method",
        "",
    ),
    "hipoteses_reconhecimento_schema": " | ".join(
        run.get("schema_notes", ())
    ),
    "regime_area": run["area_regime"],
    "regime_area_rotulo": run["area_regime_label"],
    "modo_area_solicitado": run["area_mode_requested"],
    "motivo_selecao_regime_area": run["area_regime_reason"],
    "piso_compativel_com_regime_area": (
        run["area_floor_compatible"]
    ),
    "piso_aplicado_no_regime_area": run["area_floor_applied"],
    "comparaveis_regime_privativo": (
        run["area_regime_counts"].get("privativa", 0)
    ),
    "comparaveis_regime_total_construido": (
        run["area_regime_counts"].get(
            "total_construida",
            0,
        )
    ),
    "faixa_alerta_desconto": discount_alert["band"],
    "classificacao_desconto": discount_alert["title"],
    "faixa_composicao_amostral": sample_alert["band"],
    "classificacao_composicao_amostral": sample_alert["title"],
    "guias_itbi_efetivas": n_itbi_effective,
    "ofertas_efetivas": n_offer_effective,
    **preparation.diagnostics,
    **estimate.diagnostics,
}

excel_bytes = dataframe_to_excel(
    neighbors_export,
    diagnostics,
    all_excluded_data,
    preparation.flagged_data,
)

pdf_bytes: bytes | None = None
pdf_error: str | None = None
try:
    pdf_bytes = inference_report_pdf(
        estimated_unit_value=estimate.estimated_unit_value,
        estimated_total_value=estimate.estimated_total_value,
        confidence_score=estimate.diagnostics["confidence_score"],
        neighbors=estimate.neighbors,
        purpose=run["finalidade_crawler_normalizada"],
        area_regime_label=run["area_regime_label"],
        target=run["target"],
        latitude_column=(
            mapping.latitude
            if estimate.diagnostics.get("location_used", False)
            else None
        ),
        longitude_column=(
            mapping.longitude
            if estimate.diagnostics.get("location_used", False)
            else None
        ),
        type_column=mapping.tipo_informacao,
        reference_area_column=run["reference_area_column"],
    )
except Exception as exc:
    pdf_error = str(exc)

download_excel, download_pdf = st.columns(2)
with download_excel:
    st.download_button(
        "Baixar resultado em Excel",
        data=excel_bytes,
        file_name="resultado_estimador_knn_siri.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with download_pdf:
    if pdf_bytes is not None:
        st.download_button(
            "Baixar relatório de inferência em PDF",
            data=pdf_bytes,
            file_name="relatorio_inferencia_vera.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.button(
            "PDF indisponível nesta execução",
            disabled=True,
            use_container_width=True,
        )

if pdf_error:
    st.warning(
        "Não foi possível gerar o relatório PDF. O resultado em Excel continua "
        f"disponível. Detalhe: {pdf_error}"
    )

with st.expander("Como o estimador trabalha"):
    st.markdown(
        """
- considera apenas a finalidade escolhida;
- utiliza um único regime de área por estimativa: privativo ou total/construído;
- em imóveis prediais, usa a área selecionada e o ano da construção como atributos físicos;
- quando o imóvel é térreo, pode incluir a área do terreno como atributo físico adicional sem alterar o denominador do valor unitário;
- reconhece cabeçalhos usuais de planilhas SIRI e de bases de anúncios;
- pode geocodificar endereços ou renormalizar o peso físico para 100% quando a localização não estiver disponível;
- no modo automático, troca para a base alternativa somente quando a preferencial não atinge o K inicial;
- exclui ofertas de aluguel;
- normaliza todos os registros para uma finalidade crawler única e seleciona os comparáveis por essa taxonomia;
- escolhe automaticamente parâmetros KNN calibrados nos dados dos últimos três meses, com perfis específicos apenas quando validados;
- exclui dados com conflito tipológico enquanto a amostra limpa for suficiente e os admite somente como contingência;
- remove primeiro duplicidades por tipo, inscrição SIAT e valor, mantendo a coleta mais recente;
- usa identificadores genuínos do anúncio apenas como fallback;
- exclui valores inválidos ou simbólicos e transmissões não mercadológicas identificáveis;
- analisa o ln(valor unitário) das Guias ITBI por escore Z modificado;
- exclui automaticamente extremos robustos somente com 15 ou mais Guias ITBI;
- mantém apenas em alerta os extremos identificados em amostras de 8 a 14 Guias;
- aplica filtro local adaptativo com mediana ponderada pela proximidade, critérios robustos e detecção de ruptura entre grupos de valor;
- exporta os dados excluídos e alertados para auditoria;
- aplica fator 0,90 quando a amostra contém somente ofertas;
- havendo transações e ofertas suficientes, calcula a mediana da razão observada;
- usa 20% apenas como freio do desconto empírico;
- alerta quando o desconto é moderado, relevante, elevado ou supera o freio;
- informa se a quantidade de Guias ITBI e Ofertas é insuficiente, restrita ou adequada;
- escolhe automaticamente o número de comparáveis;
- impede que um único imóvel concentre peso excessivo;
- reduz a influência de valores unitários extremos;
- verifica se o imóvel está fora da faixa observada na base;
- apresenta o valor, a confiança, os comparáveis e, quando disponível, o mapa.
        """
    )
