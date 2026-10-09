"""Endereços e composição dos pesos compartilhados pelo PDF e pelo Excel."""
from __future__ import annotations

import pandas as pd


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def comparable_addresses(data: pd.DataFrame) -> pd.Series:
    """Prioriza endereço informado; preserva complemento e bairro cadastral."""
    def column(*candidates):
        return next((c for c in candidates if c in data.columns), None)
    full = column("endereco_completo", "endereço completo", "endereco", "endereço", "address")
    street = column("siat_logradouro", "logradouro", "rua", "avenida", "street", "endereco_logradouro")
    number = column("siat_numero", "numero", "número", "numero_endereco", "number")
    complement = column("siat_complemento", "complemento", "complement")
    neighborhood = column("siat_bairro", "bairro", "bairro_portal", "bairro_descricao", "neighborhood")
    result = []
    for _, row in data.iterrows():
        address = _text(row[full]) if full else ""
        if not address:
            parts = [_text(row[c]) for c in (street, number, complement, neighborhood) if c]
            address = ", ".join(p for p in parts if p)
        result.append(address or "Endereço não informado")
    return pd.Series(result, index=data.index, dtype="string")


def prepare_audit_comparables(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Conserva dados de origem e acrescenta identificadores e fatores legíveis."""
    data = neighbors.copy()
    data["endereco_completo"] = comparable_addresses(neighbors)
    # A coluna interna anterior continua no core para notebooks existentes.
    data = data.drop(columns=["_valor_unitario_ajustado", "_fator_ajuste"], errors="ignore")
    rename = {
        "_row_excel": "linha_excel", "_valor_unitario_original": "valor_unitario",
        "_valor_unitario_robusto": "valor_unitario_robusto", "_peso_knn": "peso_knn",
        "_distancia_geografica_km": "distancia_geografica_km",
        "_distancia_caracteristicas": "distancia_caracteristicas",
        "_distancia_composta": "distancia_composta", "_fator_temporal": "fator_recencia",
        "_fator_edificio": "fator_edificio", "_peso_distancia_bruto": "peso_distancia_bruto",
        "_peso_composto_bruto": "peso_composto_bruto",
        "_idade_observacao_dias": "idade_observacao_dias",
        "_contribuicao_valor_unitario": "contribuicao_valor_unitario",
    }
    return data.rename(columns=rename).sort_values("peso_knn", ascending=False)
