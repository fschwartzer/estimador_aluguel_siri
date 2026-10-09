"""Verifica a interface com uma amostra real e a exportação completa."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Upload(BytesIO):
    name = "pesquisa_validacao.xlsx"


if __name__ == "__main__":
    source = pd.read_pickle(ROOT/"tmp/source.pkl")
    data = source.loc[source.crawler_tipo_imovel_normalizado.eq("sala comercial") & source.siat_latitude.between(-90,90) & source.siat_longitude.between(-180,180)].tail(5000).copy()
    # A leitura real da pesquisa foi validada separadamente; evita reler 42 MB
    # a cada interação do teste da interface.
    with patch.object(st,"file_uploader",return_value=Upload(b"pesquisa_siri_validada")), patch.object(pd,"ExcelFile",return_value=SimpleNamespace(sheet_names=["Plan1"])), patch.object(pd,"read_excel",side_effect=lambda *a,**k:data.copy()):
        app = AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
        assert not app.exception, str(app.exception)
        print("selectbox",[(w.label,w.options,w.value) for w in app.selectbox])
        print("radio",[(w.label,w.options,w.value) for w in app.radio])
        print("numbers",[(w.label,w.value) for w in app.number_input])
        for widget in app.radio:
            if widget.label=="Localização do imóvel avaliando":
                widget.set_value("Coordenadas")
        app.run()
        print("numbers after mode",[(w.label,w.value) for w in app.number_input])
        for widget in app.number_input:
            if widget.label.startswith("Área privativa"):
                widget.set_value(50.)
            if widget.label.startswith("Latitude"):
                widget.set_value(-30.03)
            if widget.label.startswith("Longitude"):
                widget.set_value(-51.23)
        for widget in app.button:
            if "estimar" in widget.label.casefold() or "calcular" in widget.label.casefold():
                widget.click()
        with patch("urllib.request.urlopen",side_effect=OSError("mapa offline no teste")):
            app.run()
        print("exceptions",list(app.exception))
        print("errors",[w.value for w in app.error])
        print("warnings",[w.value for w in app.warning])
        print("downloads",len(app.get("download_button")))
        assert not app.exception, str(app.exception)
        assert not app.error, str(app.error)
        assert len(app.get("download_button"))==2
        first_result = app.session_state["lite_result"]["estimate"].estimated_total_value
        for widget in app.number_input:
            if widget.label.startswith("Área privativa"):
                widget.set_value(55.)
        for widget in app.button:
            if "estimar" in widget.label.casefold() or "calcular" in widget.label.casefold():
                widget.click()
        with patch("urllib.request.urlopen",side_effect=OSError("mapa offline no teste")):
            app.run()
        assert not app.exception and not app.error
        assert len(app.get("download_button"))==2
        assert app.session_state["lite_result"]["estimate"].estimated_total_value != first_result
        print("Fluxo completo aprovado")
