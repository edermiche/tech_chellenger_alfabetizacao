"""Camada medalhão local alimentada pelas tabelas públicas do BigQuery."""

from src.medallion.pipeline import executar_camadas_medalhao

__all__ = ["executar_camadas_medalhao"]
