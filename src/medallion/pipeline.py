"""Pipeline Bronze -> Silver -> Gold para os dados de alfabetização.

As fontes e entidades são as mesmas usadas no projeto da Fase 2. A Bronze
guarda o resultado literal das consultas BigQuery; a Silver padroniza e modela
as dimensões/fatos; e a Gold publica agregações analíticas para consumo.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"

QUERIES = {
    "alunos": """
        WITH d AS (SELECT nome_coluna, chave, valor
                   FROM `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario`
                   WHERE id_tabela = 'alunos')
        SELECT a.ano, CAST(a.id_municipio AS STRING) AS id_municipio, m.nome AS id_municipio_nome,
               CAST(a.id_escola AS STRING) AS id_escola, CAST(a.id_aluno AS STRING) AS id_aluno,
               a.caderno,
               s.valor AS serie, r.valor AS rede, p.valor AS presenca,
               pc.valor AS preenchimento_caderno, al.valor AS alfabetizado,
               a.proficiencia, a.peso_aluno
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.alunos` a
        LEFT JOIN (SELECT DISTINCT id_municipio, nome FROM `basedosdados.br_bd_diretorios_brasil.municipio`) m USING (id_municipio)
        LEFT JOIN d s ON a.serie = s.chave AND s.nome_coluna = 'serie'
        LEFT JOIN d r ON a.rede = r.chave AND r.nome_coluna = 'rede'
        LEFT JOIN d p ON a.presenca = p.chave AND p.nome_coluna = 'presenca'
        LEFT JOIN d pc ON a.preenchimento_caderno = pc.chave AND pc.nome_coluna = 'preenchimento_caderno'
        LEFT JOIN d al ON a.alfabetizado = al.chave AND al.nome_coluna = 'alfabetizado'
    """,
    "bolsa_familia_municipio": """
        SELECT ano_competencia, CAST(id_municipio AS STRING) AS id_municipio, sigla_uf,
               COUNT(1) AS total_beneficiarios, SUM(valor_parcela) AS valor_total_pago
        FROM `basedosdados.br_cgu_beneficios_cidadao.novo_bolsa_familia`
        WHERE ano_competencia BETWEEN 2022 AND 2024
        GROUP BY ano_competencia, id_municipio, sigla_uf
    """,
    "uf": """
        SELECT a.ano, a.sigla_uf, u.nome AS sigla_uf_nome, s.valor AS serie, r.valor AS rede,
               a.taxa_alfabetizacao, a.media_portugues
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.uf` a
        LEFT JOIN (SELECT DISTINCT sigla, nome FROM `basedosdados.br_bd_diretorios_brasil.uf`) u ON a.sigla_uf=u.sigla
        LEFT JOIN `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario` s
          ON a.serie=s.chave AND s.nome_coluna='serie' AND s.id_tabela='uf'
        LEFT JOIN `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario` r
          ON a.rede=r.chave AND r.nome_coluna='rede' AND r.id_tabela='uf'
    """,
    "municipio": """
        SELECT a.ano, CAST(a.id_municipio AS STRING) AS id_municipio, m.nome AS id_municipio_nome,
               s.valor AS serie, r.valor AS rede, a.taxa_alfabetizacao, a.media_portugues
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.municipio` a
        LEFT JOIN (SELECT DISTINCT id_municipio, nome FROM `basedosdados.br_bd_diretorios_brasil.municipio`) m USING (id_municipio)
        LEFT JOIN `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario` s
          ON a.serie=s.chave AND s.nome_coluna='serie' AND s.id_tabela='municipio'
        LEFT JOIN `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario` r
          ON a.rede=r.chave AND r.nome_coluna='rede' AND r.id_tabela='municipio'
    """,
    "meta_alfabetizacao_brasil": """
        SELECT ano, rede, taxa_alfabetizacao, meta_alfabetizacao_2024, meta_alfabetizacao_2025,
               meta_alfabetizacao_2026, meta_alfabetizacao_2027, meta_alfabetizacao_2028,
               meta_alfabetizacao_2029, meta_alfabetizacao_2030, percentual_participacao
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_brasil`
    """,
    "meta_alfabetizacao_uf": """
        SELECT a.ano, a.sigla_uf, u.nome AS sigla_uf_nome, a.rede, a.taxa_alfabetizacao,
               a.meta_alfabetizacao_2024, a.meta_alfabetizacao_2025, a.meta_alfabetizacao_2026,
               a.meta_alfabetizacao_2027, a.meta_alfabetizacao_2028, a.meta_alfabetizacao_2029,
               a.meta_alfabetizacao_2030, a.percentual_participacao
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_uf` a
        LEFT JOIN (SELECT DISTINCT sigla, nome FROM `basedosdados.br_bd_diretorios_brasil.uf`) u ON a.sigla_uf=u.sigla
    """,
    "meta_alfabetizacao_municipio": """
        SELECT a.ano, CAST(a.id_municipio AS STRING) AS id_municipio, m.nome AS id_municipio_nome,
               a.rede, a.taxa_alfabetizacao, a.meta_alfabetizacao_2024, a.meta_alfabetizacao_2025,
               a.meta_alfabetizacao_2026, a.meta_alfabetizacao_2027, a.meta_alfabetizacao_2028,
               a.meta_alfabetizacao_2029, a.meta_alfabetizacao_2030, a.nivel_alfabetizacao,
               a.percentual_participacao
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_municipio` a
        LEFT JOIN (SELECT DISTINCT id_municipio, nome FROM `basedosdados.br_bd_diretorios_brasil.municipio`) m USING (id_municipio)
    """,
}

REGIOES = {"AC":"Norte", "AL":"Nordeste", "AP":"Norte", "AM":"Norte", "BA":"Nordeste", "CE":"Nordeste", "DF":"Centro-Oeste", "ES":"Sudeste", "GO":"Centro-Oeste", "MA":"Nordeste", "MT":"Centro-Oeste", "MS":"Centro-Oeste", "MG":"Sudeste", "PA":"Norte", "PB":"Nordeste", "PR":"Sul", "PE":"Nordeste", "PI":"Nordeste", "RJ":"Sudeste", "RN":"Nordeste", "RS":"Sul", "RO":"Norte", "RR":"Norte", "SC":"Sul", "SP":"Sudeste", "SE":"Nordeste", "TO":"Norte"}


def _salvar_particionado(df: pd.DataFrame, base: Path, nome: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    coluna_ano = next((c for c in ("ano", "ano_competencia") if c in df), None)
    if coluna_ano is None:
        df.to_parquet(base / f"{nome}.parquet", index=False)
        return
    for ano, grupo in df.groupby(coluna_ano, dropna=False):
        destino = base / f"ano={ano}"
        destino.mkdir(parents=True, exist_ok=True)
        grupo.drop(columns=[coluna_ano]).to_parquet(destino / f"{nome}.parquet", index=False)


def _ler_particoes(base: Path) -> pd.DataFrame:
    arquivos = sorted(base.rglob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {base}")
    partes = []
    for arquivo in arquivos:
        df = pd.read_parquet(arquivo)
        for pai in arquivo.parents:
            if pai.name.startswith("ano="):
                coluna = "ano_competencia" if "ano_competencia" in arquivo.parts else "ano"
                if coluna not in df:
                    df.insert(0, coluna, int(pai.name.split("=", 1)[1]))
                break
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def baixar_bronze(tabelas: list[str] | None = None, somente_dry_run: bool = False) -> None:
    """Executa as consultas da Fase 2 no BigQuery e persiste a Bronze."""
    load_dotenv(ROOT / ".env")
    projeto = os.getenv("GCP_PROJECT_ID")
    if not projeto:
        raise ValueError("Defina GCP_PROJECT_ID no arquivo .env (veja .env.example).")
    from google.cloud import bigquery
    client = bigquery.Client(project=projeto)
    limite = int(os.getenv("BQ_MAX_BYTES_BILLED", "26000000000"))
    for nome in tabelas or list(QUERIES):
        if nome not in QUERIES:
            raise ValueError(f"Tabela inválida: {nome}. Opções: {', '.join(QUERIES)}")
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        estimativa = client.query(QUERIES[nome], job_config=job_config).total_bytes_processed or 0
        print(f"[DRY RUN] {nome}: {estimativa / 1024 / 1024:.1f} MB")
        if somente_dry_run:
            continue
        df = client.query(QUERIES[nome], job_config=bigquery.QueryJobConfig(maximum_bytes_billed=limite)).to_dataframe()
        df["entidade_origem"] = nome
        df["modo_ingestao"] = "batch"
        df["data_ingestao_bronze"] = datetime.now()
        _salvar_particionado(df, BRONZE / nome / "processado", nome)
        print(f"[BRONZE] {nome}: {len(df):,} linhas")


def processar_silver(data_processamento: date) -> None:
    """Modela as dimensões e fatos Silver equivalentes aos da Fase 2."""
    b = {nome: _ler_particoes(BRONZE / nome / "processado") for nome in QUERIES}
    alunos = b["alunos"].drop_duplicates().copy()
    for coluna in ("id_municipio", "id_escola", "id_aluno"):
        alunos[coluna] = alunos[coluna].astype("string").str.strip()
    alunos["flag_alfabetizado_preenchido"] = alunos["alfabetizado"].notna() & alunos["alfabetizado"].ne("")
    alunos["data_processamento_silver"] = data_processamento.isoformat()
    uf = b["uf"].copy(); uf["sigla_uf"] = uf["sigla_uf"].str.upper().str.strip()
    municipio = b["municipio"].copy(); municipio["id_municipio"] = municipio["id_municipio"].astype("string")
    execucao = f"execution_date={data_processamento.isoformat()}"
    tabelas = {
        "dominio_regiao_uf": pd.DataFrame(REGIOES.items(), columns=["sigla_uf", "regiao_brasil"]),
        "dim_uf": uf[["sigla_uf", "sigla_uf_nome"]].drop_duplicates().merge(pd.DataFrame(REGIOES.items(), columns=["sigla_uf", "regiao_brasil"]), on="sigla_uf", how="left"),
        "dim_municipio": municipio[["id_municipio", "id_municipio_nome"]].drop_duplicates(),
        "dim_escola": alunos[["id_escola", "id_municipio", "id_municipio_nome"]].drop_duplicates(),
        "fato_aluno_alfabetizacao": alunos,
        "fato_bolsa_familia_municipio": b["bolsa_familia_municipio"].drop_duplicates(),
        "fato_resultado_uf": uf[["ano", "sigla_uf", "serie", "rede", "taxa_alfabetizacao", "media_portugues"]].drop_duplicates(),
        "fato_resultado_municipio": municipio[["ano", "id_municipio", "serie", "rede", "taxa_alfabetizacao", "media_portugues"]].drop_duplicates(),
    }
    for nome, df in tabelas.items():
        df = df.copy(); df["data_processamento_silver"] = data_processamento.isoformat()
        _salvar_particionado(df, SILVER / nome / execucao, nome)
        print(f"[SILVER] {nome}: {len(df):,} linhas")


def processar_gold(data_processamento: date) -> None:
    """Publica visões Gold de desempenho e prioridade territorial."""
    execucao = f"execution_date={data_processamento.isoformat()}"
    alunos = _ler_particoes(SILVER / "fato_aluno_alfabetizacao" / execucao)
    municipio = _ler_particoes(SILVER / "fato_resultado_municipio" / execucao)
    alunos["alfabetizado_binario"] = (
        alunos["alfabetizado"].astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .isin(["sim", "1", "s"])
    )
    perfil = alunos.groupby(["ano", "rede"], dropna=False).agg(alunos_avaliados=("id_aluno", "nunique"), taxa_alfabetizacao=("alfabetizado_binario", "mean")).reset_index()
    perfil["taxa_alfabetizacao"] *= 100
    ranking = municipio.sort_values("taxa_alfabetizacao").copy()
    ranking["rank_prioridade"] = ranking.groupby(["ano", "rede"])["taxa_alfabetizacao"].rank(method="dense")
    for nome, df in {"perfil_aluno_alfabetizacao": perfil, "ranking_municipio_prioritario": ranking}.items():
        df["data_processamento_gold"] = data_processamento.isoformat()
        _salvar_particionado(df, GOLD / nome / execucao, nome)
        print(f"[GOLD] {nome}: {len(df):,} linhas")


def executar_camadas_medalhao(tabelas: list[str] | None = None, somente_dry_run: bool = False) -> None:
    """Orquestra Bronze -> Silver -> Gold; dry-run não grava dados."""
    if tabelas and not somente_dry_run and set(tabelas) != set(QUERIES):
        raise ValueError(
            "A materialização Silver/Gold exige todas as entidades. "
            "Use --tables apenas com --dry-run ou execute sem --tables."
        )
    baixar_bronze(tabelas=tabelas, somente_dry_run=somente_dry_run)
    if not somente_dry_run:
        hoje = date.today()
        processar_silver(hoje)
        processar_gold(hoje)
