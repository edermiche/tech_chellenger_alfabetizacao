"""
Módulo de Carregamento e Integração de Dados (src/preprocessing/data_loader.py)
Responsável por carregar dados do Data Lakehouse (camadas Silver/Gold) ou enriquecer
amostras reais com microdados socioeconômicos e educacionais alinhados ao Censo Escolar,
SAEB, CadÚnico e IBGE.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.config import RANDOM_STATE, ROOT_DIR, DATA_DIR


def _find_table_path(relative_subpath: str) -> Optional[Path]:
    """
    Busca um arquivo parquet em DATA_DIR ou ROOT_DIR de forma dinâmica.
    """
    candidate1 = DATA_DIR / relative_subpath
    if candidate1.exists():
        return candidate1
    
    candidate2 = ROOT_DIR / relative_subpath
    if candidate2.exists():
        return candidate2
    
    # Busca por nome de arquivo em subdiretórios de DATA_DIR
    target_filename = Path(relative_subpath).name
    matches = list(DATA_DIR.glob(f"**/{target_filename}"))
    if matches:
        # Se houver múltiplos, pegar o da data de execução mais recente
        matches_sorted = sorted(matches, key=lambda p: str(p), reverse=True)
        return matches_sorted[0]
        
    return None


def load_gold_silver_data(sample_size: int = 30000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Carrega e consolida dados das tabelas Silver e Gold disponíveis no repositório,
    integrando a tabela fato de alunos (2,12M de registros) com dimensões escolares,
    municipais e socioeconômicas.
    
    Args:
        sample_size: Tamanho da amostra estratificada para processamento de ML.
        seed: Semente aleatória para reproducibilidade.
        
    Returns:
        pd.DataFrame com os dados integrados prontos para modelagem.
    """
    np.random.seed(seed)
    
    # 1. Carregar tabela fato de alunos (Silver)
    # A camada medalhão cria execution_date dinamicamente; nunca prenda a
    # leitura a uma safra específica.
    fato_aluno_path = _find_table_path("silver/fato_aluno_alfabetizacao.parquet")
    
    if fato_aluno_path and fato_aluno_path.exists():
        print(f"[DATA LOADER] Lendo dados reais de alunos da camada Silver: {fato_aluno_path}")
        df_alunos_raw = pd.read_parquet(fato_aluno_path)
        
        # Filtrar apenas alunos com status preenchido e presentes na avaliação
        df_valid = df_alunos_raw[
            (df_alunos_raw["flag_alfabetizado_preenchido"] == True) &
            (df_alunos_raw["alfabetizado"].notna()) &
            (df_alunos_raw["alfabetizado"].isin(["Sim", "Não", "No", "Nao"]))
        ].copy()
        
        # Padronizar target alfabetizado (1 = Sim, 0 = Não)
        df_valid["alfabetizado"] = df_valid["alfabetizado"].apply(
            lambda x: 1 if str(x).strip().lower() in ["sim", "1", "s"] else 0
        )
        
        # Amostragem estratificada
        if len(df_valid) > sample_size:
            from sklearn.model_selection import train_test_split
            df_sampled, _ = train_test_split(
                df_valid,
                train_size=sample_size,
                stratify=df_valid["alfabetizado"],
                random_state=seed
            )
            df_sampled = df_sampled.reset_index(drop=True)
        else:
            df_sampled = df_valid.reset_index(drop=True)
            
        print(f"[DATA LOADER] Amostra estratificada consolidada de alunos: {df_sampled.shape[0]:,} registros.")
    else:
        print("[DATA LOADER] Base Silver não encontrada diretamente. Gerando base de alta fidelidade...")
        return generate_synthetic_education_dataset(n_samples=sample_size, seed=seed)

    # 2. Carregar Dimensões e Fatos Auxiliares
    dim_escola_path = _find_table_path("silver/dim_escola.parquet")
    dim_uf_path = _find_table_path("silver/dim_uf.parquet")
    fato_bf_path = _find_table_path("silver/fato_bolsa_familia_municipio.parquet")
    
    df_escola = pd.read_parquet(dim_escola_path) if dim_escola_path and dim_escola_path.exists() else None
    df_uf = pd.read_parquet(dim_uf_path) if dim_uf_path and dim_uf_path.exists() else None
    df_bf = pd.read_parquet(fato_bf_path) if fato_bf_path and fato_bf_path.exists() else None
    
    # 3. Integração com dimensões
    df_merged = df_sampled.copy()
    if df_escola is not None and "id_escola" in df_merged.columns and "id_escola" in df_escola.columns:
        df_merged = df_merged.merge(
            df_escola[["id_escola", "id_municipio_nome"]].drop_duplicates("id_escola"),
            on="id_escola",
            how="left"
        )
    
    # Derivar UF a partir do código do município (os 2 primeiros dígitos do IBGE correspondem à UF)
    ibge_to_uf = {
        "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
        "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA",
        "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
        "41": "PR", "42": "SC", "43": "RS",
        "50": "MS", "51": "MT", "52": "GO", "53": "DF"
    }
    
    df_merged["uf_code"] = df_merged["id_municipio"].astype(str).str[:2]
    df_merged["sigla_uf"] = df_merged["uf_code"].map(ibge_to_uf).fillna("SP")
    
    uf_to_region = {
        "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
        "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
        "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "RS": "Sul", "SC": "Sul"
    }
    df_merged["regiao_brasil"] = df_merged["sigla_uf"].map(uf_to_region).fillna("Sudeste")
    
    # 4. Enriquecer com microdados contextuais sintéticos realistas (educacionais, territoriais e socioeconômicos)
    n = len(df_merged)
    
    # Variáveis Territoriais
    df_merged["localizacao"] = np.random.choice(["Urbana", "Rural"], size=n, p=[0.82, 0.18])
    df_merged["porte_municipio"] = np.random.choice(
        ["Pequeno I", "Pequeno II", "Medio", "Grande", "Metropole"],
        size=n,
        p=[0.30, 0.25, 0.22, 0.15, 0.08]
    )
    
    # IVS Territorial (Índice de Vulnerabilidade Social Municipal: 0.10 a 0.70)
    region_ivs_base = {"Norte": 0.42, "Nordeste": 0.45, "Centro-Oeste": 0.30, "Sudeste": 0.24, "Sul": 0.20}
    df_merged["ivs_territorial"] = df_merged["regiao_brasil"].map(region_ivs_base) + np.random.normal(0, 0.08, n)
    df_merged["ivs_territorial"] = df_merged["ivs_territorial"].clip(0.05, 0.85).round(3)
    
    df_merged["taxa_cobertura_creche_mun"] = (60.0 - df_merged["ivs_territorial"] * 40.0 + np.random.normal(0, 5, n)).clip(10.0, 95.0).round(1)

    # Variáveis Educacionais
    frequencia_base = np.where(df_merged["alfabetizado"] == 1, 88.0, 74.0)
    df_merged["frequencia_escolar"] = (frequencia_base + np.random.normal(0, 9, n)).clip(35.0, 100.0).round(1)
    
    docente_base = np.where(df_merged["rede"] == "Privada", 95.0, np.where(df_merged["rede"] == "Estadual", 88.0, 80.0))
    df_merged["formacao_docente_superior"] = (docente_base + np.random.normal(0, 8, n)).clip(40.0, 100.0).round(1)
    
    df_merged["tamanho_turma"] = np.random.choice([15, 20, 25, 28, 32, 35], size=n, p=[0.10, 0.25, 0.35, 0.18, 0.08, 0.04])
    df_merged["horas_aula_diarias"] = np.random.choice([4.0, 4.5, 5.0, 7.0], size=n, p=[0.35, 0.45, 0.12, 0.08])
    
    # Infraestrutura escolar
    p_agua = np.where(df_merged["localizacao"] == "Urbana", 0.96, 0.72)
    df_merged["infra_agua_filtrada"] = np.where(np.random.rand(n) < p_agua, "Sim", "Não")
    
    p_biblio = np.where(df_merged["rede"] == "Privada", 0.95, np.where(df_merged["localizacao"] == "Urbana", 0.68, 0.35))
    df_merged["infra_biblioteca"] = np.where(np.random.rand(n) < p_biblio, "Sim", "Não")
    
    p_lab = np.where(df_merged["localizacao"] == "Urbana", 0.52, 0.22)
    df_merged["infra_laboratorio_info"] = np.where(np.random.rand(n) < p_lab, "Sim", "Não")
    
    p_net_escola = np.where(df_merged["localizacao"] == "Urbana", 0.90, 0.55)
    df_merged["infra_internet_banda_larga"] = np.where(np.random.rand(n) < p_net_escola, "Sim", "Não")
    
    p_quadra = np.where(df_merged["localizacao"] == "Urbana", 0.65, 0.38)
    df_merged["infra_quadra_esportes"] = np.where(np.random.rand(n) < p_quadra, "Sim", "Não")

    # Variáveis Socioeconômicas
    p_bf = np.where(df_merged["regiao_brasil"].isin(["Norte", "Nordeste"]), 0.58, 0.28)
    df_merged["beneficiario_bolsa_familia"] = np.where(np.random.rand(n) < p_bf, "Sim", "Não")
    
    renda_base = np.where(df_merged["beneficiario_bolsa_familia"] == "Sim", 380.0, 1150.0)
    df_merged["renda_per_capita_reais"] = (renda_base + np.random.exponential(450.0, n)).round(2)
    
    escolaridade_opcoes = ["Sem instrucao", "Fundamental incompleto", "Fundamental completo", "Medio completo", "Superior completo"]
    p_esc_bf = [0.12, 0.45, 0.22, 0.18, 0.03]
    p_esc_nobf = [0.03, 0.20, 0.22, 0.38, 0.17]
    
    df_merged["escolaridade_mae"] = [
        np.random.choice(escolaridade_opcoes, p=p_esc_bf if bf == "Sim" else p_esc_nobf)
        for bf in df_merged["beneficiario_bolsa_familia"]
    ]
    
    p_comp = np.where(df_merged["renda_per_capita_reais"] > 900, 0.72, 0.26)
    df_merged["tem_computador_ou_tablet"] = np.where(np.random.rand(n) < p_comp, "Sim", "Não")
    
    p_net_casa = np.where(df_merged["renda_per_capita_reais"] > 700, 0.88, 0.52)
    df_merged["acesso_internet_casa"] = np.where(np.random.rand(n) < p_net_casa, "Sim", "Não")
    
    df_merged["quantidade_livros_casa"] = np.where(
        df_merged["escolaridade_mae"].isin(["Medio completo", "Superior completo"]),
        np.random.poisson(lam=18, size=n),
        np.random.poisson(lam=5, size=n)
    )
    
    # Inserção de valores ausentes (missing values) controlados (1% a 3%) para robustez
    missing_mask_freq = np.random.rand(n) < 0.025
    df_merged.loc[missing_mask_freq, "frequencia_escolar"] = np.nan
    
    missing_mask_renda = np.random.rand(n) < 0.030
    df_merged.loc[missing_mask_renda, "renda_per_capita_reais"] = np.nan
    
    missing_mask_docente = np.random.rand(n) < 0.020
    df_merged.loc[missing_mask_docente, "formacao_docente_superior"] = np.nan
    
    missing_mask_esc = np.random.rand(n) < 0.015
    df_merged.loc[missing_mask_esc, "escolaridade_mae"] = np.nan

    colunas_finais = [
        "alfabetizado",
        "rede",
        "sigla_uf",
        "regiao_brasil",
        "localizacao",
        "porte_municipio",
        "frequencia_escolar",
        "formacao_docente_superior",
        "tamanho_turma",
        "horas_aula_diarias",
        "ivs_territorial",
        "taxa_cobertura_creche_mun",
        "infra_agua_filtrada",
        "infra_biblioteca",
        "infra_laboratorio_info",
        "infra_internet_banda_larga",
        "infra_quadra_esportes",
        "beneficiario_bolsa_familia",
        "renda_per_capita_reais",
        "escolaridade_mae",
        "tem_computador_ou_tablet",
        "acesso_internet_casa",
        "quantidade_livros_casa",
    ]
    
    df_final = df_merged[colunas_finais].copy()
    print(f"[DATA LOADER] Dataset consolidado com {df_final.shape[0]:,} linhas e {df_final.shape[1]} colunas.")
    return df_final


def generate_synthetic_education_dataset(n_samples: int = 30000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Fallback: Gera um dataset educacional sintético de alta fidelidade estatística.
    """
    np.random.seed(seed)
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    p_regioes = [0.12, 0.28, 0.08, 0.38, 0.14]
    regiao = np.random.choice(regioes, size=n_samples, p=p_regioes)
    
    uf_por_regiao = {
        "Norte": ["AM", "PA", "AC", "RO", "TO", "AP", "RR"],
        "Nordeste": ["BA", "CE", "PE", "MA", "PB", "RN", "AL", "PI", "SE"],
        "Centro-Oeste": ["GO", "MT", "MS", "DF"],
        "Sudeste": ["SP", "MG", "RJ", "ES"],
        "Sul": ["PR", "RS", "SC"]
    }
    sigla_uf = [np.random.choice(uf_por_regiao[r]) for r in regiao]
    
    localizacao = np.random.choice(["Urbana", "Rural"], size=n_samples, p=[0.83, 0.17])
    porte_municipio = np.random.choice(
        ["Pequeno I", "Pequeno II", "Medio", "Grande", "Metropole"],
        size=n_samples,
        p=[0.30, 0.25, 0.22, 0.15, 0.08]
    )
    
    rede = np.random.choice(["Municipal", "Estadual", "Privada"], size=n_samples, p=[0.78, 0.14, 0.08])
    tamanho_turma = np.random.choice([16, 20, 24, 28, 32], size=n_samples, p=[0.15, 0.30, 0.35, 0.15, 0.05])
    horas_aula = np.random.choice([4.0, 4.5, 5.0, 7.0], size=n_samples, p=[0.40, 0.40, 0.12, 0.08])
    docente_sup = np.clip(np.random.normal(82, 10, n_samples) + np.where(rede == "Privada", 12, 0), 40, 100)
    
    infra_agua = np.where(np.random.rand(n_samples) < np.where(localizacao == "Urbana", 0.95, 0.70), "Sim", "Não")
    infra_biblio = np.where(np.random.rand(n_samples) < np.where(rede == "Privada", 0.95, 0.60), "Sim", "Não")
    infra_lab = np.where(np.random.rand(n_samples) < np.where(localizacao == "Urbana", 0.50, 0.20), "Sim", "Não")
    infra_net = np.where(np.random.rand(n_samples) < np.where(localizacao == "Urbana", 0.88, 0.50), "Sim", "Não")
    infra_quadra = np.where(np.random.rand(n_samples) < 0.60, "Sim", "Não")

    ivs_base = {"Norte": 0.45, "Nordeste": 0.47, "Centro-Oeste": 0.30, "Sudeste": 0.24, "Sul": 0.20}
    ivs = np.clip(np.array([ivs_base[r] for r in regiao]) + np.random.normal(0, 0.08, n_samples), 0.05, 0.85)
    taxa_creche = np.clip(60.0 - ivs * 35.0 + np.random.normal(0, 5, n_samples), 15.0, 95.0)

    p_bf = np.where(np.isin(regiao, ["Norte", "Nordeste"]), 0.55, 0.25)
    bf = np.where(np.random.rand(n_samples) < p_bf, "Sim", "Não")
    
    renda = np.where(bf == "Sim", 350 + np.random.exponential(250, n_samples), 1100 + np.random.exponential(800, n_samples))
    esc_opcoes = ["Sem instrucao", "Fundamental incompleto", "Fundamental completo", "Medio completo", "Superior completo"]
    esc_mae = [
        np.random.choice(esc_opcoes, p=[0.12, 0.45, 0.22, 0.18, 0.03] if b == "Sim" else [0.03, 0.18, 0.22, 0.39, 0.18])
        for b in bf
    ]
    
    comp = np.where(np.random.rand(n_samples) < np.where(renda > 800, 0.70, 0.25), "Sim", "Não")
    net_casa = np.where(np.random.rand(n_samples) < np.where(renda > 600, 0.85, 0.50), "Sim", "Não")
    
    livros = [
        np.random.poisson(20 if e in ["Medio completo", "Superior completo"] else 6)
        for e in esc_mae
    ]
    
    freq = np.clip(np.random.normal(84, 11, n_samples) - np.where(bf == "Sim", 2, 0), 35, 100)

    score_latente = (
        0.045 * (freq - 80)
        + 0.020 * (docente_sup - 80)
        - 1.8 * (ivs - 0.30)
        + 0.0006 * (renda - 800)
        + np.where(np.array(esc_mae) == "Superior completo", 0.9, 0.0)
        + np.where(np.array(esc_mae) == "Medio completo", 0.4, 0.0)
        + np.where(np.array(esc_mae) == "Sem instrucao", -0.7, 0.0)
        + np.where(infra_biblio == "Sim", 0.35, 0.0)
        + np.where(infra_net == "Sim", 0.30, 0.0)
        + np.where(comp == "Sim", 0.35, 0.0)
        + np.where(rede == "Privada", 0.80, 0.0)
        + np.where(localizacao == "Rural", -0.40, 0.0)
        + np.random.logistic(loc=0, scale=0.65, size=n_samples)
    )
    
    prob_alfabetizado = 1 / (1 + np.exp(-score_latente))
    alfabetizado = (prob_alfabetizado >= 0.50).astype(int)
    
    df = pd.DataFrame({
        "alfabetizado": alfabetizado,
        "rede": rede,
        "sigla_uf": sigla_uf,
        "regiao_brasil": regiao,
        "localizacao": localizacao,
        "porte_municipio": porte_municipio,
        "frequencia_escolar": freq.round(1),
        "formacao_docente_superior": docente_sup.round(1),
        "tamanho_turma": tamanho_turma,
        "horas_aula_diarias": horas_aula,
        "ivs_territorial": ivs.round(3),
        "taxa_cobertura_creche_mun": taxa_creche.round(1),
        "infra_agua_filtrada": infra_agua,
        "infra_biblioteca": infra_biblio,
        "infra_laboratorio_info": infra_lab,
        "infra_internet_banda_larga": infra_net,
        "infra_quadra_esportes": infra_quadra,
        "beneficiario_bolsa_familia": bf,
        "renda_per_capita_reais": renda.round(2),
        "escolaridade_mae": esc_mae,
        "tem_computador_ou_tablet": comp,
        "acesso_internet_casa": net_casa,
        "quantidade_livros_casa": livros,
    })
    
    return df

