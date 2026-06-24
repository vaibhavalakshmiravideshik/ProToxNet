"""
ProToxNet | Step 1: Data Acquisition
=====================================
Downloads and harmonizes all data sources:
  - DrugCentral drug-target interactions (PostgreSQL)
  - DrugCentral FAERS pharmacovigilance signals
  - STRING v12 PPI network (human, combined_score >= 700)
  - GTEx/HPA tissue expression (hpa_tissue_expression.csv assumed present)
  - CT-ADE benchmark (Figshare)
  - BindingDB / ChEMBL kinase affinities

Outputs (all to DRIVE/):
  drugcentral_drug_target.csv
  drugcentral_faers.csv
  string_ppi.csv
  core_proteins.csv
  bindingdb_kinase.csv
  ctade_drug_ae.csv
  id_maps.pkl
"""

import gzip, pickle, subprocess, sys, re, time, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

DRIVE = Path("/content/drive/MyDrive/ProToxNet/data")
DRIVE.mkdir(parents=True, exist_ok=True)

# ── DrugCentral PostgreSQL connection params ──────────────────────────────────
DC_CONN = dict(host="unmtid-dbs.net", port=5433,
               dbname="drugcentral", user="drugman", password="dosage",
               connect_timeout=20)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DrugCentral drug-target interactions
# ─────────────────────────────────────────────────────────────────────────────
def fetch_drugcentral_targets():
    out = DRIVE / "drugcentral_drug_target.csv"
    gz  = DRIVE / "drug.target.interaction.tsv.gz"
    if not gz.exists():
        url = ("https://unmtid-dbs.net/download/DrugCentral/2021_09_01/"
               "drug.target.interaction.tsv.gz")
        print("Downloading DrugCentral DTI...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(gz, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    with gzip.open(gz, "rt") as f:
        dt = pd.read_csv(f, sep="\t", low_memory=False)
    dt = dt[dt["ORGANISM"].str.contains("Homo sapiens", na=False)]
    dt = dt.dropna(subset=["ACCESSION"]).rename(columns={"ACCESSION": "uniprot"})
    keep = [c for c in ["DRUG_NAME","uniprot","GENE","SWISSPROT","TARGET_NAME",
                        "TARGET_CLASS","ACT_VALUE","ACT_UNIT","ACT_TYPE",
                        "MOA","ACTION_TYPE","TDL"] if c in dt.columns]
    dt[keep].to_csv(out, index=False)
    print(f"✅ drugcentral_drug_target.csv: {len(dt):,} rows, "
          f"{dt['uniprot'].nunique()} proteins, {dt['DRUG_NAME'].nunique()} drugs")
    return dt[keep]


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAERS pharmacovigilance signals
# ─────────────────────────────────────────────────────────────────────────────
def fetch_faers():
    out = DRIVE / "drugcentral_faers.csv"
    import psycopg2, math
    print("Fetching FAERS from DrugCentral PostgreSQL...")
    conn = psycopg2.connect(**DC_CONN)
    faers = pd.read_sql(
        "SELECT * FROM faers.faers_indi_sig LIMIT 500000;", conn)
    # Try alternate table name if above fails
    if faers.empty:
        faers = pd.read_sql("SELECT * FROM faers_signal LIMIT 500000;", conn)
    # Drug names from structures
    structs = pd.read_sql(
        "SELECT id AS struct_id, name AS drug_name FROM structures;", conn)
    conn.close()
    faers = faers.merge(structs, on="struct_id", how="left")
    # ROR from 2x2 table
    if all(c in faers.columns for c in
           ["drug_ae","drug_no_ae","no_drug_ae","no_drug_no_ae"]):
        a = faers["drug_ae"].clip(lower=1)
        b = faers["drug_no_ae"].clip(lower=1)
        c = faers["no_drug_ae"].clip(lower=1)
        d = faers["no_drug_no_ae"].clip(lower=1)
        faers["ror"] = (a * d) / (b * c)
        se = ((1/a) + (1/b) + (1/c) + (1/d)) ** 0.5
        faers["ror_lo95"] = (faers["ror"].apply(math.log) - 1.96*se).apply(math.exp)
        faers["ror_hi95"] = (faers["ror"].apply(math.log) + 1.96*se).apply(math.exp)
        faers["ror_signal"] = ((faers["ror"] > 1) &
                               (faers["ror_lo95"] > 1) &
                               (faers["drug_ae"] >= 3)).astype(int)
    faers.to_csv(out, index=False)
    print(f"✅ drugcentral_faers.csv: {len(faers):,} rows")
    return faers


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRING PPI (DC-aware canonical UniProt mapping)
# ─────────────────────────────────────────────────────────────────────────────
def build_string_ppi():
    out = DRIVE / "string_ppi.csv"
    dt  = pd.read_csv(DRIVE / "drugcentral_drug_target.csv")
    dc_proteins = set(dt["uniprot"].dropna())

    alias_path = DRIVE / "9606.protein.aliases.v12.0.txt.gz"
    links_path = DRIVE / "9606.protein.links.v12.0.txt.gz"

    for url, path in [
        ("https://stringdb-downloads.org/download/protein.aliases.v12.0/"
         "9606.protein.aliases.v12.0.txt.gz", alias_path),
        ("https://stringdb-downloads.org/download/protein.links.v12.0/"
         "9606.protein.links.v12.0.txt.gz", links_path),
    ]:
        if not path.exists():
            print(f"Downloading {path.name}...")
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(131072):
                    f.write(chunk)

    print("Building DC-aware canonical UniProt map...")
    with gzip.open(alias_path, "rt") as f:
        aliases = pd.read_csv(f, sep="\t", header=0,
                              names=["string_id","alias","source"],
                              low_memory=False)
    uni_all = (aliases[aliases["source"].str.strip() == "UniProt_AC"]
               [["string_id","alias"]]
               .rename(columns={"alias":"uniprot"})
               .drop_duplicates())
    uni_all["in_dc"] = uni_all["uniprot"].isin(dc_proteins)
    uni_sorted = uni_all.sort_values(["string_id","in_dc","uniprot"],
                                     ascending=[True, False, True])
    canonical = uni_sorted.drop_duplicates(subset="string_id")[["string_id","uniprot"]]

    print("Filtering STRING edges (combined_score >= 700)...")
    links = pd.read_csv(links_path, sep=" ", compression="gzip")
    links = links[links["combined_score"] >= 700]
    links = (links
             .merge(canonical.rename(columns={"string_id":"protein1","uniprot":"uniprot1"}),
                    on="protein1", how="inner")
             .merge(canonical.rename(columns={"string_id":"protein2","uniprot":"uniprot2"}),
                    on="protein2", how="inner"))
    result = links[["uniprot1","uniprot2","combined_score"]]
    result.to_csv(out, index=False)
    ppi_proteins = set(result["uniprot1"]) | set(result["uniprot2"])
    print(f"✅ string_ppi.csv: {len(result):,} edges | {len(ppi_proteins):,} proteins")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Core protein set (DC ∩ GTEx ∩ STRING)
# ─────────────────────────────────────────────────────────────────────────────
def build_core_proteins():
    out = DRIVE / "core_proteins.csv"
    dt   = pd.read_csv(DRIVE / "drugcentral_drug_target.csv")
    expr = pd.read_csv(DRIVE / "hpa_tissue_expression.csv",
                       usecols=["uniprot"])
    ppi  = pd.read_csv(DRIVE / "string_ppi.csv")
    dc_proteins   = set(dt["uniprot"].dropna())
    expr_proteins = set(expr["uniprot"].dropna())
    ppi_proteins  = set(ppi["uniprot1"]) | set(ppi["uniprot2"])
    core = dc_proteins & expr_proteins & ppi_proteins
    pd.Series(sorted(core)).to_csv(out, index=False, header=["uniprot"])
    print(f"✅ core_proteins.csv: {len(core):,} proteins "
          f"(DC={len(dc_proteins):,}, GTEx={len(expr_proteins):,}, "
          f"STRING={len(ppi_proteins):,})")
    return core


# ─────────────────────────────────────────────────────────────────────────────
# 5. BindingDB / ChEMBL kinase affinities
# ─────────────────────────────────────────────────────────────────────────────
def fetch_kinase_affinities():
    out = DRIVE / "bindingdb_kinase.csv"
    if out.exists() and pd.read_csv(out).shape[0] > 0:
        print(f"BindingDB cached: {pd.read_csv(out).shape}")
        return

    dt = pd.read_csv(DRIVE / "drugcentral_drug_target.csv")
    if "TARGET_CLASS" in dt.columns:
        kinase_uniprots = list(
            dt[dt["TARGET_CLASS"].str.contains("Kinase|kinase", na=False)]
            ["uniprot"].dropna().unique())[:30]
    else:
        kinase_uniprots = list(dt["uniprot"].dropna().unique())[:30]

    print(f"Fetching ChEMBL IC50/Ki for {len(kinase_uniprots)} kinase targets...")
    rows = []
    BASE = "https://www.ebi.ac.uk/chembl/api/data"
    for uniprot in tqdm(kinase_uniprots, desc=" ChEMBL"):
        try:
            tr = requests.get(f"{BASE}/target.json",
                              params={"target_components__accession": uniprot},
                              timeout=20)
            targets = tr.json().get("targets", [])
            if not targets:
                continue
            chembl_id = targets[0]["target_chembl_id"]
            ar = requests.get(f"{BASE}/activity.json", params={
                "target_chembl_id": chembl_id,
                "standard_type__in": "IC50,Ki",
                "standard_units": "nM",
                "assay_type": "B",
                "limit": 200,
            }, timeout=30)
            for a in ar.json().get("activities", []):
                rows.append({
                    "drug_name": a.get("molecule_pref_name",""),
                    "uniprot": uniprot,
                    "target_name": a.get("target_pref_name",""),
                    "affinity_nM": a.get("standard_value"),
                    "affinity_type": a.get("standard_type",""),
                })
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)
        df["affinity_nM"] = pd.to_numeric(df["affinity_nM"], errors="coerce")
        df = df.dropna(subset=["affinity_nM"])
        df.to_csv(out, index=False)
        print(f"✅ bindingdb_kinase.csv: {len(df):,} rows")
    else:
        # Fallback: DrugCentral ACT_VALUE
        dc_aff = dt[dt["ACT_VALUE"].notna()].copy() if "ACT_VALUE" in dt.columns else pd.DataFrame()
        dc_aff = dc_aff.rename(columns={"DRUG_NAME":"drug_name","TARGET_NAME":"target_name",
                                         "ACT_VALUE":"affinity_nM","ACT_TYPE":"affinity_type"})
        dc_aff.to_csv(out, index=False)
        print(f"✅ bindingdb_kinase.csv (DC fallback): {len(dc_aff):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CT-ADE benchmark
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ctade():
    out = DRIVE / "ctade_drug_ae.csv"
    if out.exists() and pd.read_csv(out).shape[0] > 0:
        print(f"CT-ADE cached: {pd.read_csv(out).shape}")
        return

    print("Downloading CT-ADE test split from Figshare...")
    url = "https://ndownloader.figshare.com/files/51498449"
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in tqdm(r.iter_content(131072),
                              desc=" CT-ADE", unit="chunk"):
                f.write(chunk)
        df = pd.read_csv(out)
        print(f"✅ ctade_drug_ae.csv: {df.shape}")
    except Exception as e:
        print(f"⚠️  CT-ADE download failed: {e}")
        print("Manual: wget https://ndownloader.figshare.com/files/51498449 "
              f"-O {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Unified ID maps
# ─────────────────────────────────────────────────────────────────────────────
def build_id_maps():
    out = DRIVE / "id_maps.pkl"
    dt   = pd.read_csv(DRIVE / "drugcentral_drug_target.csv")
    faers= pd.read_csv(DRIVE / "drugcentral_faers.csv")
    expr = pd.read_csv(DRIVE / "hpa_tissue_expression.csv",
                       usecols=["uniprot","Tissue"])
    ppi  = pd.read_csv(DRIVE / "string_ppi.csv")
    core = pd.read_csv(DRIVE / "core_proteins.csv")

    proteins = set(core["uniprot"].dropna())
    for df, col in [(dt,"uniprot"), (ppi,"uniprot1"), (ppi,"uniprot2"),
                    (expr,"uniprot")]:
        proteins |= set(df[col].dropna())
    protein2id = {p: i for i, p in enumerate(sorted(proteins))}

    drugs = set()
    if "DRUG_NAME" in dt.columns:
        drugs |= set(dt["DRUG_NAME"].str.lower().dropna())
    if "drug_name" in faers.columns:
        drugs |= set(faers["drug_name"].str.lower().dropna())
    drug2id = {d: i for i, d in enumerate(sorted(drugs))}

    tissues = set(expr["Tissue"].dropna())
    tissue2id = {t: i for i, t in enumerate(sorted(tissues))}

    ae_col = next((c for c in faers.columns
                   if "meddra_name" in c.lower() or "ae_term" in c.lower()), None)
    aes = set(faers[ae_col].dropna()) if ae_col else set()
    ae2id = {a: i for i, a in enumerate(sorted(aes))}

    maps = {"protein2id": protein2id, "drug2id": drug2id,
            "tissue2id": tissue2id, "ae2id": ae2id,
            "n_proteins": len(protein2id), "n_drugs": len(drug2id),
            "n_tissues": len(tissue2id), "n_aes": len(ae2id),
            "core_proteins": sorted(core["uniprot"].dropna().tolist())}
    with open(out, "wb") as f:
        pickle.dump(maps, f)
    print(f"✅ id_maps.pkl — proteins:{len(protein2id):,} drugs:{len(drug2id):,} "
          f"tissues:{len(tissue2id)} AEs:{len(ae2id):,}")
    return maps


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("ProToxNet — Step 1: Data Acquisition")
    print("="*60)
    fetch_drugcentral_targets()
    fetch_faers()
    build_string_ppi()
    build_core_proteins()
    fetch_kinase_affinities()
    fetch_ctade()
    build_id_maps()
    print("\n✅ Step 1 complete. Next: step2_conplex.py")
