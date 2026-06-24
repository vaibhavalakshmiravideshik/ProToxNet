"""
ProToxNet | Step 2: ConPLex Proteome-Wide Binding Scoring
==========================================================
Inputs:
  core_proteins.csv       — 1,507 UniProt IDs
  drugcentral_drug_target.csv — known binding pairs (calibration)
  id_maps.pkl             — drug/protein ID maps

Outputs:
  protein_sequences.pkl   — UniProt sequences (cached)
  protein_esm_embeddings.pkl — ESM-1b mean-pool embeddings (cached)
  conplex_scores_raw.csv  — drug × protein scores (6.5M pairs)
  conplex_scores_matrix.pkl — pivot matrix (drugs × proteins)
  conplex_calibration.pkl — Platt scaling params
  conplex_positives.csv   — known pairs with scores (sanity check)

Architecture:
  Drug:    MorganFeaturizer (ConPLex) → 2048-bit Morgan FP
  Protein: ESM-1b t33 mean-pool → 1280-dim
  Model:   SimpleCoembeddingNoSigmoid(2048, 1280, 1024) cosine similarity
  Checkpoint: ConPLex BindingDB_ExperimentalValidModel.pt

NOTE: Requires A100 GPU. ESM-1b runs on CPU, ConPLex scoring on GPU.
"""

import os, pickle, subprocess, sys, time, urllib.request
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
from pipeline import get_data_dir, get_repo_root

DRIVE  = get_data_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ConPLex repo
CONPLEX_DIR = str(get_repo_root() / ".cache" / "ConPLex")
if not os.path.exists(CONPLEX_DIR):
    subprocess.check_call(["git", "clone", "--depth=1",
                           "https://github.com/samsledje/ConPLex.git",
                           CONPLEX_DIR])
sys.path.insert(0, CONPLEX_DIR)

from conplex_dti.featurizer import MorganFeaturizer
from conplex_dti.model.architectures import SimpleCoembeddingNoSigmoid
import esm as esm_lib

CKPT_DIR = DRIVE / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "BindingDB_ExperimentalValidModel.pt"
CKPT_URL  = ("https://cb.csail.mit.edu/cb/conplex/data/models/"
             "BindingDB_ExperimentalValidModel.pt")


# ── Download ConPLex checkpoint ───────────────────────────────────────────────
def download_checkpoint():
    if CKPT_PATH.exists():
        print("✅ ConPLex checkpoint already cached")
        return
    print("Downloading ConPLex checkpoint (~200 MB)...")
    import ssl
    ctx = ssl.SSLContext()
    with urllib.request.urlopen(CKPT_URL, context=ctx) as r, \
         open(CKPT_PATH, "wb") as f:
        f.write(r.read())
    print(f"✅ Saved to {CKPT_PATH}")


# ── ESM-1b featurizer ─────────────────────────────────────────────────────────
class ESM1bFeaturizer:
    """ESM-1b t33 mean-pool. Runs on CPU; embeddings cached after first run."""
    shape = 1280

    def __init__(self, cache_dir: Path):
        self.device = torch.device("cpu")
        torch.hub.set_dir(str(cache_dir))
        print("Loading ESM-1b on CPU (~2.5 GB RAM)...")
        self._model, self._alphabet = esm_lib.pretrained.esm1b_t33_650M_UR50S()
        self._model = self._model.eval()
        self._converter = self._alphabet.get_batch_converter()
        self._max_len = 1022
        print("✅ ESM-1b ready")

    @torch.no_grad()
    def embed(self, seq: str) -> np.ndarray:
        seq = seq.upper()[:self._max_len]
        _, _, tokens = self._converter([("s", seq)])
        out = self._model(tokens, repr_layers=[33])
        emb = out["representations"][33][0, 1:len(seq)+1].mean(0)
        return emb.numpy()


# ── Fetch UniProt sequences ───────────────────────────────────────────────────
def fetch_sequences(protein_ids):
    cache = DRIVE / "protein_sequences.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            seqs = pickle.load(f)
        print(f"Sequences cached: {len(seqs):,}")
        return seqs

    seqs, failed = {}, []
    for i in tqdm(range(0, len(protein_ids), 50), desc="UniProt batches"):
        batch = protein_ids[i:i+50]
        query = " OR ".join(f"accession:{u}" for u in batch)
        url = ("https://rest.uniprot.org/uniprotkb/stream"
               f"?query={urllib.request.quote(query)}&format=fasta")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                fasta = r.read().decode()
            cur_id, cur_seq = None, []
            for line in fasta.split("\n"):
                if line.startswith(">"):
                    if cur_id:
                        seqs[cur_id] = "".join(cur_seq)
                    parts = line.split("|")
                    cur_id = parts[1] if len(parts) > 1 else line[1:].split()[0]
                    cur_seq = []
                elif line.strip():
                    cur_seq.append(line.strip())
            if cur_id:
                seqs[cur_id] = "".join(cur_seq)
        except Exception:
            failed.extend(batch)
        time.sleep(0.3)

    for uid in tqdm(failed, desc="Retrying"):
        try:
            with urllib.request.urlopen(
                f"https://rest.uniprot.org/uniprotkb/{uid}.fasta",
                timeout=15) as r:
                fasta = r.read().decode()
            seq = "".join(l.strip() for l in fasta.split("\n")
                          if not l.startswith(">") and l.strip())
            if seq:
                seqs[uid] = seq
        except Exception:
            pass
        time.sleep(0.1)

    with open(cache, "wb") as f:
        pickle.dump(seqs, f)
    print(f"✅ Fetched {len(seqs):,} / {len(protein_ids):,} sequences")
    return seqs


# ── Compute ESM-1b embeddings ─────────────────────────────────────────────────
def compute_embeddings(protein_ids, protein_seqs, featurizer):
    cache = DRIVE / "protein_esm_embeddings.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            embs = pickle.load(f)
        print(f"Embeddings cached: {len(embs):,}")
        return embs

    embs = {}
    print(f"Computing {len(protein_ids):,} ESM-1b embeddings (~45 min on A100)...")
    for uid in tqdm(protein_ids, desc="ESM-1b"):
        if uid not in protein_seqs:
            continue
        try:
            embs[uid] = featurizer.embed(protein_seqs[uid])
        except Exception as e:
            print(f"  ⚠️ {uid}: {e}")
        if len(embs) % 100 == 0:
            with open(cache, "wb") as f:
                pickle.dump(embs, f)
    with open(cache, "wb") as f:
        pickle.dump(embs, f)
    print(f"✅ Embeddings: {len(embs):,}")
    return embs


# ── Build ConPLex model ───────────────────────────────────────────────────────
def build_model(drug_dim=2048, target_dim=1280, latent=1024):
    model = SimpleCoembeddingNoSigmoid(drug_dim, target_dim, latent)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if "drug_projector.0.weight" in state:
        model.drug_projector[0].weight.data = state["drug_projector.0.weight"]
        model.drug_projector[0].bias.data   = state["drug_projector.0.bias"]
        print("✅ Drug projector weights loaded")
    else:
        model.load_state_dict(state, strict=False)
    return model.eval().to(DEVICE)


# ── Score all drug-protein pairs ──────────────────────────────────────────────
def score_pairs(model, drug_list, drug_smiles, prot_emb_cache,
                protein_list, drug_featurizer):
    raw_out  = DRIVE / "conplex_scores_raw.csv"
    mtx_out  = DRIVE / "conplex_scores_matrix.pkl"
    ckpt_sc  = DRIVE / "conplex_scoring_checkpoint.pkl"

    all_rows, start_pi = [], 0
    if ckpt_sc.exists():
        with open(ckpt_sc, "rb") as f:
            ckpt = pickle.load(f)
        all_rows  = ckpt["rows"]
        start_pi  = ckpt["protein_idx"]
        print(f"Resuming from protein index {start_pi}")

    DRUG_BATCH = 512
    for pi, prot in enumerate(tqdm(protein_list, desc="Proteins")):
        if pi < start_pi:
            continue
        p_emb = torch.tensor(prot_emb_cache[prot],
                             dtype=torch.float32).to(DEVICE)
        for di in range(0, len(drug_list), DRUG_BATCH):
            batch = drug_list[di:di+DRUG_BATCH]
            d_embs = torch.stack(
                [drug_featurizer(drug_smiles[d]) for d in batch]).to(DEVICE)
            p_batch = p_emb.unsqueeze(0).expand(len(batch), -1)
            with torch.no_grad():
                scores = model(d_embs, p_batch).cpu().numpy()
            for drug, score in zip(batch, scores):
                all_rows.append({"drug": drug, "protein": prot,
                                 "conplex_score": float(score)})
        if (pi + 1) % 50 == 0:
            with open(ckpt_sc, "wb") as f:
                pickle.dump({"rows": all_rows, "protein_idx": pi+1}, f)

    scores_df = pd.DataFrame(all_rows)
    scores_df.to_csv(raw_out, index=False)
    matrix = scores_df.pivot(index="drug", columns="protein",
                             values="conplex_score")
    with open(mtx_out, "wb") as f:
        pickle.dump(matrix, f)
    print(f"✅ Scores: {len(scores_df):,} pairs | Matrix: {matrix.shape}")
    return scores_df


# ── Platt scaling calibration ─────────────────────────────────────────────────
def calibrate(scores_df, drug_target):
    cal_out = DRIVE / "conplex_calibration.pkl"
    pos_out = DRIVE / "conplex_positives.csv"

    name_col = next((c for c in drug_target.columns
                     if "name" in c.lower() or "drug" in c.lower()),
                    drug_target.columns[0])
    acc_col  = next((c for c in drug_target.columns
                     if "accession" in c.lower() or "uniprot" in c.lower()),
                    drug_target.columns[1])
    pos_pairs = set(zip(drug_target[name_col].str.lower(),
                        drug_target[acc_col]))
    scores_df["label"] = scores_df.apply(
        lambda r: 1 if (r["drug"].lower(), r["protein"]) in pos_pairs else 0,
        axis=1)

    n_pos = int(scores_df["label"].sum())
    if n_pos < 10:
        print("⚠️  Insufficient positives for calibration")
        return

    pos_df = scores_df[scores_df["label"] == 1]
    neg_df = scores_df[scores_df["label"] == 0].sample(
        min(n_pos*10, len(scores_df)-n_pos), random_state=42)
    cal    = pd.concat([pos_df, neg_df]).dropna(subset=["conplex_score"])
    scaler = StandardScaler()
    X = scaler.fit_transform(cal[["conplex_score"]].values)
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    platt.fit(X, cal["label"].values)
    auc = roc_auc_score(cal["label"], platt.predict_proba(X)[:,1])
    print(f"Calibration AUC: {auc:.4f}")
    scores_df["calibrated_prob"] = platt.predict_proba(
        scaler.transform(scores_df[["conplex_score"]].values))[:,1]
    scores_df.to_csv(DRIVE / "conplex_scores_raw.csv", index=False)
    with open(cal_out, "wb") as f:
        pickle.dump({"scaler": scaler, "platt": platt, "auc": auc}, f)
    scores_df[scores_df["label"]==1].to_csv(pos_out, index=False)
    print(f"✅ Calibration saved | positives: {n_pos:,}")


# ── Drug SMILES from DrugCentral ──────────────────────────────────────────────
def get_drug_smiles():
    import psycopg2
    conn = psycopg2.connect(**dict(host="unmtid-dbs.net", port=5433,
                                   dbname="drugcentral", user="drugman",
                                   password="dosage"))
    structs = pd.read_sql(
        "SELECT name, smiles FROM structures WHERE smiles IS NOT NULL", conn)
    conn.close()
    def canon(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol else None
        except:
            return None
    structs["canon"] = structs["smiles"].apply(canon)
    result = (structs.dropna(subset=["canon"])
              .drop_duplicates("name")
              .set_index("name")["canon"].to_dict())
    print(f"SMILES: {len(result):,}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("ProToxNet — Step 2: ConPLex Scoring")
    print("="*60)

    download_checkpoint()

    core_proteins = pd.read_csv(DRIVE / "core_proteins.csv")
    protein_ids   = core_proteins["uniprot"].tolist()
    drug_target   = pd.read_csv(DRIVE / "drugcentral_drug_target.csv")

    drug_smiles = get_drug_smiles()
    drug_list   = sorted(drug_smiles.keys())

    protein_seqs = fetch_sequences(protein_ids)
    valid_proteins = [p for p in protein_ids if p in protein_seqs]

    featurizer = ESM1bFeaturizer(CKPT_DIR)
    prot_emb_cache = compute_embeddings(valid_proteins, protein_seqs, featurizer)
    del featurizer
    import gc; gc.collect()

    drug_featurizer = MorganFeaturizer(save_dir=str(CKPT_DIR)).to(DEVICE)
    drug_featurizer.preload([drug_smiles[d] for d in drug_list],
                            force_recompute=False)

    model = build_model()
    protein_list = [p for p in valid_proteins if p in prot_emb_cache]

    scores_df = score_pairs(model, drug_list, drug_smiles,
                            prot_emb_cache, protein_list, drug_featurizer)
    calibrate(scores_df, drug_target)

    print("\n✅ Step 2 complete. Next: step3_exposure.py")
