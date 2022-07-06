
import os.path as osp
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from typing import Optional


def data_encapsulation(data, save=False):
    '''读csv数据，from txt2data.py
    '''
    allowed_columns = ["GeneName", "UMICount", "Label", "x", "y", "Tag"]
    for col in data.columns:
        assert col in allowed_columns, "Error, Got an invalid column name, which only support {}".format(allowed_columns)
    cell_list = data["Label"].astype("category")
    gene_list = data["GeneName"].astype("category")
    vals = data["UMICount"].to_numpy()
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    coo = data.groupby("Label").mean()[["x", "y"]]
    obs = pd.DataFrame(index=(map(str, cell_list.cat.categories)))
    var = pd.DataFrame(index=(map(str, gene_list.cat.categories)))
    adata_x = sp.csr_matrix((vals, (row, col)), shape=(len(obs), len(var)))
    adata = AnnData(adata_x, obs=obs, var=var)
    adata.obsm['spatial'] = coo.to_numpy()
    if isinstance(save, str):
        adata.write_h5ad(osp.join(save, "adata.h5ad"))
    return adata


def load_data(dfs, batch_key: str = "batch", join: str = "inner",
              index_unique: Optional[str] = "-") -> AnnData:

    adata_list = [data_encapsulation(df) for df in dfs]
    concat_adata = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key, index_unique=index_unique)

    return concat_adata


def preprocessing(adata: AnnData,
                  filter_mt: bool = False,
                  norm_and_log: bool = True,
                  z_score: bool = True) -> AnnData:
    """ preprocessing adata """
    if filter_mt:
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 10].copy()

    if norm_and_log:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000.0)
        sc.pp.log1p(adata)

    if z_score:
        adata.X = adata.X.toarray()
        adata.X = (adata.X - adata.X.mean(0)) / adata.X.std(0)

    return adata

