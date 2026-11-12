"""Regenerate fig12 from the saved results of step 05 (no need to re-run the experiment)."""
import pandas as pd

from utils import plot_selected_genes

sel = pd.read_csv("results/05_expr_selected.csv")
coefs = pd.read_csv("results/05_expr_gene_coefficients.csv")
plot_selected_genes(sel, coefs)