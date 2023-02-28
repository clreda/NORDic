#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kendalltau

root_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/"
##file_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/MDD/MDDMale_JURKAT/"
#file_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/MDD/MDDFemale_JURKAT/"
###file_folder="refractory_epilepsy2/"
file_folder="refractory_epilepsy/"
#cytoscape_name = root_folder+"M30_code/cytoscape/networks_M30_brainlines/solution.sif default node.csv"
###spreads_name=file_folder+"application_regulators.csv"
spreads_name=file_folder+"spread_values.csv"
##cytoscape_name=file_folder+"solution_cytoscape.sif_1 default node.csv"
cytoscape_name=file_folder+"solution_cytoscape.sif default node.csv"

max_show=15
normalize=lambda x : (x-x.min())/(x-x.min()).max()

## Spread values
spreads = pd.read_csv(spreads_name, index_col=0)
spreads = spreads.sort_values(by=spreads.columns[0], ascending=False)
## ControlCentrality (CC) values as computed by Cytoscape
cytoscape = pd.read_csv(cytoscape_name)
cytoscape.index = cytoscape[["shared name"]].values.flatten().tolist()
cytoscape = cytoscape[["ControlCentrality"]].astype(float)
for g in [gg for gg in list(spreads.index) if (gg not in list(cytoscape.index))]:
    cytoscape.loc[g] = 0
cytoscape = cytoscape.sort_values(by="ControlCentrality", ascending=False)
## pLI scores from gnomad
ffolder = root_folder+"data/gnomad/"
ffile = "gnomad.v2.1.1.lof_metrics.by_gene.txt"
pli = pd.read_csv(ffolder+ffile, index_col=0, sep="\t")[['pLI']]
pli = pli.loc[[g for g in list(spreads.index) if (g in pli.index)]].astype(np.float64).sort_values(by="pLI", ascending=False)
pli.index = list(map(str,pli.index))
for g in [gg for gg in list(spreads.index) if (gg not in list(pli.index))]:
    pli.loc[g] = 0

pli = pli.loc[list(spreads.index)]
cytoscape = cytoscape.loc[list(spreads.index)]

spreads.columns = ["Spread"]
spreads["pLI"] = pli["pLI"]
spreads["CC"] = cytoscape["ControlCentrality"]

#thres = {"ttt": 0.133, "pLI":0.9, "CC": 5, "Spread": 0.14}
##thres = {"ttt": 0.133, "pLI":0.9, "CC": 30.5, "Spread": 0.25}
###thres = {"ttt": 0.02, "pLI":0.5, "CC": 5, "Spread": 0.14}
thres = {"ttt": 0.005, "pLI":0.5, "CC": 5, "Spread": 0.021}

tau1, p1 = kendalltau(list(spreads["pLI"]), list(spreads[spreads.columns[0]]))
tau2, p2 = kendalltau(list(spreads["CC"]), list(spreads[spreads.columns[0]]))

ttt = thres["ttt"]
spreads = spreads.loc[spreads[spreads.columns[0]]>ttt].sort_values(by=spreads.columns[0], ascending=False)
ffsize=30#43*14/spreads.shape[0]
cm = ["Blues", "Oranges", "Greens", "Purples", "winter", "bone", 'Reds']
f, axs = plt.subplots(spreads.columns.size, 1, gridspec_kw={'wspace': 0},figsize=(50,2))
for i, (s, a, c) in enumerate(zip(spreads.columns, axs, cm)):
    im = a.imshow(np.array([spreads[s].values]), interpolation="None", cmap=c, aspect='auto')
    for u in range(spreads.shape[0]):
        text = a.text(u, 0, np.round(spreads[s].values.flatten().tolist()[u],3), ha="center",va="center",color="w" if (spreads[s].values.flatten().tolist()[u]>thres[s]) else "k",fontsize=ffsize)
    divider = make_axes_locatable(a)
    ca = divider.append_axes("right",size="2%",pad=0)#size="5%",pad=0.05)
    cbar = plt.colorbar(im,cax=ca)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(ffsize-10)
    if (i==0):
        a.set_title(r"Kendall's $\tau$:          pLI||Spread $\tau=%.2f$ ($p=%.3f$)          CC||Spread $\tau=%.2f$ ($p=%.3f$)" % (tau1, p1, tau2, p2), fontsize=ffsize)
    if (i==spreads.shape[1]-1):
        a.set_xticks([0]+list(range(spreads.shape[0])))
        a.set_xticklabels([""]+list(spreads.index), rotation=50,fontsize=ffsize)
    else:
        a.set_xticks([])
        a.set_xticklabels([])
    a.set_yticks([])
    a.set_yticklabels([])
    #a.set_yticks([np.mean(spreads[s].values.flatten().tolist())])
    #a.set_yticklabels([s], fontsize=ffsize)
    #a.tick_params(axis='y', colors='black')
#plt.savefig(file_folder+"values.pdf",bbox_inches="tight")
plt.savefig(file_folder+"values_newmaboss.pdf",bbox_inches="tight")
plt.close()
