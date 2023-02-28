#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kendalltau

file_folder="refractory_epilepsy2/"
max_show=15
normalize=lambda x : (x-x.min())/(x-x.min()).max()

#spreads_mp = pd.read_csv(file_folder+"application_regulators.csv", index_col=0)
spreads_mp = pd.read_csv(file_folder+"spread_values_maboss.csv", index_col=0)
spreads_mp.columns = ["0"]
spreads_maboss = pd.read_csv(file_folder+"spread_values.csv", index_col=0)
spreads_maboss.columns = ["1"]
spreads = spreads_mp.join(spreads_maboss, how="inner").sort_values(by=spreads_mp.columns[0],ascending=False)
#spreads.columns = ["MP", "MABOSS"]
spreads.columns = ["old MABOSS", "new MABOSS"]

#tau, p = kendalltau(list(spreads["MP"]), list(spreads["MABOSS"]))
tau, p = kendalltau(list(spreads[spreads.columns[0]]), list(spreads[spreads.columns[1]]))

ttt = 0.01#0.005
f, axs = plt.subplots(spreads.columns.size*2+1, 1, gridspec_kw={"wspace": 0}, figsize=(50, spreads.shape[1]*2+2))
ax = axs[spreads.shape[1]]
ax.grid(False)
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])
#ffsize=43*14/spreads.shape[0]
ffsize=500*14/spreads.shape[0]
for ic, col in enumerate(spreads.columns):
    spreads = spreads.loc[spreads[spreads.columns[1]]>ttt].sort_values(by=col, ascending=False)
    cm = ["Blues", "Oranges", "Greens", "Purples", "winter", "bone", 'Reds']
    for i, (s, a, c) in enumerate(zip(spreads.columns, axs[(ic*spreads.shape[1]+ic):(ic*spreads.shape[1]+ic+spreads.shape[1])], cm)):
        im = a.imshow(np.array([spreads[s].values]), interpolation="None", cmap=c, aspect='auto')
        for u in range(spreads.shape[0]):
            text = a.text(u, 0, np.round(spreads[s].values.flatten().tolist()[u],3), ha="center",va="center",color="w" if (("MABOSS" in s) and spreads[s].values.flatten().tolist()[u]>0.03) or (("MP" in s) and spreads[s].values.flatten().tolist()[u]>0.14) else "k",fontsize=ffsize)
        a.set_xticks([0,0])
        a.set_xticklabels(["",spreads.index[i]])
        divider = make_axes_locatable(a)
        ca = divider.append_axes("right",size="2%",pad=0.)
        cbar = plt.colorbar(im,cax=ca)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(ffsize*2/3)
        a.set_yticks([0])
        a.set_yticklabels(["%s\n(sorted by %s)" % (s, col) if (i==0) else s],fontsize=ffsize)
        if (i>spreads.shape[1]-2):
            a.set_xticks([0]+list(range(spreads.shape[0])))
            a.set_xticklabels([""]+list(spreads.index), rotation=0,fontsize=ffsize)
        else:
            a.set_xticks([])
            a.set_xticklabels([])
axs[0].set_title(r"Kendall's $\tau=%.2f$ ($p=%s$)" % (tau, p), fontsize=ffsize)
#plt.savefig(file_folder+"comparaison_values.pdf",bbox_inches="tight")
plt.savefig(file_folder+"comparaison_values_OLDMABOSS_VS_NEWMABOSS.pdf",bbox_inches="tight")
plt.close()
