#coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns

def influences2graph(influences, fname, optional=False, compile2png=True, engine=["sfdp","dot"][0]):
    '''
    Plots a network by conversion to a DOT file and then to PNG

    ...

    Parameters
    ----------
    influences : Pandas DataFrame
        rows/[genes] x columns/[genes], contains {-1,1,2}
    fname : Python character string
        filename of png file
    optional : Python bool
        [default=False] : should interactions be drawn as optional (dashed lines)?

    Returns
    ----------
    None
        writes a DOT file which can be converted to PNG image (if compile2png=True)
    '''
    dotfile = fname+".dot"
    filename = fname+".png"
    graph = ["digraph {"]
    for idx in influences.columns:
        if (len(idx)>0):
            graph += ["\""+idx+"\" [shape=circle,fillcolor=grey,style=filled];"]
    directed, undirected = ["subgraph D {"], ["subgraph U {"]
    in_undirected = []
    for x,y in np.argwhere(influences.values!=0).tolist():
        nx, ny = influences.index[x], influences.columns[y]
        sign = influences.values[x,y]
        if ((y,x,sign) in in_undirected):
            continue
        edge = " ".join(["\""+nx+"\"", "->", "\""+ny+"\""])
        edge += " ["
        if (influences.values[y,x]!=0):
            edge += "dir=none,"
        edge += "label="+("\"-\"" if (sign<0) else ("\"+\"" if ((sign>0) and (sign<2)) else "\"+-\""))
        edge += ",penwidth=3,color="+("red" if (sign<0) else ("green" if ((sign>0) and (sign<2)) else "black"))
        edge += ",arrowhead=tee" if (sign < 0) else ""
        edge += ",style=dashed" if (optional) else ""
        edge += "];"
        if (influences.values[y,x]!=0):
            undirected += [edge]
            in_undirected += [(x,y,sign)]
        else:
            directed += [edge]
    directed += ["}"]
    undirected += ["}"]
    graph += (directed if (len(directed)>2) else [])+(undirected if (len(undirected)>2) else [])+["}"]
    with open(dotfile, "w+") as f:
        f.write("\n".join(graph))
    if (compile2png):
        from subprocess import call as sbcall
        sbcall(engine+" -Tpng "+dotfile+" > "+filename+" && rm -f "+dotfile, shell=True)
    return None

def plot_influence_graph(network_df, input_col, output_col, sign_col, direction_col=None, fname="graph.png", optional=True):
    '''
    Converts a network into a PNG picture

    ...

    Parameters
    ----------
    network_df : Pandas DataFrame
        rows/[index] x columns/[input_col,output_col,sign_col]
    input_col,output_col,sign_col,direction_col : Python character string
        columns of network_df
    fname : Python character string
        [default="graph.png"] : file name for PNG picture
    optional : Python bool
        [default=True] : should edges be plotted as dashed lines?

    Returns
    ----------
    None
        Creates a image of the graph in file fname
    '''
    influences = network_df.pivot_table(index=input_col, columns=output_col, values=sign_col, aggfunc='mean')
    influences[influences==0] = 2
    influences = influences.fillna(0)
    influences[influences<0] = -1
    influences[(influences>0)&(influences<2)] = 1
    missing_cols = [g for g in influences.index if (g not in influences.columns)]
    influences = pd.concat((influences, pd.DataFrame([], index=influences.index, columns=missing_cols).fillna(0)), axis=1)
    missing_idxs = [g for g in influences.columns if (g not in influences.index)]
    influences = pd.concat((influences, pd.DataFrame([], index=missing_idxs, columns=influences.columns).fillna(0)), axis=0)
    infl_mat = influences.values
    if (direction_col is not None):
        for x in network_df.index:
            is_directed = bool(network_df.loc[x][direction_col])
            if (not is_directed):
                inp = list(influences.index).index(str(network_df.loc[x]["preferredName_A"]))
                out = list(influences.columns).index(str(network_df.loc[x]["preferredName_B"]))
                infl_mat[inp, out] = int(network_df.loc[x][sign_col])
                infl_mat[out, inp] = infl_mat[inp, out]
    influences = pd.DataFrame(infl_mat, index=influences.index, columns=influences.columns).loc[influences.index][influences.index]
    influences = influences.astype(int)
    influences.index = influences.index.astype(str)
    influences.columns = influences.columns.astype(str)
    influences2graph(influences, fname, optional=optional)
    return None

def plot_signatures(signatures, perturbed_genes=None, width=10, height=10, max_show=50, fname="signatures"):
    '''
    Print signatures

    ...

    Parameters
    ----------
    signatures : Pandas DataFrame
        rows/[genes] x columns/[signature IDs]
    perturbed_genes : Python character string list
        [default=None] : list of gene names perturbed in the signatures
    width, height : Python integer
        [default=10] : dimensions of image
    max_show : Python integer
        [default=50] : maximum number of genes shown (as only the @max_show genes with highest variance across signatures are plotted)
    fname : Python character string
        [default="signatures"] : path of resulting PNG image

    Returns
    ----------
    None
        plots the signatures as heatmaps in file fname
    '''
    from matplotlib import colors,rc
    rc("ytick", labelsize=5)
    rc("xtick", labelsize=10)
    sigs = deepcopy(signatures)
    sigs[sigs == 0] = -1
    sigs[pd.isnull(sigs)] = 0 
    max_genes = np.argsort(np.var(sigs.values, axis=1))[-max_show:]
    max_genes = sigs.index[max_genes]
    for g in perturbed_genes:
        if (g not in sigs.index):
            sigs.loc[g] = 0
    max_genes = perturbed_genes+[g for g in max_genes if (g not in perturbed_genes)]
    sigs_ = sigs.loc[max_genes]
    fig, ax = plt.subplots(figsize=(width,height))
    cmap = colors.ListedColormap(['red', 'black', 'green'])
    bounds=[-1, -0.5, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    pos = ax.imshow(sigs_, cmap=cmap, origin="lower", interpolation='nearest', norm=norm) #remove score line
    plt.colorbar(pos, ax=ax, ticks=[-1, 0, 1], shrink=0.12, aspect=4)
    ax.set_xticks(range(sigs_.shape[1]))
    ax.set_xticklabels(sigs.columns, rotation=90)
    ax.set_yticks(range(sigs_.shape[0]))
    ax.set_yticklabels(sigs_.index)
    plt.savefig(fname+".png", bbox_inches="tight")
    plt.close()
    return None

def plot_distributions(profiles, fname="gene_expression_distribution.png", thres=None):
    '''
    Plots the distributions (boxplots) of gene expression across samples for each gene, and the selected threshold for binarization

    ...

    Parameters
    ----------
    profiles : Pandas DataFrame
        rows/[genes+annotations] x columns/[samples]
    fname : Python character string
        [default="gene_expression_distribution.png"] : file name
    thres : Python float or None
        [default=None] : binarization threshold (if there is any)


    Returns
    ----------
    None
        plots boxplots of expression for each gene in profiles
    '''
    add_rows_profiles = ["annotation", "perturbed", "perturbation", "cell_line", "sigid"]
    bp = profiles.loc[[g for g in profiles.index if (g not in add_rows_profiles)]].T.apply(pd.to_numeric).boxplot(rot=90, figsize=(25,15))
    if (str(thres)!="None"):
        K = profiles.shape[0]-3
        for t in [thres, 1-thres]:
            plt.plot(list(range(K)), [t]*K, "r--")
    plt.xlabel("Genes")
    plt.ylabel("Gene expression")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return None

def plot_discrete_distributions(signatures, fname="signature_expression_distribution.png"):
    '''
    Plots the distributions (histograms) of genes with determined status across signatures

    ...

    Parameters
    ----------
    signatures : Pandas DataFrame
        rows/[genes] x columns/[samples] with values in {0,NaN,1}. Determined status is either 0 or 1.
    fname : Python character string
        [default="signature_expression_distribution.png"] : file name

    Returns
    ----------
    None
        plots the number of genes with expression values 0, 1 or NaN in each signature
    '''
    N = int(np.sqrt(signatures.shape[1]))+1
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(25,15))
    for iax in range(signatures.shape[1]):
        i,j = iax//N, iax%N
        sig = signatures.iloc[:, iax].fillna(0.5)
        axes[i,j].set_xticks([0,0.5,1])
        axes[i,j].set_xticklabels(('0','NaN','1'))
        axes[i,j].hist(sig.values.T)
        axes[i,j].set_ylabel("#genes")
        axes[i,j].set_title(signatures.columns[iax])
    plt.xlabel("Gene expression level")
    plt.ylabel("# genes")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return None

def plot_boxplots(scores, patient_scores, ground_truth=None, fsize=12, msize=5, fname="boxplots.pdf"):
    '''
    Plots one boxplot per treatment (all values obtained on patient profiles)

    ...

    Parameters
    ----------
    scores : Pandas DataFrame
        rows/[drug names] x column/[value]
    patient_scores : Pandas DataFrame
        rows/[drug names] x columns/[patient samples]
    ground_truth : Pandas DataFrame
        [default=None] : rows/[drug names] x column/[class] Values in 1: treatment, 0: unknown, -1: aggravating. If not provided: does not color boxplots according to the class
    fsize : Python integer
        [default=18] : font size
    msize : Python integer
        [default=5] : marker size
    fname : Python character string
        [default="boxplots"] : file name for the plot

    Returns
    ----------
    None
        create boxplots of reward scores across patients for each drug
    '''
    scores_ = deepcopy(scores).sort_values(by=scores.columns[0], ascending=False)
    drug_names = list(scores_.index)
    sorted_rewards_list = list(scores_[scores_.columns[0]])
    sorted_rewards = patient_scores.loc[drug_names,:]
    if (ground_truth is not None):
        gt_scores = ground_truth.loc[[t for t in scores_.index if (t in ground_truth.index)]]
        gt_missing = pd.DataFrame([], index=[t for t in scores_.index if (t not in ground_truth.index)], columns=ground_truth.columns)
        gt_scores = pd.concat((gt_scores, gt_missing), axis=0).fillna(0).loc[drug_names].astype(int)
        sorted_ground_truth = list(gt_scores[gt_scores.columns[0]])
        positive = [si+1 for si, s in enumerate(sorted_ground_truth) if (s > 0)]
        negative = [si+1 for si, s in enumerate(sorted_ground_truth) if (s < 0)]
    else:
        positive, negative = [], []
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
    bplot = sns.boxplot(sorted_rewards.T, showmeans=False, meanline=False, notch=False, ax=ax)
    if (ground_truth is not None):
        colors = {1: "green", -1: "red", 0: "white"}
        for i in range(len(sorted_rewards)):
            mybox = bplot.artists[i]
            mybox.set_facecolor(colors[sorted_ground_truth[i]])
    ax.plot(range(len(drug_names)), sorted_rewards_list, "kD", label='score')
    ax.set_ylabel("Scores across patient profiles", fontsize=fsize)
    ax.set_xlabel("Drug", fontsize=fsize)
    ax.set_xticklabels(["%s (%.2f)" % (drug_names[i], r) for i, r in enumerate(sorted_rewards_list)], rotation=90, fontsize=fsize)
    plt.yticks(rotation=0)
    cols=["r" if (i in negative) else ("g" if (i in positive) else "k") for i in range(1, scores.shape[0]+1)]
    for xtick, color in zip(ax.get_xticklabels(), cols):
        xtick.set_color(color)
    ax.tick_params(axis='y', which='major', labelsize=fsize)
    plt.legend(fontsize=fsize)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def plot_heatmap(X, ground_truth=None, fname="heatmap.pdf", w=20, h=20, bfsize=20, fsize=20, rot=75):
    '''
    Plots an heatmap of the signatures, with the potential ground truth

    ...

    Parameters
    ----------
    X : Pandas DataFrame
        rows/[features] x columns/[drug names]
    ground_truth : Pandas DataFrame
        [default=None] : rows/[drug names] x column/[class] Values in 1: treatment, 0: unknown, -1: aggravating. If not provided: does not color boxplots according to the class
    fname : Python character string
        [default="heatmap.pdf"] : file name for the plot
    w : Python integer
        [default=20] : figure width
    h : Python integer
        [default=20] : figure height
    bfsize : Python integer
        [default=20] : font size in the color bar
    rot : Python integer
        [default=75] : rotation angle of labels

    Returns
    ----------
    None
        plots an heatmap of similarity across drugs based on the Pearson correlation
    '''
    colors = {1: "green", -1: "red", 0: "black"}
    if (ground_truth is not None):
        gt_scores = ground_truth.loc[[t for t in X.columns if (t in ground_truth.index)]]
        gt_missing = pd.DataFrame([], index=[t for t in X.columns if (t not in ground_truth.index)], columns=ground_truth.columns)
        gt_scores = pd.concat((gt_scores, gt_missing), axis=0).fillna(0).loc[X.columns].astype(int)
    else:
        gt_scores = pd.DataFrame([], index=X.columns, columns=["Ground Truth"]).fillna(0)
    row_colors = pd.Series([colors[i] for i in list(gt_scores[gt_scores.columns[0]])], index=list(gt_scores.index))
    im = sns.clustermap(X.corr(method="pearson"), cmap="Blues", figsize=(w,h), row_cluster=True, col_cluster=True, 
	row_colors=row_colors, col_colors=row_colors, vmin=-1, vmax=1)
    im.ax_heatmap.axes.set_yticklabels([t.get_text() for t in im.ax_heatmap.axes.get_yticklabels()], fontsize=fsize)
    im.ax_heatmap.axes.set_xticklabels([t.get_text() for t in im.ax_heatmap.axes.get_xticklabels()], fontsize=fsize, rotation=rot)
    for ti, tick_label in enumerate(im.ax_heatmap.axes.get_yticklabels()):
        tick_label.set_color(colors[gt_scores.loc[tick_label.get_text()][gt_scores.columns[0]]])
    for ti, tick_label in enumerate(im.ax_heatmap.axes.get_xticklabels()):
        tick_label.set_color(colors[gt_scores.loc[tick_label.get_text()][gt_scores.columns[0]]])
    cbar = im.ax_heatmap.axes.collections[0].colorbar
    cbar.ax.tick_params(labelsize=bfsize)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def plot_roc_curve(pr, prs, tr, fname="ROC.pdf", method_name="predictor", fsize=18):
    '''
    Plots a ROC curve (with variations across samples)

    ...

    Parameters
    ----------
    pr : Pandas DataFrame
        rows/[drug names] x column/[value]
    prs : Pandas DataFrame
        rows/[drug names] x columns/[patient samples]
    tr : Pandas DataFrame
        [default=None] : rows/[drug names] x column/[class] 
    fname : Python character string
        [default="ROC.pdf"] : file name for the plot
    method_name : Python character string
        [default="predictor"] : name of the predictor
    fsize : Python integer
        [default=18] : font size


    Returns
    ----------
    None
        Plots a ROC curve based on the drug repurposing predictions
    '''
    from sklearn.metrics import roc_curve, roc_auc_score
    from scipy import interp
    predicted = pr.values.tolist()
    truth = tr.loc[pr.index].values.tolist()
    ids = np.argsort(predicted).tolist()
    ids.reverse()
    sorted_rewards = [predicted[i] for i in ids]
    sorted_ground_truth = [int(truth[i]>0) for i in ids]
    AUC = roc_auc_score(sorted_ground_truth, sorted_rewards)
    base = np.linspace(0, 1, 101)
    tprs, tprs2 = [], []
    aucs = []
    for idx in prs.columns:
        predicted_patient = prs.loc[pr.index][idx].values.tolist()
        fper, tper, _ = roc_curve(truth, predicted_patient)
        aucs.append(roc_auc_score(truth, predicted_patient))
        tpr = interp(base, fper, tper)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
    plt.plot(base, mean_tprs, 'r', alpha = 0.8, label="%s (AUC=%.2f)" % (method_name.split("_")[0],AUC))
    plt.fill_between(base, tprs_lower, tprs_upper, color = 'red', alpha = 0.2)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', alpha=0.8)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel('False Positive Rate', fontsize=fsize)
    plt.ylabel('True Positive Rate', fontsize=fsize)
    plt.title('Receiver Operating Characteristic Curve', size=fsize)
    plt.legend(loc="lower right", fontsize=fsize)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def plot_precision_recall(pr, prs, tr, beta=1, thres=0.5, fname="PRC.pdf", method_name="predictor", fsize=18):
    '''
    Plots a Precision-Recall curve (with variations across samples)

    ...

    Parameters
    ----------
    pr : Pandas DataFrame
        rows/[drug names] x column/[value]
    prs : Pandas DataFrame
        rows/[drug names] x columns/[patient samples]
    tr : Pandas DataFrame
        [default=None] : rows/[drug names] x column/[class] 
    beta : Python float
        [default=1] : value of coefficient beta for the F-measure
    thres : Python float
        [default=0.5] : decision threshold
    fname : Python character string
        [default="PRC.pdf"] : file name for the plot
    method_name : Python character string
        [default="predictor"] : name of the predictor
    fsize : Python integer
        [default=18] : font size


    Returns
    ----------
    None
        Plots a Precision-Recall curve based on the drug repurposing predictions
    '''
    from sklearn.metrics import precision_recall_curve, fbeta_score
    from scipy import interp
    predicted = pr.values.tolist()
    truth = tr.loc[pr.index].values.tolist()
    ids = np.argsort(predicted).tolist()
    ids.reverse()
    sorted_rewards = [int(predicted[i]>thres) for i in ids]
    sorted_ground_truth = [int(truth[i]>0) for i in ids]
    f = fbeta_score(sorted_ground_truth, sorted_rewards, average='weighted', beta=beta)
    base = np.linspace(0, 1, 101)
    recs, recs2 = [], []
    for idx in prs.columns:
        predicted_patient = prs.loc[pr.index][idx].values.tolist()
        pres, rec, _ = precision_recall_curve(truth, predicted_patient)
        rec = interp(base, pres, rec)
        recs.append(rec)
    recs = np.array(recs)
    mean_recs = recs.mean(axis=0)
    std_recs = recs.std(axis=0)
    recs_upper = np.minimum(mean_recs + std_recs, 1)
    recs_lower = np.maximum(mean_recs - std_recs, 0)
    plt.plot(base, mean_recs, 'b', alpha=0.8, label=r'%s (F_%.1f = %.2f)' % (method_name.split("_")[0], beta, f))
    plt.fill_between(base, recs_lower, recs_upper, color="blue", alpha=0.2)
    plt.plot([0,1], [1,0], linestyle="--", lw=2, color="blue", alpha=0.8)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel('Precision', fontsize=fsize)
    plt.ylabel('Recall', fontsize=fsize)
    plt.title('Precision-Recall Curve', fontsize=fsize)
    plt.legend(loc='lower left', fontsize=fsize)
    plt.savefig(fname,bbox_inches="tight")
    plt.close()
