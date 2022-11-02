#coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def influences2graph(influences, fname, optional=False, compile2png=True, engine=["sfdp","dot"][0]):
    '''
        Plots a network by conversion to a DOT file and then to PNG
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes], contains {-1,1,2}
        @param\tfname\tPython character string: filename of png file
        @param\toptional\tPython bool[default=False]: should interactions be drawn as optional (dashed lines)?
        @return\tNone\t
    '''
    dotfile = fname+".dot"
    filename = fname+".png"
    graph = ["digraph {"]
    for idx in influences.columns:
        if (len(idx)>0):
            graph += [idx+" [shape=circle,fillcolor=grey,style=filled];"]
    directed, undirected = ["subgraph D {"], ["subgraph U {"]
    in_undirected = []
    for x,y in np.argwhere(influences.values != 0).tolist():
        nx, ny = influences.index[x], influences.columns[y]
        sign = influences.values[x,y]
        if ((y,x,sign) in in_undirected):
            continue
        edge = " ".join([nx, "->",ny])
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

def plot_influence_graph(network_df, input_col, output_col, sign_col, fname="graph.png", optional=True):
    '''
        Converts a network into a PNG picture
        @param\tnetwork_df\tPandas DataFrame: rows/[index] x columns/[@input_col,@output_col,@sign_col]
        @param\tinput_col,output_col,sign_col\tPython character string: columns of @network_df
        @param\tfname\tPython character string[default="graph.png"]: file name for PNG picture
        @param\toptional\tPython bool[default=True]: should edges be plotted as dashed lines?
        @return\tNone\t
    '''
    influences = network_df.pivot_table(index=input_col, columns=output_col, values=sign_col, aggfunc='mean')
    influences[influences==0] = 2
    influences = influences.fillna(0)
    influences[influences<0] = -1
    influences[(influences>0)&(influences<2)] = 1
    for g in influences.index:
        if (g not in influences.columns):
            influences[g] = 0
    for g in influences.columns:
        if (g not in influences.index):
            influences.loc[g] = 0
    influences = influences.astype(int)
    influences.index = influences.index.astype(str)
    influences.columns = influences.columns.astype(str)
    influences2graph(influences, fname, optional=optional)
    return None

def plot_signatures(signatures, width=10, height=10, max_show=50, fname="signatures"):
    '''
        Print signatures
        @param\tsignatures\tPandas DataFrame: rows/[genes] x columns/[signature IDs]
        @param\twidth, height\tPython integer[default=10,default=10]: dimensions of image
        @param\tmax_show\tPython integer[default=50]: maximum number of genes shown (as only the @max_show genes with highest variance across signatures are plotted)
        @param\tfname\tPython character string[default="signatures"]: path of resulting PNG image
        @return\tNone\t
    '''
    from matplotlib import colors,rc
    from copy import deepcopy
    rc("ytick", labelsize=5)
    rc("xtick", labelsize=10)
    sigs = deepcopy(signatures)
    sigs[sigs == 0] = -1
    sigs[pd.isnull(sigs)] = 0 
    max_genes = np.argsort(np.var(sigs.values, axis=1))[-max_show:]
    max_genes = sigs.index[max_genes]
    selected_genes = list(set([y for y in [s.split("_")[0] for s in signatures.columns] if (y != "initial")]))
    for g in selected_genes:
        if (g not in sigs.index):
            sigs.loc[g] = 0
    max_genes = selected_genes+[g for g in max_genes if (g not in selected_genes)]
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
        @param\tprofiles\tPandas DataFrame: rows/[genes+annotations] x columns/[samples]
        @param\tfname\tPython character string[default="gene_expression_distribution.png"]
        @param\tthres\tPython float or None[default=None]: binarization threshold
        @return\tNone\t
    '''
    bp = profiles.iloc[:-3,:].T.apply(pd.to_numeric).boxplot(rot=90, figsize=(25,15))
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
        @param\tsignatures\tPandas DataFrame: rows/[genes] x columns/[samples] with values in {0,NaN,1}. Determined status is either 0 or 1.
        @param\tfname\tPython character string[default="signature_expression_distribution.png"]
        @return\tNone\t
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

ffsize=18 #fontsize on plots
french=False

def plot_roc_curve(prs, tr, name, AUC, prs2=None, name2=None, AUC2=None, french=french):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from scipy import interp
    base = np.linspace(0, 1, 101)
    tprs, tprs2 = [], []
    aucs = []
    for i in range(prs.shape[0]):
        fper, tper, _ = roc_curve(tr, prs[i,:].flatten())
        aucs.append(roc_auc_score(tr,prs[i,:].flatten()))
        tpr = interp(base, fper, tper)
        tpr[0] = 0.0
        tprs.append(tpr)
        if ((prs2 is not None) and (i in range(prs2.shape[0]))):
            fper2, tper2, _ = roc_curve(tr, prs2[i,:].flatten())
            tpr2 = interp(base, fper2, tper2)
            tpr2[0] = 0.0
            tprs2.append(tpr2)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
    plt.plot(base, mean_tprs, 'r', alpha = 0.8, label="%s (AUC=%.2f)" % (name.split("_")[0],AUC))
    plt.fill_between(base, tprs_lower, tprs_upper, color = 'red', alpha = 0.2)
    #plt.plot(fper, tper, color='red', label='ROC')
    if (prs2 is not None):
        tprs2 = np.array(tprs2)
        mean_tprs2 = tprs2.mean(axis=0)
        std_tprs2 = tprs2.std(axis=0)
        tprs_upper2 = np.minimum(mean_tprs2 + std_tprs2, 1)
        tprs_lower2 = np.maximum(mean_tprs2 - std_tprs2, 0)
        plt.plot(base, mean_tprs2, 'g', alpha = 0.8, label="%s (AUC=%.2f)" % (name2.split("_")[0], AUC2))
        plt.fill_between(base, tprs_lower2, tprs_upper2, color = 'green', alpha = 0.2)
    if (name2 is None):
        print(name)
        #print(np.round(AUC,2))
        print("%.2f +- %.2f" % (np.mean(aucs), np.sqrt(np.var(aucs))))
        print("____")
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', alpha=0.8)
    plt.xticks(fontsize=ffsize)
    plt.yticks(fontsize=ffsize)
    plt.xlabel('Taux de faux positifs' if (french) else 'False Positive Rate', fontsize=ffsize)
    plt.ylabel('Taux de vrais positifs' if (french) else 'True Positive Rate', fontsize=ffsize)
    plt.title("Fonction d'efficacité du récepteur" if (french) else 'Receiver Operating Characteristic Curve', size=ffsize)
    plt.legend(loc="lower right", fontsize=ffsize)
    plt.savefig(plot_folder+"ROC"+("_"+name if (prs2 is None) else ("_"+use_small_dataset))+".pdf", bbox_inches="tight")
    plt.close()

def plot_precision_recall(prs,tr,name,f,prs2=None, name2=None, f2=None, french=french):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve as PRC
    from scipy import interp
    base = np.linspace(0, 1, 101)
    recs, recs2 = [], []
    for i in range(prs.shape[0]):
        pres, rec, _ = PRC(tr, prs[i,:].flatten())
        rec = interp(base, pres, rec)
        recs.append(rec)
        if ((prs2 is not None) and (i in range(prs2.shape[0]))):
            pres2, rec2, _ = PRC(tr, prs2[i,:].flatten())
            rec2 = interp(base, pres2, rec2)
            recs2.append(rec2)
    recs = np.array(recs)
    mean_recs = recs.mean(axis=0)
    std_recs = recs.std(axis=0)
    recs_upper = np.minimum(mean_recs + std_recs, 1)
    recs_lower = np.maximum(mean_recs - std_recs, 0)
    plt.plot(base, mean_recs, 'r', alpha=0.8, label=r'%s' % (name.split("_")[0]))#' ($F_{1}=%.2f$)' % (name.split("_")[0],f))
    plt.fill_between(base, recs_lower, recs_upper, color="red", alpha=0.2)
    if (prs2 is not None):
        recs2 = np.array(recs2)
        mean_recs2 = recs2.mean(axis=0)
        std_recs2 = recs2.std(axis=0)
        recs_upper2 = np.minimum(mean_recs2 + std_recs2, 1)
        recs_lower2 = np.maximum(mean_recs2 - std_recs2, 0)
        plt.plot(base, mean_recs2, 'g', alpha=0.8, label='%s' % (name2.split("_")[0]))#' ($F_{1}=%.2f$)' % (name2.split("_")[0],f2))
        plt.fill_between(base, recs_lower2, recs_upper2, color="green", alpha=0.2)
    plt.plot([0,1], [1,0], linestyle="--", lw=2, color="blue", alpha=0.8)
    plt.xticks(fontsize=ffsize)
    plt.yticks(fontsize=ffsize)
    plt.xlabel('Spécificité (precision)' if (french) else 'Precision', fontsize=ffsize)
    plt.ylabel('Sensibilité (recall)' if (french) else 'Recall', fontsize=ffsize)
    plt.title('Courbe sensibilité-spécificité' if (french) else 'Precision-Recall Curve', fontsize=ffsize)
    plt.legend(loc='upper right', fontsize=ffsize)
    plt.savefig(plot_folder+"PRC"+("_"+name if (prs2 is None) else ("_"+use_small_dataset))+".pdf",bbox_inches="tight")
    plt.close()

## Boxplots to compare ground truth to predictions
def boxplots(treatment_scores, scores, ranking_by="mean", fsize=ffsize, msize=5, thres=0., name="boxplots", french=french):
    assert ranking_by in ["mean", "median"]
    import matplotlib.pyplot as plt
    ## Get scores
    nnames = [x for x in list(treatment_scores.index) if (x in scores.index)]
    names = [scores.loc[x]["drug_name"] for x in nnames]
    rewards = treatment_scores.loc[nnames].values
    ## Order scores
    if (rewards.shape[1]==1):
        rewards = rewards.T
        rewards_list = rewards.tolist()[0]
        ids = np.argsort(rewards_list).tolist()
        ids.reverse()
        sorted_rewards = rewards[:,ids]
    else:
        rewards_list = eval("np."+ranking_by)(rewards, axis=1).flatten().tolist()
        rewards = rewards.T
    ids = np.argsort(rewards_list).tolist()
    ids.reverse()
    sorted_rewards = rewards[:,ids]
    scores_list = scores[['score']].loc[nnames].values.flatten().tolist()
    positive = [i+1 for i in range(len(ids)) if (scores_list[ids[i]] > 0)]
    negative = [i+1 for i in range(len(ids)) if (scores_list[ids[i]] < 0)]
    ## Compute metrics
    if (len(np.unique(scores.values[:,1]))>1):
        res_di = compute_metrics(rewards_list, scores_list, K=[2,5,10], use_negative_class=False, thres=thres)
    else:
        res_di = {}
    ## Plot boxplot
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))#(12,12))#(50, 15))
    medianprops = {}#dict(linestyle='-', linewidth=5, color='lightcoral')
    meanpointprops = {}#dict(marker='.', markersize=1, markerfacecolor='white', alpha=0.)
    sorted_rewards_list = [rewards_list[i] for i in ids]
    sorted_ground_truth = [scores_list[i] for i in ids]
    labels=[names[i] for i in ids]
    ax.boxplot(sorted_rewards, showmeans=False, vert=True, meanline=False, meanprops=meanpointprops, notch=False, medianprops=medianprops)
    sorted_rewards_ = [np.median(rewards.T, axis=1).flatten().tolist()[i] for i in ids]
    ax.plot(range(1, len(names)+1), sorted_rewards_, "k-", label=ranking_by)
    #ax.plot(range(1,len(names)+1), sorted_rewards_list, 'k-', label=ranking_by)
    ax.plot(positive, [rewards_list[ids[i-1]] for i in positive], 'gD', label="score > 0", markersize=msize)
    ax.plot(negative, [rewards_list[ids[i-1]] for i in negative], 'rD', label="score < 0", markersize=msize)
    ax.set_ylabel("Score", fontsize=fsize)
    ax.set_xlabel("Drug" if (not french) else "Traitement", fontsize=fsize)
    ax.set_xticklabels(["%s (%.2f)" % (names[idx], rewards_list[idx]) for idx in ids], rotation=90, fontsize=fsize)
    plt.yticks(rotation=0)
    cols=["r" if (i in negative) else ("g" if (i in positive) else "k") for i in range(1, len(names)+1)]
    for xtick, color in zip(ax.get_xticklabels(), cols):
        xtick.set_color(color)
    ax.tick_params(axis='y', which='major', labelsize=fsize)
    #ax.set_title("Boxplots of rewards from GRN across "+str(treatment_scores.shape[1])+" patient"+("s" if (treatment_scores.shape[1]>1) else "")+"\n(versus known associations/scores for "+str(len(names))+" drugs)", fontsize=fsize)
    plt.savefig(plot_folder+"boxplot_"+name+".pdf", bbox_inches="tight")
    plt.close()
    if (len(np.unique(scores.values[:,1]))>1):
        print(pd.DataFrame({name: res_di}))
        pd.DataFrame({name: res_di}).to_csv(file_folder+"results-DR_"+name+".csv")
        ## Take into account possible variation to patients
        plot_roc_curve(rewards, scores_list, name, AUC=res_di["AUC"])
        plot_precision_recall(rewards, scores_list, name, res_di["F"])
    return rewards, scores_list, name, res_di

def plot_phenotypes():
    ## Plot control vs patient sample + CD signature
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    plt.figure(figsize=(8,8))
    xx = np.linspace(np.min(X_pca[:,0]), np.max(X_pca[:,1]), num=500)
    yy = -a*xx/b
    plt.scatter(X_pca[:ctrl2.shape[1],0], X_pca[:ctrl2.shape[1],1], c="g", label="Control")
    plt.scatter(X_pca[ctrl2.shape[1]:,0], X_pca[ctrl2.shape[1]:,1], c="r", label="Patient")
    ## Plot Frontier
    plt.plot(yy,-xx,"b-",label="Frontier")
    plt.plot(xx,yy,"b--",label="CD signature")
    plt.title("PCA Controls (N=%d), Patients (N=%d), Signature" % (ctrl2.shape[1], ptt2.shape[1]))
    plt.xlabel("PCA1 (explained variance "+str(np.round(pca.explained_variance_ratio_[0]*100, 1))+"%)")
    plt.ylabel("PCA2 (explained variance "+str(np.round(pca.explained_variance_ratio_[1]*100, 1))+"%)")
    plt.legend()
    plt.savefig(plot_folder+"pca_controls_patients.png", bbox_inches="tight")
    plt.close()


def plot_results():
    ## PLOT RESULTS (projection & point)
    plt.figure(figsize=(8,8))
    X_c_, proj_X_c_, sc_c_ = prediction_score(ctrl2,ctrl2,genes_in_network, return_full=True)
    X_p_, proj_X_p_, sc_p_ = prediction_score(ptt2, ptt2, genes_in_network, return_full=True)
    #X_c_,proj_X_c_,X_p_,proj_X_p_ = [pca.transform(scaler.transform(x)) for x in [X_c_,proj_X_c_,X_p_,proj_X_p_]]
    print("Score mean (control): %.2f\tScore mean (patient): %.2f" % (np.mean(sc_c_), np.mean(sc_p_)))
    plt.scatter(X_pca[:ctrl2.shape[1],0], X_pca[:ctrl2.shape[1],1], c="g", label="Control")
    plt.scatter(X_pca[ctrl2.shape[1]:,0], X_pca[ctrl2.shape[1]:,1], c="r", label="Patient")
    #for i in range(ctrl2.shape[1],X_pca.shape[1]):
    #    plt.text(X_pca[i,0],X_pca[i,1],str(np.round(sc_c_[i],2)),fontsize=2)
    ## Should overlap points
    #plt.scatter(X_c_[:,0], X_c_[:,1], c="m", label="Control points")
    #plt.scatter(X_p_[:,0], X_p_[:,1], c="y", label="Patient points")
    ## Should belong to the Frontier
    plt.scatter(proj_X_c_[:,0], proj_X_c_[:,1], c="c", label="Projection of controls")
    plt.scatter(proj_X_p_[:,0], proj_X_p_[:,1], c="k", label="Projection of patients")
    ## Plot Frontier
    plt.plot(yy,-xx, "b-", label="Frontier")
    plt.plot(xx,yy, "b--", label="CD Signature")
    plt.title("PCA Controls (N=%d), Patients (N=%d), Projections" % (ctrl2.shape[1], ptt2.shape[1]))
    plt.xlabel("PCA1 (explained variance "+str(np.round(pca.explained_variance_ratio_[0]*100, 1))+"%)")
    plt.ylabel("PCA2 (explained variance "+str(np.round(pca.explained_variance_ratio_[1]*100, 1))+"%)")
    plt.legend()
    plt.savefig(plot_folder+"pca_controls_patients_projection.png", bbox_inches="tight")
    plt.close()

