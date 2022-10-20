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
    rc("ytick", labelsize=5)
    rc("xtick", labelsize=10)
    sigs = pd.DataFrame(signatures.values, index=signatures.index, columns=signatures.columns)
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

#' @param profiles DataFrame from get_experimental_constraints
#' @param fname name of image
#' @return None
def plot_distributions(profiles, fname="gene_expression_distribution.png", thres=None):
    bp = profiles.iloc[:-3,:].T.apply(pd.to_numeric).boxplot(rot=90, figsize=(25,15))
    if (str(thres)!="None"):
        K = profiles.shape[0]-3
        for t in [thres, 1-thres]:
            plt.plot(list(range(K)), [t]*K, "r--")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return None

def plot_discrete_distributions(signatures, fname="signature_expression_distribution.png"):
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
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return None
