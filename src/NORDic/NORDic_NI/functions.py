#coding:utf-8

from random import seed as rseed
import os
from subprocess import call as sbcall
from subprocess import check_output as sbcheck_output
import pandas as pd
import numpy as np
from copy import deepcopy
from glob import glob

from NORDic.UTILS.DISGENET_utils import get_genes_proteins_from_DISGENET, get_user_key_DISGENET
from NORDic.UTILS.STRING_utils import get_network_from_STRING, get_app_name_STRING
from NORDic.UTILS.LINCS_utils import get_user_key
from NORDic.UTILS.utils_grn import get_genes_interactions_from_PPI, build_influences, create_grn, build_observations, infer_network, zip2df
from NORDic.UTILS.utils_grn import get_minimal_edges, general_topological_parameter, get_weakly_connected, solution2influences, save_grn, get_grfs_from_solution
from NORDic.UTILS.utils_exp import get_experimental_constraints, profiles2signatures
from NORDic.UTILS.utils_data import convert_genes_EntrezGene, convert_EntrezGene_LINCSL1000, get_all_celllines
from NORDic.UTILS.utils_plot import plot_influence_graph, plot_signatures, influences2graph

from NORDic.UTILS.utils_grn import get_grfs_from_solution
from NORDic.NORDic_NI.cytoscape_style import style_file

def network_identification(file_folder, taxon_id, path_to_genes=None, disgenet_args=None, network_fname=None, string_args=None, experiments_fname=None, lincs_args=None, edge_args=None, sig_args=None, bonesis_args=None, weights=None, seed=0, njobs=1, force_experiments=True, accept_nonRNA=False, preserve_network_sign=True):
    '''
        Generates or retrieves the optimal network model
        @param\tfile_folder\tPython character string: path to which files should be saved
        @param\ttaxon_id\tPython integer: NCBI Taxonomy ID for the considered species
        @param\tpath_to_genes\tPython character string or None[default=None]: path to the file containing gene names (one per line)
        @param\tdisgenet_args\tPython dictionary or None[default=None]: arguments to the DisGeNET API (see test files)
        @param\tnetwork_fname\tPython character string or None[default=None]: path to the file containing the prior knowledge network (see test files)
        @param\tstring_args\tPython dictionary or None[default=None]: arguments to the STRING API (see test files)
        @param\texperiments_fname\tPython character string or None[default=None]: path to the file containing the matrix of gene expression data (genes x samples) (see test files for format)
        @param\tlincs_args\tPython dictionary or None[default=None]: arguments to the LINCS L1000 API (see test files)
        @param\tedge_args\tPython dictionary or None[default=None]: arguments to the processing of edges (see test files)
        @param\tsig_args\tPython dictionary or None[default=None]: arguments to the processing of signatures (see test files)
        @param\tbonesis_args\tPython dictionary or None[default=None]: arguments to building constraints and generating solutions using BoneSIS (see test files)
        @param\tweights\tPython dictionary or None[default=None]: weights to the optimal model selection procedure (see test files)
        @param\tseed\tPython integer[default=0]
        @param\tnjobs\tPython integer[default=1]
        @param\tnjobs\tPython integer[default=1]
        @param\tforce_experiments\tPython bool[default=True]: if set to True, returns an error if no experimental profile associated with the genes is found
        @param\taccept_nonRNA\tPython bool[default=False]: if set to False, ignores gene names which cannot be converted to EntrezIDs or which are not present in LINCS L1000
        @return\tsolution\tPython character string: optimal model selected by the procedure
    '''
    solution_fname, nsol = solution_generation(file_folder, taxon_id, path_to_genes=path_to_genes, disgenet_args=disgenet_args, network_fname=network_fname, string_args=string_args, experiments_fname=experiments_fname, lincs_args=lincs_args, edge_args=edge_args, sig_args=sig_args, bonesis_args=bonesis_args, seed=seed, njobs=njobs, force_experiments=force_experiments, accept_nonRNA=accept_nonRNA, preserve_network_sign=preserve_network_sign)

    if (nsol==0):
        return None

    print("--- SELECTION OF OPTIMAL MODEL ---")
    ###########################
    ## SELECT OPTIMAL MODEL  ##
    ###########################
    print("A. Select an optimal model based on a topological criteria... ", end=" ...")

    solutions = import_all_solutions(solution_fname)
    print("... %d solutions (%d/%d constant genes in average)" % (solutions.shape[1], np.mean([solutions.loc[(solutions[c].astype(str)=="0")|(solutions[c].astype(str)=="1")].shape[0] for c in solutions.columns]), solutions.shape[0]))
    visualize_models(solutions, file_folder)
    solution = select_optimal_model(solutions, weights, file_folder)

    return solution

def solution_generation(file_folder, taxon_id, path_to_genes=None, disgenet_args=None, network_fname=None, string_args=None, experiments_fname=None, lincs_args=None, edge_args=None, sig_args=None, bonesis_args=None, weights=None, seed=0, njobs=1, force_experiments=True, accept_nonRNA=False,preserve_network_sign=True):
    '''
        Generates or retrieves the optimal network model
        @param\tfile_folder\tPython character string: path to which files should be saved
        @param\ttaxon_id\tPython integer: NCBI Taxonomy ID for the considered species
        @param\tpath_to_genes\tPython character string or None: path to the file containing gene names (one per line)
        @param\tdisgenet_args\tPython dictionary or None: arguments to the DisGeNET API (see test files)
        @param\tnetwork_fname\tPython character string or None: path to the file containing the prior knowledge network (see test files)
        @param\tstring_args\tPython dictionary or None: arguments to the STRING API (see test files)
        @param\texperiments_fname\tPython character string or None: path to the file containing the matrix of gene expression data (genes x samples) (see test files for format)
        @param\tlincs_args\tPython dictionary or None: arguments to the LINCS L1000 API (see test files)
        @param\tedge_args\tPython dictionary or None: arguments to the processing of edges (see test files)
        @param\tsig_args\tPython dictionary or None: arguments to the processing of signatures (see test files)
        @param\tbonesis_args\tPython dictionary or None: arguments to building constraints and generating solutions using BoneSIS (see test files)
        @param\tseed\tPython integer[default=0]
        @param\tnjobs\tPython integer[default=1]
        @param\tnjobs\tPython integer[default=1]
        @param\tforce_experiments\tPython bool[default=True]: if set to True, returns an error if no experimental profile associated with the genes is found
        @param\taccept_nonRNA\tPython bool[default=False]: if set to False, ignores gene names which cannot be converted to EntrezIDs or which are not present in LINCS L1000
        @return\tsolution\tPython character string: optimal model selected by the procedure
    '''
    sbcall("mkdir -p "+file_folder, shell=True)
    solution_fname=file_folder+("SOLUTIONS-%d_binthres=%.3f_thresiscale=%s_score=%.2f_maxclause=%d" % (bonesis_args.get("limit", 1), sig_args.get("bin_thres", 0.5), str(lincs_args.get("thres_iscale", 0.)), string_args.get("score", 1), bonesis_args.get("max_maxclause", 5)))
    solution_fname_ls, solution_ls = glob(solution_fname+"_*.zip"), []
    #####################################
    ## RETRIEVE EXISTING RESULTS       ##
    #####################################
    if (len(solution_fname_ls)>0):
        for fname in solution_fname_ls:
            nbits = int(sbcheck_output("ls -l "+fname+"  | cut -d\" \" -f5", shell=True).decode("utf-8"))
            if (nbits==0):
                sbcall("rm "+fname, shell=True)
            else:
                solution_ls.append(fname)
    if (len(solution_ls)>=bonesis_args.get("niterations", 1)):
        print("%d solutions are already generated, and saved at %s_{1,...%d}.zip" % (len(solution_ls)*bonesis_args.get("limit", 1), solution_fname, len(solution_ls)))
        return solution_fname, len(solution_ls)

    np.random.seed(seed)
    rseed(seed)

    ## Create folder
    sbcall("mkdir -p "+file_folder, shell=True)

    print("--- DATA IMPORT ---")

    ###########################
    ## IMPORT GENE SET       ##
    ###########################
    print("1. Import gene set from %s" % ("DisGeNET" if (path_to_genes is None) else path_to_genes), end="... "),

    gene_set_file = file_folder+"GENESET.txt"
    if (path_to_genes is None and network_fname is None):
        ## Import from DisGeNET
        assert disgenet_args
        assert disgenet_args.get("credentials", False)
        assert disgenet_args.get("disease_cids", False)
        user_key = get_user_key_DISGENET(disgenet_args["credentials"])
        gene_df = get_genes_proteins_from_DISGENET(disgenet_args["disease_cids"], 
                            min_score=disgenet_args.get("min_score", 0), 
                            min_ei=disgenet_args.get("min_ei", 0),
                            min_dsi=disgenet_args.get("min_dsi", 0.25),
                            min_dpi=disgenet_args.get("min_dpi", 0),
                            user_key=user_key
        )
        model_genes = list(set([g for x in gene_df["Gene Name"] for g in x.split("; ")]))
    elif (path_to_genes is None):
        ## Import prepared network: four columns ["preferredName_A","preferred_B","sign","directed","score"], sep="\t"
        ## preferredName_A, preferredName_B are HUGO gene symbols
        ## score in [0,1], directed in {1}, sign in {1,-1} (duplicate edges which are not directed / signed)
        network = pd.read_csv(network_fname, sep="\t")
        cols = ["preferredName_A","preferredName_B","sign","directed","score"]
        assert network.shape[1] == len(cols)
        for coli, col in enumerate(network.columns):
            assert col == cols[coli]
        for col in cols:
            assert col in network.columns
        assert all([v in [1,2,-1] for v in list(network["sign"])])
        assert all([v in [1,0] for v in list(network["directed"])])
        assert all([v <= 1 and v >= 0 for v in list(network["score"])])
        model_genes = list(set(list(network["preferredName_A"])+list(network["preferredName_B"])))
    else:
        ## Import prepared gene set: one gene name per line
        assert os.path.exists(path_to_genes)
        with open(path_to_genes, "r") as f:
            model_genes = f.read().split("\n")[:-1]

    with open(gene_set_file, "w") as f:
        f.write(("\n".join(model_genes))+"\n")

    print("... %d genes imported." % len(model_genes))
    assert len(model_genes)>0

    ###########################
    ## IMPORT NETWORK        ##
    ###########################
    print("2. Import network from %s" % ("STRING" if (network_fname is None) else network_fname), end="... ")

    network_file = file_folder+"NETWORK.tsv"
    if (not os.path.exists(network_file)):
        if (network_fname is None):
            ## Import from STRING
            app_name = get_app_name_STRING(string_args["credentials"])
            network = get_network_from_STRING(model_genes, 
                            taxon_id,
                            min_score=string_args.get("score", 0), 
                            app_name=app_name,
                            quiet=True
            )
        else:
            ## Import prepared network: four columns ["preferredName_A","preferred_B","sign","directed","score"], sep="\t"
            ## preferredName_A, preferredName_B are HUGO gene symbols
            ## score in [0,1], directed in {1,0}, sign in {1,2,-1}
            network = pd.read_csv(network_fname, sep="\t")
            cols = ["preferredName_A","preferredName_B","sign","directed","score"]
            assert network.shape[1] == len(cols)
            for coli, col in enumerate(network.columns):
                assert col == cols[coli]
            for col in cols:
                assert col in network.columns
            assert all([v in [1,2,-1] for v in list(network["sign"])])
            assert all([v in [1,0] for v in list(network["directed"])])
            assert all([v <= 1 and v >= 0 for v in list(network["score"])])

        network.to_csv(network_file, index=None, sep="\t")
    network = pd.read_csv(network_file, sep="\t")
    network_genes = list(set([g for a in ["preferredName_A","preferredName_B"] for g in list(network[a])]))

    print("... %d edges in model (including %d directed edges) with a total of %d non-isolated genes" % (network.shape[0], network[network["directed"]==1].shape[0], len(network_genes)))

    ## Write down the list of missing genes in PPI
    if (len(model_genes)>len(network_genes)):
        with open(file_folder+"missing_genes_in_PPI.txt", "w") as f:
            missing_genes = [g for g in model_genes if (g not in network_genes)]
            f.write("\n".join(missing_genes))

    if (network.shape[0]==0):
        raise ValueError("No network could be found.")
    assert network.shape[0]>0
    model_genes = network_genes ## trim out isolated nodes

    ###########################
    ## IMPORT EXPERIMENTS    ##
    ###########################
    print("3. Import experiments from %s" % ("LINCS L1000" if (experiments_fname is None) else experiments_fname), end="... "),

    ## Get cell line names from LINCS L1000
    if (len(lincs_args.get("cell_lines", []))==0):
        user_key = get_user_key(lincs_args["credentials"])
        cell_lines = get_all_celllines(pert_inames, user_key)
    else:
        cell_lines = lincs_args["cell_lines"]

    print("\n\t%d cell lines are considered (%s)" % (len(cell_lines), cell_lines))
    if (force_experiments and (experiments_fname is None)):
        assert len(cell_lines)>0

    pert_df = None
    if (experiments_fname is None):
        ## Convert gene names in EntrezGene IDs
        entrezgene_fname=file_folder+"ENTREZGENES.csv"
        if (not os.path.exists(entrezgene_fname)):
            app_name = get_app_name_STRING(string_args["credentials"])
            probes = convert_genes_EntrezGene(model_genes, taxon_id, app_name=app_name)
            probes.to_csv(entrezgene_fname)
        probes = pd.read_csv(entrezgene_fname,index_col=0)

        if (probes.shape[0]>probes[probes["Gene ID"]!="-"].shape[0]):
            with open(file_folder+"missing_genes_in_EntrezID.txt", "w") as f:
                missing_genes = [g for g in probes.index if (g not in list(probes[probes["Gene ID"]!="-"].index))]
                f.write("\n".join(missing_genes))

        if (probes[probes["Gene ID"]=="-"].shape[0]>0 and not accept_nonRNA):
            print("\tNot found genes: %s" % str(list(probes[probes["Gene ID"]=="-"].index)))
        probes = probes[probes["Gene ID"]!="-"]
        print("\t%d genes available (convertable to EntrezIDs)" % probes.shape[0])
        if (probes.shape[0]<len(model_genes) and not accept_nonRNA):
            print("\t\tCheck that\n\t\t1. All input genes are HUGO Gene Symbols / Ensembl IDs / HGNC IDs / STRING IDs;\n\t\t2. The correct taxon id (%d) was provided." % taxon_id)

        ## Convert EntrezGene IDs to LINCS L1000 symbols
        symbols_fname=file_folder+"SYMBOLS.csv"
        if (not os.path.exists(symbols_fname)):
            user_key = get_user_key(lincs_args["credentials"])
            pert_df = convert_EntrezGene_LINCSL1000(file_folder, list(probes["Gene ID"]), user_key)
            pert_df.to_csv(symbols_fname)
        pert_df = pd.read_csv(symbols_fname, index_col=0)

        missing_genes = [list(probes.index)[list(probes["Gene ID"]).index(g)] for g in pert_df.index if (g not in pert_df[pert_df["Gene Symbol"]!="-"].index)]
        pert_df = pert_df[pert_df["Gene Symbol"]!="-"]
        pert_inames = list(pert_df["Gene Symbol"])
        pert_df.index = pert_inames
        entrez_ids = list(pert_df["Entrez ID"])
        print("\t\t%d/%d genes retrieved in LINCS L1000" % (pert_df.shape[0], probes.shape[0]))

        if (len(missing_genes)>0):
            with open(file_folder+"missing_genes_in_LINCSL1000.txt", "w") as f:
                f.write("\n".join(missing_genes))

    add_rows_profiles = ["annotation", "perturbed", "perturbation", "cell_line", "sigid", "interference_scale"]
    thres_iscale = lincs_args.get("thres_iscale", None)
    profiles_fname = file_folder+"PROFILES_"+"-".join(cell_lines[:4])+"_thres=%s.csv" % str(thres_iscale)
    if (not os.path.exists(profiles_fname)):
        if (experiments_fname is None):
            full_profiles_fname = file_folder+"FULL_PROFILES_"+"-".join(cell_lines[:4])+".csv"
            assert string_args
            assert string_args.get("credentials", False)
            assert lincs_args
            assert lincs_args.get("credentials", False)

            ## Get experimental profiles
            pert_types = lincs_args.get("pert_types", ["trt_sh","trt_oe","trt_xpr"])
            selection = lincs_args.get("selection", "distil_ss")
            path_to_lincs = lincs_args.get("path_to_lincs", "./")
            nsigs = lincs_args.get("nsigs", 2)
            user_key = get_user_key(lincs_args["credentials"])
            pert_di = pert_df[["Entrez ID","Gene Symbol"]].to_dict()["Entrez ID"]
            if (not os.path.exists(full_profiles_fname)):
                full_profiles = get_experimental_constraints(file_folder, cell_lines, pert_types, pert_di, taxon_id, selection, user_key, path_to_lincs, thres_iscale=thres_iscale, nsigs=nsigs, quiet=False)
                full_profiles.to_csv(full_profiles_fname)
            else:
                full_profiles = pd.read_csv(full_profiles_fname, index_col=0, low_memory=False)

            if (full_profiles.shape[1]==0):
                profiles = None
            else:
                if (thres_iscale is not None):
                    profiles_ = full_profiles[[c for ic, c in enumerate(full_profiles.columns) if (float(list(full_profiles.loc["interference_scale"])[ic])>thres_iscale)]]
                    if (profiles_.shape[1]==0):
                        raise ValueError("The selected value of thres_iscale (%s) is too large, maximum value in collected experiments=%s" % (thres_iscale, np.max(full_profiles.loc["interference_scale"].values)))
                    min_iscale = np.min(profiles_.loc["interference_scale"].values)
                else:
                    profiles_ = full_profiles
                    min_iscale = None
                profiles = profiles_.loc[[i for i in profiles_.index if (i != "interference_scale")]]

                Nfullprofs = len(list(set(["_".join(full_profiles[c].loc[["perturbed", "perturbation", "cell_line"]]) for c in full_profiles.columns])))
                Nprofs = len(list(set(["_".join(profiles[c].loc[["perturbed", "perturbation", "cell_line"]]) for c in profiles.columns])))
                print("\t\t%d unique experiments (including %d with criterion thres_iscale > %s, min_value %s)" % (Nfullprofs, Nprofs, thres_iscale, min_iscale))
                profiles = profiles.loc[~profiles.index.duplicated()]
                profiles.index = [g if (g in add_rows_profiles) else pert_df.loc[pert_df["Entrez ID"]==float(g)]["Gene Symbol"].values[0] for g in list(profiles.index)]

            if (profiles is None):
                profiles = pd.DataFrame([], index=pert_inames, columns=["0","1"]).astype(str)
                for a in add_rows_profiles:
                    profiles.loc[a] = ["nan"]*profiles.shape[1]

        else:

            profiles = pd.read_csv(experiments_fname, sep=",", index_col=0)
            ## Pandas DataFrame with index=HUGO gene symbols, columns=sample names, and 5 additional rows
            ## 'annotation' 2 if the sample is treated, 1 otherwise
            ## 'perturbed' gene which is perturbed
            ## 'perturbation' experiment UNIQUE identifier, common to samples from the same experiment
            ## 'cell_line' experiment UNIQUE identifier, common to samples from the same experiment
            ## 'sigid' sample unique identifier from LINCS L1000 (can be filled with NaNs)
            ## /!\ SHOULD ALREADY BE BINARIZED
            assert "annotation" in list(profiles.index) and all([v in ["1", "2"] for v in list(profiles.loc["annotation"])])
            assert "perturbed" in list(profiles.index)
            assert "perturbation" in list(profiles.index) and all([v in ["KD", "OE", "None"] for v in list(profiles.loc["perturbation"])])
            assert "cell_line" in list(profiles.index)
            assert "sigid" in list(profiles.index)
            assert all([v in ["0","1",np.nan] for v in list(profiles.loc[[g for g in model_genes if (g in profiles.index)]].values.flatten())])
            profiles = profiles[[c for c in profiles.columns if ((profiles.loc["perturbed"][c] in model_genes+["None"]) and (profiles.loc["cell_line"][c] in cell_lines))]]
            profiles = profiles.loc[[g for g in list(profiles.index) if (g in model_genes+add_rows_profiles)]]
        profiles.to_csv(profiles_fname)

    profiles = pd.read_csv(profiles_fname, index_col=0, low_memory=False)
    if (not pd.isnull(profiles.values).all()):
        nconds = len(set(["_".join([profiles.loc[v][c] for v in ["perturbed","perturbation","cell_line"]]) for c in profiles.columns]))
    else:
        nconds = 0
    ngenes = len([p for p in profiles.index if (p not in add_rows_profiles)])
    print("... %d genes in %d profiles (%d experiments)" % (0 if (profiles is None) else ngenes, 0 if (profiles is None) else profiles.shape[1], 0 if (profiles is None) else nconds))
    if (profiles is None and force_experiments):
        raise ValueError("No experimental profile could be found.")
        assert (not force_experiments) or ((profiles is not None) and profiles.shape[1]>0)
    assert (not force_experiments) or ((profiles is not None) and profiles.shape[0]-len(add_rows_profiles)>0)

    ## Write down the list of missing genes in profiles
    if ((profiles is not None) and (pert_df is not None) and (pert_df.shape[0]>ngenes)):
        with open(file_folder+"missing_genes_in_profiles.txt", "w") as f:
            missing_genes = [g for g in pert_inames if (g not in list(profiles.index))]
            f.write("\n".join(missing_genes))

    print("\n--- CONSTRAINT BUILDING ---")

    #####################################
    ## FILTERING OUT EDGES             ##
    #####################################
    print("1. Filtering out edges by minimum set of edges with highest score which preserve connectivity...", end=" ")

    network = pd.read_csv(network_file, sep="\t")

    ## Filter out edges which do not involve nodes which are both present in the experiments
    if (not accept_nonRNA and profiles is not None):
        test_in_profiles = np.vectorize(lambda x : x in [p for p in list(profiles.index) if (p not in add_rows_profiles)])
        network = network.loc[(test_in_profiles(network["preferredName_A"])&test_in_profiles(network["preferredName_B"]))]
        ## Write down the list of missing genes in profiles
        genes_in_network = list(set(list(network["preferredName_A"])+list(network["preferredName_B"])))
        if (len(model_genes)>len(genes_in_network)):
            with open(file_folder+"missing_genes_in_filtered_network.txt", "w") as f:
                missing_genes = [g for g in list(profiles.index) if (g not in genes_in_network+add_rows_profiles)]
                f.write("\n".join(missing_genes))
        model_genes = genes_in_network
    ## Ensure that all considered edges are unique
    network = network.drop_duplicates(keep="first")
    print("... %d unique edges involving genes both in experiments (%d genes in total)" % (network.shape[0], len(model_genes)))
    plot_influence_graph(network, "preferredName_A", "preferredName_B", "sign", "directed", file_folder+"network", True)

    score_thres = 0 if (string_args is None) else string_args.get("score", 0)
    edges_file = file_folder+"EDGES_score=%f.tsv" % (score_thres)
    if (not os.path.exists(edges_file) and ((network["sign"]==2).any() or (network["directed"]==0).any())):
        network_df = get_genes_interactions_from_PPI(network, connected=(edge_args is not None and edge_args.get("connected", True)), score=score_thres, filtering=(edge_args is not None and edge_args.get("filter", True)))
        network_df.to_csv(edges_file, sep="\t", index=None)
    elif (not os.path.exists(edges_file)):
        network_df = deepcopy(network)
        network_df["sscore"] = np.multiply(network_df["sign"], network_df["score"])
        network_df = network_df[["preferredName_A","preferredName_B", "sscore"]]
        network_df.columns = ["Input", "Output", "SSign"]
        network_df.to_csv(edges_file, sep="\t", index=None)
    network_df = pd.read_csv(edges_file, sep="\t")
    ## Restrict genes to non-isolated genes
    model_genes = [m for m in list(set(list(network_df["Input"])+list(network_df["Output"]))) if ((profiles is None) or (m in model_genes))]

    print("... score_STRING %f\t%d genes (non isolated in PPI)\t%d edges in PPI" % (string_args["score"], len(model_genes), network_df.shape[0]))

    ## Restrict profiles to non isolated genes 
    if (profiles is not None):
        profiles = profiles.loc[[m for m in model_genes if (m in profiles.index)]+[a for a in add_rows_profiles if (a in profiles.index)]]

    ## Write down the list of missing genes in network
    ngenes = len([p for p in profiles.index if (p not in add_rows_profiles)])
    if ((profiles is not None) and (len(model_genes)>ngenes)):
        with open(file_folder+"missing_genes_in_network.txt", "w") as f:
            missing_genes = [g for g in model_genes if (g not in list(profiles.index))]
            f.write("\n".join(missing_genes))

    ###################################
    ## BUILD TOPOLOGICAL CONSTRAINTS ##
    ###################################
    print("2. Build topological constraints from filtered edges using gene expression data... ", end=" ...")

    influences_fname = file_folder+"INFLUENCES_"+"-".join(cell_lines[:4])+"_tau="+str(edge_args.get("tau", 0))+"_beta="+str(edge_args.get("beta", 1))+"_score_thres="+str(score_thres)+".csv"
    expr_df = None if (profiles is None) else profiles.loc[[g for g in profiles.index if (g not in add_rows_profiles)]].apply(pd.to_numeric)
    expr_df=expr_df if (not pd.isnull(expr_df.values).all()) else None
    if (not os.path.exists(influences_fname) and (network_fname is None or edge_args.get("tau", False))):
        influences = build_influences(network_df, edge_args.get("tau", 0), beta=edge_args.get("beta", 1), cor_method=edge_args.get("cor_method", "pearson"), expr_df=expr_df, accept_nonRNA=accept_nonRNA)
        influences.to_csv(influences_fname)
    elif (not os.path.exists(influences_fname)):
        if (preserve_network_sign or (expr_df is None)):
            ## use the sign given by the score in the existing network
            network_df.index = ["-".join(list(map(str, list(network_df.loc[idx][["Input","Output"]])))) for idx in network_df.index]
            influences = network_df.groupby(level=0).max()
            network_df.index = range(network_df.shape[0])
            influences = influences.pivot_table(index="Input",columns="Output",values="SSign", fill_value=0)
            missing_index = [g for g in influences.columns if (g not in influences.index)]
            if (len(missing_index)>0):
                missing_index_df = pd.DataFrame([], index=missing_index, columns=influences.columns).fillna(0)
                influences = pd.concat((influences, missing_index_df), axis=0)
            missing_columns = [g for g in influences.index if (g not in influences.columns)]
            if (len(missing_columns)>0):
                missing_columns_df = pd.DataFrame([], index=influences.index, columns=missing_columns).fillna(0)
                influences = pd.concat((influences, missing_columns_df), axis=1)
            assert influences.shape[0]==influences.shape[1]
            influences[influences<0] = -1
            influences[influences>0] = 1
        else:
            ## compute the sign based on the gene pairwise correlation on expression data
            influences = build_influences(network_df, 0, beta=edge_args.get("beta", 1), cor_method=edge_args.get("cor_method", "pearson"), expr_df=expr_df, accept_nonRNA=accept_nonRNA)
        influences.to_csv(influences_fname)
    influences = pd.read_csv(influences_fname, index_col=0, header=0)
    if (not pd.isnull(profiles.values).all() is not None):
        profiles = profiles.loc[[m for m in list(influences.index) if (m in profiles.index)]+[a for a in add_rows_profiles if (a in profiles.index)]]
    print("... %d negative, %d positive undirected interactions (%d edges in total), %d non isolated genes in experiments" % (np.sum(influences.values==-1), np.sum(influences.values==1), np.sum(influences.abs().values), influences.shape[0]))
    assert np.sum(influences.abs().values) > 0
    
    influences_df = influences.melt(ignore_index=False)
    influences_df["id"] = influences_df.index
    influences_df = influences_df[influences_df["value"]!=0]
    plot_influence_graph(influences_df, "id", "variable", "value", None, influences_fname.split(".csv")[0], optional=True)

    ###################################
    ## BUILD DYNAMICAL CONSTRAINTS   ##
    ###################################
    print("3. Build dynamical constraints by binarization of experimental profiles... ", end=" ...")

    ## Signatures are vectors containing values {NaN,0,1}
    sigs_fname = file_folder+"SIGNATURES_"+"-".join(cell_lines[:4])+"_binthres="+str(sig_args.get("bin_thres", 0.5))+"_thres_iscale="+str(lincs_args.get("thres_iscale", None))+".csv"
    if (not pd.isnull(profiles.values).all() and not os.path.exists(sigs_fname)):
        save_fname=file_folder+"CELL"
        if (experiments_fname is None):
            assert lincs_args
            assert lincs_args.get("credentials", False)
            assert lincs_args.get("path_to_lincs", False)
            save_fname=file_folder+"CELL"
            user_key = get_user_key(lincs_args["credentials"])
            profiles_index = list(profiles.index)
            profiles.index = [int(pert_df.loc[g]["Entrez ID"]) if (g not in add_rows_profiles) else g for g in profiles_index]
            signatures = profiles2signatures(profiles, user_key, lincs_args["path_to_lincs"], save_fname, thres=sig_args.get("bin_thres", 0.5), selection=lincs_args.get("selection", "distil_ss"), backgroundfile=True, bin_method=sig_args.get("bin_method", "binary"), nbackground_limits=(sig_args.get("min_selection", 4), sig_args.get("max_selection", 50)))
            profiles.index = profiles_index
            signatures.index = [pert_inames[list(pert_df["Entrez ID"]).index(e)] for e in signatures.index]
        else:
            conditions = ["_".join([profiles.loc[v][c] for v in ["perturbed","perturbation","cell_line"]]) if (profiles.loc["annotation"][c]=="2") else "initial_"+profiles.loc["cell_line"][c] for c in profiles.columns]
            signatures = profiles.loc[[g for g in profiles.index if (g not in add_rows_profiles)]].apply(pd.to_numeric)
            signatures.columns = conditions
        signatures.to_csv(sigs_fname)
    elif (pd.isnull(profiles.values).all()):
        signatures = pd.DataFrame([], index=model_genes, columns=["0","1"])
        nb_absent_genes = len(model_genes)
        nb_constant_genes = 0
        nb_undetermined_genes, nexps, ncells, ngenes = [0]*4
        fnorm = np.nan
    else:
        signatures = pd.read_csv(sigs_fname, index_col=0, header=0).dropna(how="all")

    if (not pd.isnull(signatures.values).all()):
        signatures_copy = deepcopy(signatures)
        signatures_copy[signatures_copy==0] = -1
        signatures_copy = signatures_copy.fillna(0)
        fnorm = np.linalg.norm(signatures_copy.values)

        nb_absent_genes = len(model_genes)-signatures.shape[0]
        nb_constant_genes = signatures.loc[(signatures.mean(axis=1, skipna=True)==0)|(signatures.mean(axis=1, skipna=True)==1)].shape[0]+nb_absent_genes
        nb_undetermined_genes = signatures.loc[pd.isnull(signatures.mean(axis=1, skipna=True))].shape[0]+nb_absent_genes
        nexps, ncells, ngenes = len(set([c for c in signatures.columns if ("initial" not in c)])), len(set([c for c in signatures.columns if ("initial" in c)])), signatures.shape[0]

        plot_signatures(signatures, perturbed_genes=list(set(list(profiles.loc["perturbed"]))), fname=file_folder+"signatures_binthres="+str(sig_args.get("bin_thres",0.5))+"_thresiscale="+str(lincs_args.get("thres_iscale",0.)))

        print(("... %d experiments on %d cell lines and %d/%d genes (Frobenius norm signature matrix: %.3f, %d possibly constant genes: %.1f" % (nexps, ncells, ngenes, len(model_genes), fnorm, nb_constant_genes, nb_constant_genes*100/len(model_genes)))+"%, "+str(nb_undetermined_genes)+" genes with undetermined status")
    else:
        print("... no experiments.")
    print("\n--- INFER BOOLEAN NETWORK ---")

    ###########################
    ## INFER BOOLEAN NETWORK ##
    ###########################
    print("1. Generate solutions from topological & dynamical constraints... ", end=" ...")

    solution_fname_ls = glob(solution_fname+"_*.zip")
    solution_ls = []
    if (len(solution_ls)<bonesis_args.get("niterations", 1)):
        grn = create_grn(influences, exact=bonesis_args.get("exact", False), max_maxclause=bonesis_args.get("max_maxclause", 3))
        gene_list = grn.nodes
        if (not pd.isnull(signatures.values).all()):
            signatures = signatures[[c for c in signatures.columns if (("initial" in c) or any([(g in c) for g in gene_list]))]]
            signatures = signatures[list(sorted(list(signatures.columns)))]
            cols = [c for c in signatures.columns if ("initial" not in c)]
        if (pd.isnull(signatures.values).all() or sig_args.get("bin_thres",0.5)==0):
            signatures = []
        ## If considering only a subset of experiments
        elif (len(bonesis_args.get("exp_ids", []))>0):
            exps = [cols[i] for i in bonesis_args["exp_ids"]]
            signatures = signatures[[e for e in exps]]+signatures[["initial_"+cell for cell in list(set([c.split("_")[-1] for c in cols]))]]
        BO = build_observations(grn, signatures)
        ## File-readable by BoneSiS: to easily modify constraints
        BO.boolean_networks().standalone(output_filename=solution_fname+".asp")
        nsolutions = infer_network(BO, fname=solution_fname, limit=bonesis_args.get("limit", 1), use_diverse=bonesis_args.get("use_diverse", True), niterations=bonesis_args.get("niterations",1), njobs=njobs)
        if (sum(nsolutions)==0):
            print("No solution found. Try decreasing value bin_thres=%.3f in [0,0.5], or decreasing STRING score=%.2f in [0,1], or increasing max_maxclause=%d, or increasing thres_iscale=%s." % (sig_args.get("bin_thres",0.5),string_args.get("score", 1), bonesis_args.get("max_maxclause", 5), lincs_args.get("thres_iscale", None)))
            sbcall("rm -f "+solution_fname+"*.zip", shell=True)
            return None, 0
        else:
            params = {"file_folder":file_folder, "taxon_id":taxon_id, "path_to_genes":path_to_genes, 
              "disgenet_args": disgenet_args, "string_args" : string_args, "lincs_args": lincs_args, 
              "edge_args" : edge_args, "sig_args" : sig_args, "bonesis_args": bonesis_args, 
              "weights": weights, "seed": seed, "network_fname" : network_fname, "njobs": njobs}
            import json
            with open(solution_fname+'.json', 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
            return solution_fname, nsolutions

def import_all_solutions(solution_fname, quiet=False):
    '''
        Import all solutions which have been generated
        @param\tsolution_fname\tPython character string: header of solution files
        @param\tquiet\tPython bool[default=False]
        @return\tsolutions\tPandas DataFrame: rows/[genes] x columns/[solution IDs] containing regulatory functions for each gene in each solution
    '''
    fname_ls = glob(solution_fname+"_*.zip")
    assert len(fname_ls)>0
    fname_ls = list(sorted(fname_ls, key=lambda x: int(x.split(".zip")[0].split("_")[-1])))
    sol_ls, nsol = [], 1
    try:
        for fi, fname in enumerate(list(sorted(fname_ls))):
            solutions = zip2df(fname)
            cols = list(range(nsol,nsol+solutions.shape[0]))
            nsol = np.max(cols)+1
            sol_ls.append(pd.DataFrame(solutions.T.values, index=solutions.columns, columns=cols))
    except:
        raise ValueError("'"+fname+"' not loaded.")
    nsol -= 1
    if (len(sol_ls)>1):
        solutions = sol_ls[0].join(sol_ls[1:], how="outer")
    else:
        solutions = sol_ls[0]
    if (not quiet):
        print("%d solutions (%d unique solutions)" % (solutions.shape[1], solutions.T.drop_duplicates().shape[0]))
    return solutions

def visualize_models(sols, file_folder):
    '''
        Selection of an optimal model in a set of solutions, based on a topology-based desirability function
        @param\tsols\tPandas DataFrame: rows/[genes] x columns/[solution IDs]
        @param\tweights\tPython dictionary: weight for each graph characteristic
        @return\tsolution\tPandas DataFrame: rows/[genes] x column/[solution ID] selected solution
    '''
    ## * Minimal model (number of edges) *
    minimal, nminimal = get_minimal_edges(sols)
    influences_minimal = solution2influences(minimal)
    influences2graph(influences_minimal, fname=file_folder+"inferred_minimal_solution", optional=False)
    print("<MODEL VISUALIZATION> Minimal solution: %d edges" % nminimal)
    ## * Maximal model (number of edges) *
    maximal, nmaximal = get_minimal_edges(sols, maximal=True)
    influences_maximal = solution2influences(maximal)
    influences2graph(influences_maximal, fname=file_folder+"inferred_maximal_solution", optional=False)
    print("<MODEL VISUALIZATION> Maximal solution: %d edges" % nmaximal)
    return None

def select_optimal_model(sols, weights, file_folder):
    ## * Maximizer of general topological criterion (GTP)
    GTPs = [general_topological_parameter(solution2influences(sols[c]), weights) for c in sols.columns]
    GTP_df = pd.DataFrame([GTPs], index=["GTP"], columns=sols.columns)
    print(GTP_df.head())
    GTP_df.to_csv(file_folder+"GPT.csv")
    max_criterion_solution = sols[sols.columns[np.argmax(GTPs)]]
    max_criterion_influences = solution2influences(max_criterion_solution)
    influences2graph(max_criterion_influences, file_folder+"inferred_max_criterion_solution", optional=False)
    print("<MODEL SELECTION> Saving optimal model in '%s/solution.bnet'" % file_folder, end =" ...")
    save_grn(max_criterion_solution, file_folder+"solution")
    print("... saved!")
    return max_criterion_solution

def solution2cytoscape(solution, fname):
    '''
        Convert a solution into a Cytoscape-readable file
        @param\tsolution\tPandas Series: rows/[genes]
        @param\tfname\tPython character string: path to Cytoscape-readable SIF and XML (style) files (no extension)
        @return\tNone\t
    '''
    ## 1. Create SIF file
    target = []
    grfs = get_grfs_from_solution(solution)
    target = [" ".join([r, "->"+("+" if (grfs[g][r]>0) else "-"), g]) for g in grfs for r in grfs[g]]
    with open(fname+".sif", "w+") as f:
        f.write("\n".join(target))

    ## 2. Build associated Cytoscape style file
    target = []
    target += style_file[0]
    target.append("<visualProperty default=\""+fname+"\" name=\"SOLUTION\"/>")
    target += style_file[1]
    target += ["<visualProperty default=\"#FFCC99\" name=\"NODE_FILL_COLOR\">", "<discreteMapping attributeName=\"name\" attributeType=\"string\">"]
    target += ["</discreteMapping>", "</visualProperty>"]
    target += style_file[2]

    with open(fname+".xml", "w+") as f:
        f.write("\n".join(target))
