#:coding:utf-8

import json
import requests
from subprocess import call as sbcall
from subprocess import check_output as sbcheck_output
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from functools import reduce
from time import sleep

from NORDic.NORDic_DS.get_drug_signatures import drugname2pubchem, pubchem2drugname
from NORDic.UTILS.LINCS_utils import get_user_key, build_url, post_request

def retrieve_drug_targets(file_folder, drug_names, TARGET_args={}, gene_list=[], sources=["DrugBank","MINERVA","LINCS","TTD","DrugCentral"], quiet=False):
    '''
        Retrieve drug targets from several online sources
        @param\tdrug_names\tPython character string list: list of common drug names
        @param\tTARGET_args\tPython dictionary[default={}]: see for each source
        @param\tgene_list\tPython character string list[defaul=[]]: list of HGNC symbols
        @param\tsources\tPython character string list[default=["DrugBank","MINERVA","LINCS","TTD","DrugCentral"]]: list of source names (in ["DrugBank","MINERVA","LINCS","TTD","DrugCentral"])
        @param\tquiet\tPython bool[default=False]
        @param\ttarget_df\tPandas DataFrame: rows/[genes] x columns/[drug names], values are the number of times a target is connected to a drug
    '''
    assert all([s in ["DrugBank","MINERVA","LINCS","TTD","DrugCentral"] for s in sources])
    targets_list = []

    if ("MINERVA" in sources):
        if (not quiet):
            print("<DRUG_TARGETS> MINERVA")
        if (not os.path.exists(file_folder+"drug_targets_MINERVA.csv")):
           targets_MINERVA = get_targets_MINERVA(drug_names, quiet=quiet)
           targets_MINERVA.to_csv(file_folder+"drug_targets_MINERVA.csv")
           print(targets_MINERVA.head())
        targets_MINERVA = pd.read_csv(file_folder+"drug_targets_MINERVA.csv", index_col=0)
        if (targets_MINERVA.shape[1]<len(drug_names)):
            targets_MINERVA = pd.concat(tuple([targets_MINERVA]+[pd.DataFrame([],index=targets_MINERVA.index,columns=[d]) for d in drug_names if (d not in targets_MINERVA.columns)]), axis=1).fillna("")
        targets_list.append(targets_MINERVA[drug_names])

    if ("DrugBank" in sources):
        if (not quiet):
            print("<DRUG_TARGETS> DrugBank")
        if ("DrugBank" in TARGET_args and all([x in TARGET_args["DrugBank"] for x in ["path_to_drugbank","target_fname","drug_fname"]])):
            if (not os.path.exists(file_folder+"drug_targets_DrugBank.csv")):
                targets_DrugBank = get_targets_DrugBank(drug_names, **TARGET_args["DrugBank"], quiet=quiet)
                targets_DrugBank.to_csv(file_folder+"drug_targets_DrugBank.csv")
                print(targets_DrugBank.head())
            targets_DrugBank = pd.read_csv(file_folder+"drug_targets_DrugBank.csv", index_col=0)
            if (targets_DrugBank.shape[1]<len(drug_names)):
                targets_DrugBank = pd.concat(tuple([targets_DrugBank]+[pd.DataFrame([],index=targets_DrugBank.index,columns=[d]) for d in drug_names if (d not in targets_DrugBank.columns)]), axis=1).fillna("")
            targets_list.append(targets_DrugBank[drug_names])
        else:
            print("One of the following fields: %s is not present in input" % str(["path_to_drugbank","target_fname","drug_fname"]))

    if ("LINCS" in sources):
        if (not quiet):
            print("<DRUG_TARGETS> LINCS")
        if ("LINCS" in TARGET_args):
            if (not os.path.exists(file_folder+"drug_targets_LINCS.csv")):
                targets_LINCS = get_targets_LINCS(drug_names, **TARGET_args["LINCS"], quiet=quiet)
                targets_LINCS.to_csv(file_folder+"drug_targets_LINCS.csv")
                print(targets_LINCS.head())
            targets_LINCS = pd.read_csv(file_folder+"drug_targets_LINCS.csv", index_col=0)
            if (targets_LINCS.shape[1]<len(drug_names)):
                targets_LINCS = pd.concat(tuple([targets_LINCS]+[pd.DataFrame([],index=targets_LINCS.index,columns=[d]) for d in drug_names if (d not in targets_LINCS.columns)]), axis=1).fillna("")
            targets_list.append(targets_LINCS[drug_names])
        else:
            print("One of the following fields: %s is not present in input" % str(["path_to_lincs","credentials"]))

    if ("TTD" in sources):
        if (not quiet):
            print("<DRUG_TARGETS> TTD")

        if (not os.path.exists(file_folder+"drug_targets_TTD.csv")):
            targets_TTD = get_targets_TTD(drug_names, quiet=quiet)
            targets_TTD.to_csv(file_folder+"drug_targets_TTD.csv")
            print(targets_TTD.head())
        targets_TTD = pd.read_csv(file_folder+"drug_targets_TTD.csv", index_col=0)
        if (targets_TTD.shape[1]<len(drug_names)):
            targets_TTD = pd.concat(tuple([targets_TTD]+[pd.DataFrame([],index=targets_TTD.index,columns=[d]) for d in drug_names if (d not in targets_TTD.columns)]), axis=1).fillna("")
        targets_list.append(targets_TTD[drug_names])

    if ("DrugCentral" in sources):
        if (not quiet):
            print("<DRUG_TARGETS> DrugCentral")
        if (not os.path.exists(file_folder+"drug_targets_DrugCentral.csv")):
            targets_DrugCentral = get_targets_DrugCentral(drug_names, quiet=quiet)
            targets_DrugCentral.to_csv(file_folder+"drug_targets_DrugCentral.csv")
            print(targets_DrugCentral.head())
        targets_DrugCentral = pd.read_csv(file_folder+"drug_targets_DrugCentral.csv", index_col=0)
        if (targets_DrugCentral.shape[1]<len(drug_names)):
            targets_DrugCentral = pd.concat(tuple([targets_DrugCentral]+[pd.DataFrame([],index=targets_DrugCentral.index,columns=[d]) for d in drug_names if (d not in targets_DrugCentral.columns)]), axis=1).fillna("")
        targets_list.append(targets_DrugCentral[drug_names])

    index_full = list(sorted(list(set([g for t in targets_list for g in t.index]+gene_list))))
    targets_list = [pd.concat((t, pd.DataFrame([], index=[g for g in index_full if (g not in t.index)], columns=t.columns)), axis=0).loc[index_full].fillna("") for t in targets_list]
    targets_df = reduce(lambda x,y: np.add(x,y), targets_list)
    is_inhibitor = np.vectorize(lambda x : "-" in x)
    targets_df[is_inhibitor(targets_df.values)] = "-1"
    is_activator = np.vectorize(lambda x : "+" in x)
    targets_df[is_activator(targets_df.values)] = "1"
    targets_df[targets_df==""] = "0"
    targets_df[(targets_df!="0")&(targets_df!="-1")&(targets_df!="1")] = "-1"#"1"
    targets_df = targets_df.astype(int)
    if (len(gene_list)>0):
        targets_df = targets_df.loc[gene_list]
    return targets_df

#################
##   MINERVA   ##
#################
## https://minerva.pages.uni.lu/doc/api/15.0/projects/
def get_targets_MINERVA(drug_names, quiet=False):
    '''
        Utility function using httr:GET to send queries to a given MINERVA Platform instance
        @param\tdrug_names\tPython character string list: list of common drug names
        @param\tquiet\tPython bool[default=False]
        @return\ttarget_df\tPandas DataFrame: rows/[lists of HGNC symbols of targets] x columns/[drug names], values are "*" (known regulator) or "" (no known regulation)
    '''
    base_url="https://minerva-dev.lcsb.uni.lu/minerva/api/projects/"
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    ## 1. List all disease maps in MINERVA
    response = requests.get(base_url, headers=headers)
    assert (response.status_code==200 and (len(json.loads(response.text))>0))
    projects = json.loads(response.text)
    project_ids = [pi['projectId'] for pi in projects]
    target_lst = [[]]*len(drug_names)
    for idi, drug_name in tqdm(enumerate(drug_names)):
        for project_id in tqdm(project_ids):
            sleep(0.01)
            response = requests.get(base_url+project_id+"/drugs:search?query="+drug_name, headers=headers)
            ### 2. Omit results with zero map elements, these are drug targets not in the map
            if (response.status_code!=200):
                continue
            targets = json.loads(response.text)
            if (len(targets)==0):
                continue
            map_hits = [x for x in targets[0]["targets"] if (len(x["targetElements"])>0)]
            ### 3. Extract HGNC symbols from the 'targetParticipants' part of the results
            target_lst[idi] += [xx["resource"] for x in map_hits for xx in x["targetParticipants"]]
        target_lst[idi] = list(set(target_lst[idi]))
        if (not quiet):
            print("<MINERVA> Drug %s (%d/%d): %d targets" % (drug_name, idi+1, len(drug_names), len(target_lst[idi])))
    target_df = pd.DataFrame({dn: {x:"*" for x in target_lst[idn]} for idn, dn in enumerate(drug_names)}).fillna("")
    return target_df

#################
##  DRUGBANK   ##
#################
def get_targets_DrugBank(drug_names, path_to_drugbank, drug_fname, target_fname, quiet=False):
    '''
        Utility function which gets drug targets from a local DrugBank database
        @param\tdrug_names\tPython character string list: list of common drug names
        @param\tpath_to_drugbank\tPython character string: (relative) path to DrugBank files
        @param\tdrug_fname\tPython character string: file name of .XML file of the complete database (full_database.xml) in DrugBank
        @param\ttarget_fname\tPython character string: file name of .CSV file of the protein targets (all.csv) in DrugBank
        @param\tquiet\tPython bool[default=False]
        @return\ttarget_df\tPandas DataFrame: rows/[lists of HGNC symbols of targets] x columns/[drug names], values are "*" (known regulator) or "" (no known regulation)
    '''
    ## 1. Find drug names from DrugBank identifiers (using file from DrugBank after registration)
    if (not os.path.exists(path_to_drugbank+drug_fname)):
        sbcall("unzip "+"/".join(path_to_drugbank+drug_fname.split("/")[:-1])+"drugbank_all_full_database.xml.zip", shell=True)
    cmd_file="cat \'"+path_to_drugbank+drug_fname+"\'"
    cmd_drugbank_names="grep -e '^  <name>' | sed 's/  <name>//g' | sed 's/<\/name>//g'"
    cmd_drugbank_ids="grep -e '^  <drugbank-id primary=\"true\">' | sed 's/<\/drugbank-id>//g' | sed 's/<drugbank-id primary=\"true\">//g' | sed 's/ //g'"
    drugbank_ids=sbcheck_output(cmd_file+" | "+cmd_drugbank_ids, shell=True).decode("utf-8").split("\n")
    drugbank_names=sbcheck_output(cmd_file+" | "+cmd_drugbank_names, shell=True).decode("utf-8").split("\n")
    di_drugbankid2drugname = dict(zip(drugbank_ids, drugbank_names))
    ## 2. Manually retrieved from DrugBank website
    new_names = {"DB00510": "Divalproex sodium", "DB01402": "Bismuth", "DBCAT004271": 'Asparaginase', "DB00371" : "Meprobamate",
        "DB00394": "Beclomethasone dipropionate", "DB00422": "Methylphenidate", "DB00462": "Methscopolamine bromide",
        "DB00464": "Sodium tetradecyl sulfate", "DB00525": "Tolnaftate", "DB00527": "Cinchocaine", "DB00563": "Methotrexate",
        "DB05381": "Histamine", "DB00931": "Metacycline", "DB00326": "Calcium glucoheptonate",
        "DB14520": "Tetraferric tricitrate decahydrate", "DB00717": "Norethisterone", "DB01258": "Aliskiren", "DB00006" : "Bivalirudin",
    }
    di_drugbankid2drugname.update(new_names)
    ## 3. Build dictionary (keys=DrugBank IDs, values=drug_names) for the input drug names
    convert_di = {x:y for x,y in list(di_drugbankid2drugname.items()) if (y in drug_names)}
    ## 4. Get protein DB
    protein_db = pd.read_csv(path_to_drugbank+target_fname, index_col=5).query("Species=='Humans'")[["Gene Name", "Drug IDs"]]
    protein_db["Drug IDs"] = ["; ".join([convert_di[xx] for xx in list(set(x.split("; "))) if (xx in convert_di)]) for x in protein_db["Drug IDs"]]
    protein_db = protein_db.loc[protein_db["Drug IDs"]!=""]
    protein_db["presence"] = "*"
    ## 5. Decouple rows with a list of several drugs
    add_rows = protein_db.loc[np.vectorize(lambda x : "; " in x)(list(protein_db["Drug IDs"]))]
    add_rows = [[g, d, sign] for g, d_lst, sign in add_rows.values.tolist() for d in d_lst.split("; ")]
    add_rows = pd.DataFrame(add_rows, index=range(len(add_rows)), columns=protein_db.columns)
    protein_db = pd.concat((protein_db.loc[np.vectorize(lambda x : "; " not in x)(list(protein_db["Drug IDs"]))], add_rows), axis=0)
    protein_db.index = ["--".join(list(protein_db.loc[idx][["Drug IDs", "Gene Name"]])) for idx in protein_db.index]
    protein_db = protein_db.loc[~protein_db.index.duplicated()]
    target_df = protein_db.pivot(columns="Drug IDs", index="Gene Name", values="presence").fillna("")
    return target_df

####################
##  LINCS L1000   ##
####################
def get_targets_LINCS(drug_names, path_to_lincs, credentials, selection=None, nsigs=None, quiet=False):
    '''
        Utility function to retrieve drug targets from the LINCS L1000 database
        @param\tdrug_names\tPython character string list: list of common drug names
        @param\tpath_to_lincs\tPython character string: (relative) path to LINCS files
        @param\tcredentials\tPython character string: (relative) path to LINCS credentials file
        @param\tquiet\tPython bool[default=False]
        @return\ttarget_df\tPandas DataFrame: rows/[lists of HGNC symbols of targets] x columns/[drug names], values are "*" (known regulator) or "" (no known regulation)
    '''
    target_lst = [[]]*len(drug_names)
    pubchem_cids = drugname2pubchem(drug_names, lincs_args={"path_to_lincs": path_to_lincs, "credentials": credentials})
    pubchem_cids.update({"dmcm": 104999,"glycerin":753})
    user_key = get_user_key(credentials)
    for idi, drug_name in tqdm(enumerate(drug_names)):
        if (np.isnan(pubchem_cids[drug_name])):
            continue
        params = {"where": {"pubchem_cid": pubchem_cids[drug_name]}, "fields": ["target"]}
        url_perts = build_url("perts", method="filter", params=params, user_key=user_key)
        sleep(0.01)
        data = post_request(url_perts, quiet=quiet)
        if (len(data)>0):
            targets = data[0]["target"]
            target_lst[idi] = targets
        if (not quiet):
            print("<LINCS> Drug %s (%d/%d): %d targets" % (drug_name, idi+1, len(drug_names), len(target_lst[idi])))
    target_df = pd.DataFrame({dn: {x:"*" for x in target_lst[idn]} for idn, dn in enumerate(drug_names)}).fillna("")
    return target_df

##########################################
##  THERAPEUTIC TARGET DATABASE (TTD)   ##
##########################################
def get_targets_TTD(drug_names, quiet=False):
    '''
        Utility function to retrieve drug targets from the Therapeutic Target Database (TTD)
        @param\tdrug_name\tPython character string list: list of common drug names
        @param\tquiet\tPython bool[default=False]
        @return\ttarget_df\tPandas DataFrame: rows/[lists of HGNC symbols of targets] x columns/[drug names], values are "-" (inhibitor) or "+" (activator) or "" (no known regulation)
    '''
    ## 1. Download files (TARGET-ID and DRUG-ID matchings)
    ttd_url = "http://db.idrblab.net/ttd/sites/default/files/ttd_database/"
    target_info = "P1-01-TTD_target_download.txt"
    drug_target = "P1-07-Drug-TargetMapping.xlsx"
    sbcall("wget -nc "+ttd_url+target_info, shell=True)
    sbcall("wget -nc "+ttd_url+drug_target, shell=True)
    target_matchings = dict([x.split("\t") for x in sbcheck_output("cat "+target_info+" | grep '\tGENENAME\t' | cut -f1,3", shell=True).decode("utf-8").split("\n")[:-1]])
    drug_matchings = dict([[x.split("\t")[0],drug_names[[d.lower() for d in drug_names].index(x.split("\t")[-1].lower())]] for x in sbcheck_output("cat "+target_info+" | grep '\tDRUGINFO\t' | cut -f3,4", shell=True).decode("utf-8").split("\n")[:-1] if (x.split("\t")[-1].lower() in [d.lower() for d in drug_names])])
    ## 2. Get TARGET-DRUG MATCHINGS
    drug_targets = pd.DataFrame([x.split(",") for x in sbcheck_output("xlsx2csv "+drug_target+" | cut -d',' -f1,2,4", shell=True).decode("utf-8").split("\n")[1:-1]], columns=["Target", "Drug", "MOA"])
    drug_targets["Drug"] = [drug_matchings.get(g, "") for g in drug_targets["Drug"]]
    drug_targets = drug_targets.loc[drug_targets["Drug"]!=""]
    decrease_expr_MOA = ["inhibitor","blocker","degrader","antagonist","antisense","breaker","disrupter", "inactivator","inverse agonist","replacement","suppressor"]
    drug_targets["MOA"] = [("-" if (m in decrease_expr_MOA) else "+") for m in drug_targets["MOA"]]
    drug_targets["Target"] = [target_matchings.get(t,"") for t in drug_targets["Target"]]
    drug_targets = drug_targets.loc[drug_targets["Target"]!=""]
    ## 3. Decouple rows with several targets
    add_rows = drug_targets.loc[np.vectorize(lambda x : "; " in x)(list(drug_targets["Target"]))]
    add_rows = [[g, d, sign] for g_lst, d, sign in add_rows.values.tolist() for g in g_lst.split("; ")]
    add_rows = pd.DataFrame(add_rows, index=range(len(add_rows)), columns=drug_targets.columns)
    drug_targets = pd.concat((drug_targets.loc[np.vectorize(lambda x : "; " not in x)(list(drug_targets["Target"]))], add_rows), axis=0)
    drug_targets.index = ["--".join(list(drug_targets.loc[idx][["Drug", "Target"]])) for idx in drug_targets.index]
    drug_targets = drug_targets.loc[~drug_targets.duplicated()]
    sbcall("rm -f "+target_info+" "+drug_target, shell=True)
    target_df = drug_targets.pivot(index="Target", columns="Drug", values="MOA").fillna("")
    return target_df

##########################################
##  DrugCentral 2021                    ##
##########################################
def get_targets_DrugCentral(drug_names, quiet=False):
    '''
        Utility function to retrieve drug targets from the DrugCentral 2021
        @param\tdrug_name\tPython character string list: list of common drug names
        @param\tquiet\tPython bool[default=False]
        @return\ttarget_df\tPandas DataFrame: rows/[lists of HGNC symbols of targets] x columns/[drug names], values are "*" (known regulator) or "" (no known regulation)
    '''
    ffile = [[drug_names[[d.lower() for d in drug_names].index(x.split("\t")[0][1:-1].lower())], [xx for xx in x.split("\t")[1][1:-1].split("|") if ((len(xx)>1) and (xx[1].upper()==xx[1]))]] for x in sbcheck_output("wget -qO- https://unmtid-shinyapps.net/download/DrugCentral/2021_09_01/drug.target.interaction.tsv.gz | gzip -d | cut -f1,6", shell=True).decode("utf-8").split("\n")[:-1] if (x.split("\t")[0][1:-1].lower() in [d.lower() for d in drug_names])]
    ffile = [[x, y] for x,y in ffile if (len(y)>0)]
    ffile = pd.DataFrame([[drug_name, xx] for drug_name in drug_names for x in [v for u, v in ffile if (u==drug_name)] for xx in x])
    ffile = ffile.drop_duplicates(keep="first")
    ffile["presence"] = "*"
    target_df = ffile.pivot(index=1, columns=0, values="presence").fillna("")
    return target_df