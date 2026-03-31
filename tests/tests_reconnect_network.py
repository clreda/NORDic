#coding:utf-8

from NORDic.UTILS.utils_grn import load_grn, solution2influences, reconnect_network
from NORDic.UTILS.utils_plot import influences2graph
import pandas as pd
import os
from IPython.display import Image

file_folder = "./"
solution_name = "synthetic"

# Prints original network
network_fname = f"{file_folder}/{solution_name}.bnet"
with open(network_fname, "r") as f:
    network = pd.DataFrame({"Solution": dict([["_".join(g.split("-")) for g in x.split(", ")] for x in f.read().split("\n") if (len(x)>0)])})
influences = solution2influences(network["Solution"])
influences2graph(influences, file_folder+"network", optional=False, compile2png=True, engine=["sfdp","dot"][0])

# Reconnect network
solution = load_grn(f"{file_folder}/{solution_name}.bnet")
if (not os.path.exists(f"{file_folder}/{solution_name}_connected.bnet")):
    network_fname = reconnect_network(f"{file_folder}/{solution_name}.bnet") ## creates the "solution_connected.bnet" file

# Prints reconnected network
network_fname = f"{file_folder}/{solution_name}_connected.bnet"
with open(network_fname, "r") as f:
    network_connected = pd.DataFrame({"Solution": dict([["_".join(g.split("-")) for g in x.split(", ")] for x in f.read().split("\n") if (len(x)>0)])})
influences_connected = solution2influences(network_connected["Solution"])
influences2graph(influences_connected, file_folder+"network_connected", optional=False, compile2png=True, engine=["sfdp","dot"][0])
