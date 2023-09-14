import hashlib
import os, sys
import pickle
import shutil

import dgl

import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs, Subset
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from tqdm import tqdm


class PeptidesFunctionalDataset(DGLDataset):
    r"""
    DGL dataset of 15,535 peptides represented as their molecular graph
    (SMILES) with 10-way multi-task binary classification of their
    functional classes.

    The goal is use the molecular representation of peptides instead
    of amino acid sequence representation ('peptide_seq' field in the file,
    provided for possible baseline benchmarking but not used here) to test
    GNNs' representation capability.

    The 10 classes represent the following functional classes (in order):
        ['antifungal', 'cell_cell_communication', 'anticancer',
        'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
        'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

    Statistics:
    Train examples: 10,873
    Valid examples: 2,331
    Test examples: 2,331
    Average number of nodes: 150.94
    Average number of edges: 307.30
    Number of atom types: 9
    Number of bond types: 3

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: "~/.dgl/".
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    smiles2graph (callable):
        A callable function that converts a SMILES string into a graph object. We use the OGB featurization.
        * The default smiles2graph requires rdkit to be installed *

    Examples
    ---------
    >>> from dgl.data import PeptidesStructuralDataset
    >>> dataset = PeptidesStructuralDataset()
    >>> len(dataset)
    15535
    >>> dataset.num_atom_types
    9
    >>> graph, label = dataset[0]
    >>> graph
    Graph(num_nodes=119, num_edges=244,
        ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)})

    >>> split_dict = dataset.get_idx_split()
    >>> trainset = dataset[split_dict["train"]]
    >>> graph, label = trainset[0]
    >>> graph
    Graph(num_nodes=338, num_edges=682,
        ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)})

    """

    def __init__(
        self, raw_dir=None, force_reload=None, verbose=None, smiles2graph=smiles2graph
    ):
        self.smiles2graph = smiles2graph
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        super(PeptidesFunctionalDataset, self).__init__(
            name="Peptides-func",
            raw_dir=raw_dir,
            url="https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1",
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def raw_data_path(self):
        return os.path.join(self.raw_path, "peptide_multi_class_dataset.csv.gz")

    @property
    def split_data_path(self):
        return os.path.join(self.raw_path, "splits_random_stratified_peptide.pickle")

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "Peptides-func.bin")

    @property
    def num_atom_types(self):
        return 9

    @property
    def num_bond_types(self):
        return 3

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download(self.url, path=self.raw_data_path)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(os.path.join(self.raw_path, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download(self.url_stratified_split, path=self.split_data_path)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(self.raw_data_path)

        smiles_list = data_df["smiles"]

        print("Converting SMILES strings into graphs...")
        self.graphs = []
        self.labels = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            dgl_graph = dgl.graph(
                (graph["edge_index"][0], graph["edge_index"][1]),
                num_nodes=graph["num_nodes"],
            )
            dgl_graph.edata["feat"] = torch.from_numpy(graph["edge_feat"]).to(
                torch.int64
            )
            dgl_graph.ndata["feat"] = torch.from_numpy(graph["node_feat"]).to(
                torch.int64
            )

            self.graphs.append(dgl_graph)
            self.labels.append(eval(data_df["labels"].iloc[i]))

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert all([not any(torch.isnan(self.labels[i])) for i in split_dict["train"]])
        assert all([not any(torch.isnan(self.labels[i])) for i in split_dict["val"]])
        assert all([not any(torch.isnan(self.labels[i])) for i in split_dict["test"]])

    def load(self):
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict["labels"]

    def save(self):
        save_graphs(self.graph_path, self.graphs, labels={"labels": self.labels})

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        with open(self.split_data_path, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            "Only integers and long are valid "
            "indices (got {}).".format(type(idx).__name__)
        )


# Collate function for ordinary graph classification
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels


if __name__ == "__main__":
    dataset = PeptidesFunctionalDataset(raw_dir="data")
    graph, label = dataset[0]
    print(graph)
    print(len(dataset))
    split_dict = dataset.get_idx_split()
    print(split_dict)
    trainset = dataset[split_dict["train"]]
    print(trainset[0][0])
