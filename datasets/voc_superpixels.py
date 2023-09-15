import os
import pickle

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive, Subset
from dgl.dataloading import GraphDataLoader
from ogb.utils.url import makedirs
from tqdm import tqdm


class VOCSuperpixelsDataset(DGLDataset):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.

    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels.
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node

    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.

    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor

    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: "~/.dgl/".
    split : str
        Should be chosen from ["train", "val", "test"]
        Default: "train".
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    construct_format : str, optional:
        Option to select the graph construction format.
        Should be chosen from ["edge_wt_only_coord", "edge_wt_coord_feat", "edge_wt_region_boundary"]
        If : "edge_wt_only_coord", the graphs are 8-nn graphs with the edge weights computed based on
        only spatial coordinates of superpixel nodes.
        If : "edge_wt_coord_feat", the graphs are 8-nn graphs with the edge weights computed based on
        combination of spatial coordinates and feature values of superpixel nodes.
        If : "edge_wt_region_boundary", the graphs region boundary graphs where two regions (i.e.
        superpixel nodes) have an edge between them if they share a boundary in the original image.
        Default: "edge_wt_region_boundary"
    slic_compactness : int, optional:
        Option to select compactness of slic that was used for superpixels
        Should be chosen from [10, 30]
        Default: 30.

    Examples
    ---------
    >>> from dgl.data import VOCSuperpixelsDataset

    >>> train_dataset = VOCSuperpixelsDataset(split="train")
    >>> len(train_dataset)
    8498
    >>> train_dataset.num_classes
    21
    >>> graph= train_dataset[0]
    >>> graph
    Graph(num_nodes=460, num_edges=2632,
        ndata_schemes={'feat': Scheme(shape=(14,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(2,), dtype=torch.float32)})
    """

    urls = {
        10: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
        30: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
    }

    def __init__(
        self,
        raw_dir=None,
        split="train",
        force_reload=None,
        verbose=None,
        construct_format="edge_wt_region_boundary",
        slic_compactness=30,
    ):
        self.construct_format = construct_format
        self.slic_compactness = slic_compactness
        assert split in ["train", "val", "test"]
        assert construct_format in [
            "edge_wt_only_coord",
            "edge_wt_coord_feat",
            "edge_wt_region_boundary",
        ]
        assert slic_compactness in [10, 30]
        self.split = split
        super(VOCSuperpixelsDataset, self).__init__(
            name="PascalVOC-SP",
            raw_dir=raw_dir,
            url=self.urls[self.slic_compactness][self.construct_format],
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def save_path(self):
        return os.path.join(
            self.raw_path,
            "slic_compactness_" + str(self.slic_compactness),
            self.construct_format,
        )

    @property
    def raw_data_path(self):
        return os.path.join(self.save_path, f"{self.split}.pickle")

    @property
    def graph_path(self):
        return os.path.join(self.save_path, f"processed_{self.split}.pkl")

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 21

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.graphs)

    def download(self):
        zip_file_path = os.path.join(
            self.raw_path, "voc_superpixels_" + self.construct_format + ".zip"
        )
        path = download(self.url, path=zip_file_path)
        extract_archive(path, self.raw_path, overwrite=True)
        makedirs(self.save_path)
        os.rename(
            os.path.join(self.raw_path, "voc_superpixels_" + self.construct_format),
            self.save_path,
        )
        os.unlink(path)

    def process(self):
        with open(self.raw_data_path, "rb") as f:
            graphs = pickle.load(f)

        indices = range(len(graphs))

        pbar = tqdm(total=len(indices))
        pbar.set_description(f"Processing {self.split} dataset")

        self.graphs = []
        for idx in indices:
            graph = graphs[idx]

            """
            Each `graph` is a tuple (x, edge_attr, edge_index, y)
                Shape of x : [num_nodes, 14]
                Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                Shape of edge_index : [2, num_edges]
                Shape of y : [num_nodes]
            """
            dgl_graph = dgl.graph(
                (graph[2][0], graph[2][1]),
                num_nodes=len(graph[3]),
            )
            dgl_graph.ndata["feat"] = graph[0].to(torch.float)
            dgl_graph.edata["feat"] = graph[1].to(torch.float)
            dgl_graph.ndata["label"] = torch.LongTensor(graph[3])
            self.graphs.append(dgl_graph)

            pbar.update(1)

        pbar.close()
        print("Saving...")

    def load(self):
        with open(self.graph_path, "rb") as f:
            f = pickle.load(f)
            self.graphs = f

    def save(self):
        with open(os.path.join(self.graph_path), "wb") as f:
            pickle.dump(self.graphs, f)

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def __getitem__(self, idx):
        r"""Get the idx^th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels and edge features.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``edata['feat']``: edge features
        """
        if isinstance(idx, int):
            return self.graphs[idx]
        elif torch.is_tensor(idx):
            if torch.ndim(idx) == 0:
                return self.graphs[idx]
            elif torch.ndim(idx) == 1:
                return Subset(self, idx.cpu())


if __name__ == "__main__":
    train_dataset = VOCSuperpixelsDataset(raw_dir="data")
    val_dataset = VOCSuperpixelsDataset(raw_dir="data", split="val")
    test_dataset = VOCSuperpixelsDataset(raw_dir="data", split="test")

    graph = train_dataset[0]
    print(graph)
    print(len(train_dataset))
    print("# of classes for each node", train_dataset.num_classes)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=32, shuffle=False)
    for i, batched_g in enumerate(train_dataloader):
        print("batched graph", batched_g)
        assert batched_g.num_nodes() == batched_g.ndata["label"].shape[0]
        break
