import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from reactot.dataset.datasets_config import ATOM_MAPPING, SAM_CHARGED_ATOM_MAPPING
from reactot.dataset.embedding import atomic_fingerprint, row_one_hot, col_one_hot

class BaseDataset(Dataset):
    def __init__(
        self,
        npz_path,
        center=True,
        zero_charge=False,
        device="cpu",
        remove_h=False,
        n_fragment=3,
        atom_mapping=ATOM_MAPPING,
        use_fingerprint_embedding=False,
        use_periodic_embedding=False,
    ) -> None:
        super().__init__()

        if ".npz" in str(npz_path):
            with np.load(npz_path, allow_pickle=True) as f:
                data = {key: val for key, val in f.items()}
        elif ".pkl" in str(npz_path):
            data = pickle.load(open(npz_path, "rb"))
        else:
            raise ValueError("data file should be either .npz or .pkl")

        self.raw_dataset = data
        self.n_samples = -1
        self.data = {}
        self.n_fragment = n_fragment

        self.remove_h = remove_h
        self.zero_charge = zero_charge
        self.center = center
        self.device = device
        
        self.atom_mapping = atom_mapping
        self.n_element = len(list(atom_mapping.keys()))
        self.use_fingerprint_embedding = use_fingerprint_embedding
        self.use_periodic_embedding = use_periodic_embedding

    def __len__(self):
        return len(self.data["size_0"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        sizes = []
        for k in batch[0].keys():
            if "size" in k:
                sizes.append(int(k.split("_")[-1]))
        n_fragment = len(sizes)
        out = [{} for _ in range(n_fragment)]
        res = {}
        for prop in batch[0].keys():
            # print(prop)
            if prop not in ["target", "rmsd", "ediff", "ts_guess"]: #! important change. Treat condition for each fragment
                idx = int(prop.split("_")[-1])
                _prop = prop.replace(f"_{idx}", "")
            if "size" in prop:
                out[idx][_prop] = torch.tensor(
                    [x[prop] for x in batch],
                    device=batch[0][prop].device,
                )
            elif "mask" in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[idx][_prop] = torch.cat(
                    [
                        i * torch.ones(len(x[prop]), device=x[prop].device).long()
                        for i, x in enumerate(batch)
                    ],
                    dim=0,
                )
            elif prop in ["target", "rmsd", "ediff", "ts_guess"]: #! important change. Treat condition for each fragment
                res[prop] = torch.cat([x[prop] for x in batch], dim=0)
            else:
                out[idx][_prop] = torch.cat([x[prop] for x in batch], dim=0)

        res["condition"] = [v.pop("condition") for v in out]
        if len(list(res.keys())) == 1:
            return out, res["condition"]
        return out, res

    def patch_dummy_molecules(self, idx):
        self.data[f"size_{idx}"] = torch.ones_like(
            self.data[f"size_0"], device=self.device,
        )
        self.data[f"pos_{idx}"] = [
            torch.tensor([[0, 0, 0]], device=self.device,)
            for _ in range(self.n_samples)
        ]

        self.data[f"one_hot_{idx}"] = [
            torch.tensor([0], device=self.device,)
            for _ in range(self.n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            F.one_hot(_z, num_classes=self.n_element) for _z in self.data[f"one_hot_{idx}"]
        ]

        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                torch.zeros(size=(1, 1), dtype=torch.int64, device=self.device,)
                for _ in range(self.n_samples)
            ]
        else:
            self.data[f"charge_{idx}"] = [
                torch.ones(size=(1, 1), dtype=torch.int64, device=self.device,)
                for _ in range(self.n_samples)
            ]

        self.data[f"mask_{idx}"] = [
            torch.zeros(size=(1,), dtype=torch.int64, device=self.device,)
            for _ in range(self.n_samples)
        ]


    def get_charge_encoding(self, data: dict, n_samples: int, encoding_fn, charge_key: str=None):
        if charge_key is None:
            charge_key = "charges"
        
        if isinstance(encoding_fn, dict):
            mapping = encoding_fn           # keep a reference to the dict
            encoding_fn = lambda x: mapping[x]
        else:
            assert callable(encoding_fn), "encoding_fn should be either a dict or a callable function"

        return [
                torch.tensor(
                    [
                        encoding_fn(int(_at))
                        for _at in data[charge_key][ii][: data["num_atoms"][ii]]
                    ],
                    device=self.device,
                )
                for ii in range(n_samples)
            ]
            
    def process_molecules(self, dataset_name, n_samples, idx, append_charge=None,
                          position_key="positions", swap_charges_key=None, use_swap_charges_one_hot=False):
        data = getattr(self, dataset_name)
        self.data[f"size_{idx}"] = torch.tensor(data["num_atoms"], device=self.device)
        self.data[f"pos_{idx}"] = [
            torch.tensor(
                data[position_key][ii][: data["num_atoms"][ii]],
                device=self.device,
                dtype=torch.float32,
            )
            for ii in range(n_samples)
        ]

        # Make a (one hot) embedding of the atomic numbers. 
        charges_key = "charges"
        if swap_charges_key is not None and use_swap_charges_one_hot:
            charges_key = swap_charges_key
        
        self.data[f"one_hot_{idx}"] = self.get_charge_encoding(data, n_samples, self.atom_mapping, charge_key=charges_key)

        if self.use_fingerprint_embedding:
            self.data[f"fingerprint_{idx}"] = self.get_charge_encoding(data, n_samples, atomic_fingerprint, charge_key=charges_key)
            
        if self.use_periodic_embedding:
            self.data[f"periodic_row_{idx}"] = self.get_charge_encoding(data, n_samples, row_one_hot, charge_key=charges_key)
            self.data[f"periodic_col_{idx}"] = self.get_charge_encoding(data, n_samples, col_one_hot, charge_key=charges_key)
                
                
        self.data[f"one_hot_{idx}"] = [
            F.one_hot(_z, num_classes=self.n_element)
            for _z in self.data[f"one_hot_{idx}"]
        ]


        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                torch.zeros(size=(_size, 1), dtype=torch.int64, device=self.device,)
                for _size in data["num_atoms"]
            ]
        else:
            if append_charge is None:
                # Use the charges of the swapped system
                charges_key = "charges"
                if swap_charges_key is not None:
                    charges_key = swap_charges_key 
                
                self.data[f"charge_{idx}"] = [
                    torch.tensor(
                        data[charges_key][ii][: data["num_atoms"][ii]],
                        device=self.device,
                    ).view(-1, 1)
                    for ii in range(n_samples)
                ]
            else:
                self.data[f"charge_{idx}"] = [
                    torch.cat(
                        [
                            torch.tensor(
                                data["charges"][ii][: data["num_atoms"][ii]],
                                device=self.device,
                            ).view(-1, 1),
                            torch.tensor(
                                [append_charge for _ in range(data["num_atoms"][ii])],
                                device=self.device,
                            ).view(-1, 1),
                        ],
                        dim=1,
                    )
                    for ii in range(n_samples)
                ]

        self.data[f"mask_{idx}"] = [
            torch.zeros(size=(_size,), dtype=torch.int64, device=self.device,)
            for _size in data["num_atoms"]
        ]

        if self.center:
            self.data[f"pos_{idx}"] = [
                pos - torch.mean(pos, dim=0) for pos in self.data[f"pos_{idx}"]
            ]
