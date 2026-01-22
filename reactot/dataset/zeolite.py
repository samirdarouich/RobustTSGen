import copy
import numpy as np
import torch

from reactot.dataset.base_dataset import BaseDataset
from reactot.dataset.datasets_config import ZEOLITE_ATOM_MAPPING


FRAG_MAPPING = {
    "reactant": "reactant",
}



def reflect_z(x):
    x = np.array(x)
    x[:, -1] = -x[:, -1]
    return x


class ProcessedZeolite(BaseDataset):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cuda",
        zero_charge=False,
        remove_h=False,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=True,
        only_one=True,
        for_scalar_target=False,
        position_key="positions",
        atom_mapping=ZEOLITE_ATOM_MAPPING,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
            atom_mapping=atom_mapping,
        )
        if remove_h:
            print("remove_h is ignored because it is not reasonble for TS.")

        single_frag_inds = np.array(range(len(self.raw_dataset["reactant"]["idx"])))
        if use_by_ind:
            use_inds = self.raw_dataset["use_ind"]
        else:
            try:
                use_inds = range(len(self.raw_dataset["single_fragment"]))
            except:
                use_inds = np.array(range(len(self.raw_dataset["reactant"]["idx"])))
        single_frag_inds = list(set(single_frag_inds).intersection(set(use_inds)))

        data_duplicated = copy.deepcopy(self.raw_dataset)
        for k, mapped_k in FRAG_MAPPING.items():
            for v, val in data_duplicated[k].items():
                self.raw_dataset[k][v] = [val[ii] for ii in single_frag_inds]
                if swapping_react_prod:
                    mapped_val = data_duplicated[mapped_k][v]
                    self.raw_dataset[k][v] += [
                        mapped_val[ii] for ii in single_frag_inds
                    ]
        if reflection:
            for k, mapped_k in FRAG_MAPPING.items():
                for v, val in self.raw_dataset[k].items():
                    if v in ["wB97x_6-31G(d).forces", position_key]:
                        self.raw_dataset[k][v] += [reflect_z(_val) for _val in val]
                    else:
                        self.raw_dataset[k][v] += val

        self.reactant = self.raw_dataset["reactant"]

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        n_samples = len(self.reactant["charges"])
        self.n_samples = len(self.reactant["charges"])

        self.data = {}
        repeat = 2 if swapping_react_prod else 1
        if for_scalar_target:
            self.data["target"] = torch.tensor(
                [
                    self.raw_dataset["target"][ii] for ii in single_frag_inds    
                ] * repeat
            ).unsqueeze(1).float()

        if not only_one:
            if not append_frag:
                self.process_molecules(
                    "reactant", n_samples, idx=0, position_key=position_key
                )
                self.process_molecules("transition_state", n_samples, idx=1)
                self.process_molecules(
                    "product", n_samples, idx=2, position_key=position_key
                )
            else:
                self.process_molecules(
                    "reactant",
                    n_samples,
                    idx=0,
                    append_charge=0,
                    position_key=position_key,
                )
                self.process_molecules(
                    "transition_state", n_samples, idx=1, append_charge=1
                )
                self.process_molecules(
                    "product",
                    n_samples,
                    idx=2,
                    append_charge=0,
                    position_key=position_key,
                )

            for idx in range(pad_fragments):
                self.patch_dummy_molecules(idx + 3)
        else:
            if not append_frag:
                self.process_molecules("reactant", n_samples, idx=0)
            else:
                self.process_molecules(
                    "reactant", n_samples, idx=0, append_charge=1
                )

        self.data["condition"] = [
            torch.zeros(size=(1, 1), dtype=torch.int64, device=self.device,)
            for _ in range(self.n_samples)
        ]