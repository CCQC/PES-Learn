from ..model import Model
from ...constants import bohr2angstroms
import re

class DiffModel(Model):
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, der_lvl=0, train_path=None, test_path=None, valid_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        nletters = re.findall(r"[A-Z]", self.molecule_type)
        nnumbers = re.findall(r"\d", self.molecule_type)
        nnumbers2 = [int(i) for i in nnumbers]
        self.natoms = len(nletters) + sum(nnumbers2) - len(nnumbers2)
        # Assuming Cartesian coordinates, and Cartesian gradients and Hessians in Bohr
        ncart = self.natoms*3
        nhess = ncart**2
        if der_lvl == 1:
            self.raw_grad = self.raw_X[:,ncart:2*ncart]
            self.raw_X = self.raw_X[:, :ncart]
        elif der_lvl == 2:
            self.raw_hess = self.raw_X[:, 2*ncart:nhess+2*ncart]
            self.raw_grad = self.raw_X[:, ncart:2*ncart]
            self.raw_X = self.raw_X[:, :ncart]
        else:
            raise ValueError(f"Error: Invalid value for `der_lvl': {der_lvl}")
