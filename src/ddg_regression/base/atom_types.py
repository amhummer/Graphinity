import struct

from openbabel import openbabel
import pandas as pd
from openbabel import pybel
from Bio.SeqUtils import seq1
from biopandas.pdb import PandasPdb

# saves info about specific atom type
class Info:
    def __init__(
        self,
        sm,
        smina_name,
        adname,
        anum,
        ad_radius,
        ad_depth,
        ad_solvation,
        ad_volume,
        covalent_radius,
        xs_radius,
        xs_hydrophobe,
        xs_donor,
        xs_acceptor,
        ad_heteroatom,
    ):
        self.sm = sm
        self.smina_name = smina_name  # this must be more than 2 chars long
        self.adname = adname  # this must be no longer than 2 chars
        self.anum = anum
        self.ad_radius = ad_radius
        self.ad_depth = ad_depth
        self.ad_solvation = ad_solvation
        self.ad_volume = ad_volume
        self.covalent_radius = covalent_radius
        self.xs_radius = xs_radius
        self.xs_hydrophobe = xs_hydrophobe
        self.xs_donor = xs_donor
        self.xs_acceptor = xs_acceptor
        self.ad_heteroatom = ad_heteroatom

class Typer:
    """Python reimplementation of the gninatyper function,
    as per https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740
    """
    def __init__(self):
        # self.etab = openbabel.OBElementTable()
        self.non_ad_metal_names = [
            "Cu",
            "Fe",
            "Na",
            "K",
            "Hg",
            "Co",
            "U",
            "Cd",
            "Ni",
            "Si",
        ]
        self.atom_equivalence_data = [("Se", "S")]
        self.atom_type_data = [
            Info(
                "Hydrogen",
                "Hydrogen",
                "H",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "PolarHydrogen",
                "PolarHydrogen",
                "HD",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSHydrophobe",
                "AliphaticCarbonXSHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSNonHydrophobe",
                "AliphaticCarbonXSNonHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSHydrophobe",
                "AromaticCarbonXSHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSNonHydrophobe",
                "AromaticCarbonXSNonHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "Nitrogen",
                "Nitrogen",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonor",
                "NitrogenXSDonor",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonorAcceptor",
                "NitrogenXSDonorAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "NitrogenXSAcceptor",
                "NitrogenXSAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Oxygen",
                "Oxygen",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "OxygenXSDonor",
                "OxygenXSDonor",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "OxygenXSDonorAcceptor",
                "OxygenXSDonorAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "OxygenXSAcceptor",
                "OxygenXSAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Sulfur",
                "Sulfur",
                "S",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "SulfurAcceptor",
                "SulfurAcceptor",
                "SA",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Phosphorus",
                "Phosphorus",
                "P",
                15,
                2.100000,
                0.200000,
                -0.001100,
                38.792400,
                1.060000,
                2.100000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Fluorine",
                "Fluorine",
                "F",
                9,
                1.545000,
                0.080000,
                -0.001100,
                15.448000,
                0.710000,
                1.500000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Chlorine",
                "Chlorine",
                "Cl",
                17,
                2.045000,
                0.276000,
                -0.001100,
                35.823500,
                0.990000,
                1.800000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Bromine",
                "Bromine",
                "Br",
                35,
                2.165000,
                0.389000,
                -0.001100,
                42.566100,
                1.140000,
                2.000000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Iodine",
                "Iodine",
                "I",
                53,
                2.360000,
                0.550000,
                -0.001100,
                55.058500,
                1.330000,
                2.200000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Magnesium",
                "Magnesium",
                "Mg",
                12,
                0.650000,
                0.875000,
                -0.001100,
                1.560000,
                1.300000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Manganese",
                "Manganese",
                "Mn",
                25,
                0.650000,
                0.875000,
                -0.001100,
                2.140000,
                1.390000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Zinc",
                "Zinc",
                "Zn",
                30,
                0.740000,
                0.550000,
                -0.001100,
                1.700000,
                1.310000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Calcium",
                "Calcium",
                "Ca",
                20,
                0.990000,
                0.550000,
                -0.001100,
                2.770000,
                1.740000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Iron",
                "Iron",
                "Fe",
                26,
                0.650000,
                0.010000,
                -0.001100,
                1.840000,
                1.250000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "GenericMetal",
                "GenericMetal",
                "M",
                0,
                1.200000,
                0.000000,
                -0.001100,
                22.449300,
                1.750000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            # note AD4 doesn't have boron, so copying from carbon
            Info(
                "Boron",
                "Boron",
                "B",
                5,
                2.04,
                0.180000,
                -0.0011,
                12.052,
                0.90,
                1.920000,
                True,
                False,
                False,
                False,
            ),
        ]
        self.atom_types = [info.sm for info in self.atom_type_data]

    
    def string_to_smina_type(self, string: str):
        """Convert string type to smina type

        Args:
            string (str): string type

        Returns:
            string: smina type
        """
        if len(string) <= 2:
            for type_info in self.atom_type_data:
                # convert ad names to smina types
                if string == type_info.adname:
                    return type_info.sm
            # find equivalent atoms
            for i in self.atom_equivalence_data:
                if string == i[0]:
                    return self.string_to_smina_type(i[1])
            # generic metal
            if string in self.non_ad_metal_names:
                return "GenericMetal"
            # if nothing else found --> generic metal
            return "GenericMetal"

        else:
            # assume it's smina name
            for type_info in self.atom_type_data:
                if string == type_info.smina_name:
                    return type_info.sm
            # if nothing else found, return numtypes
            # technically not necessary to call this numtypes,
            # but including this here to make it equivalent to the cpp code
            return "NumTypes"

    def adjust_smina_type(self, t, hBonded, heteroBonded):
        if (
            t == "AliphaticCarbonXSNonHydrophobe" or t == "AliphaticCarbonXSHydrophobe"
        ):  # C_C_C_P,
            if heteroBonded:
                return "AliphaticCarbonXSNonHydrophobe"
            else:
                return "AliphaticCarbonXSHydrophobe"
        elif (
            t == "AromaticCarbonXSNonHydrophobe" or t == "AromaticCarbonXSHydrophobe"
        ):  # C_A_C_P,
            if heteroBonded:
                return "AromaticCarbonXSNonHydrophobe"
            else:
                return "AromaticCarbonXSHydrophobe"
        elif t == "Nitrogen" or t == "NitogenXSDonor":  # N_N_N_P, no hydrogen bonding
            if hBonded:
                return "NitrogenXSDonor"
            else:
                return "Nitrogen"
        elif (
            t == "NitrogenXSAcceptor" or t == "NitrogenXSDonorAcceptor"
        ):  # N_NA_N_A, also considered an acceptor by autodock
            if hBonded:
                return "NitrogenXSDonorAcceptor"
            else:
                return "NitrogenXSAcceptor"
        elif t == "Oxygen" or t == "OxygenXSDonor":  # O_O_O_P,
            if hBonded:
                return "OxygenXSDonor"
            else:
                return "Oxygen"
        elif (
            t == "OxygenXSAcceptor" or t == "OxygenXSDonorAcceptor"
        ):  # O_OA_O_A, also an autodock acceptor
            if hBonded:
                return "OxygenXSDonorAcceptor"
            else:
                return "OxygenXSAcceptor"
        else:
            return t

    def obatom_to_smina_type(self, ob_atom):
        # obtain atom names via openbabel
        num = ob_atom.atomicnum
        ename = openbabel.GetSymbol(num)
        if num == 1: # hydrogen
            ename = "HD"
        elif num == 6 and ob_atom.OBAtom.IsAromatic(): # aromatic carbon
            ename = "A"
        elif num == 8: # oxygen
            ename = "OA"
        elif num == 7 and ob_atom.OBAtom.IsHbondAcceptor(): # nitrogen H-bond acceptor
            ename = "NA"
        elif num == 16 and ob_atom.OBAtom.IsHbondAcceptor(): # sulphur H-bond acceptor
            ename = "SA"
        # convert atom name to smina typ
        atype = self.string_to_smina_type(ename)

        hBonded = False
        heteroBonded = False
        for neighbour in openbabel.OBAtomAtomIter(ob_atom.OBAtom):
            if neighbour.GetAtomicNum() == 1:
                hBonded = True
            elif neighbour.GetAtomicNum() != 6:
                heteroBonded = True

        return self.adjust_smina_type(atype, hBonded, heteroBonded)

    def read_file(self, infile: str, add_hydrogens: bool):
        """Use openbabel to read in a pdb file.

        Args:
            infile (str): Path to input file
            add_hydrogens (bool): Add hydrogens to the openbabel OBMol object

        Returns:
            pybel.Molecule
        """
        molecules = []

        # need to convert to string, openbabel cannot handle Path
        file_read = pybel.readfile("pdb", str(infile))

        for mol in file_read:
            molecules.append(mol)

        if len(molecules) != 1:
            print(
                "Something went wrong with %s. There should be 1 molecule, "
                "but there are %d" % (infile, len(molecules))
            )
            with open("more_than_one_mol.txt", "a") as log_mol:
                log_mol.write(infile)
                log_mol.write("\n")
            return 1

        mol = molecules[0]

        if add_hydrogens:
            mol.OBMol.AddHydrogens()
        return mol

    def get_types(
        self,
        mol,
        inf: str,
        return_occupancy_value: bool,
    ):
        """Write mol object to types file or parquet dataframe.

        Args:
            mol (pybel.Molecule): molecule object to be written to file
            inf (str): Path to pdb file, only necessary for occupancy value saving
            return_occupancy_value (bool): If True, save the value in columns 55-58 of the pdb.
        """
        types = []
        occupancy_values = []
        
        pdb_types = []
        res_numbers = []
        res_types = []
        xs = []
        ys = []
        zs = []

        for atom in mol:
            smina_type = self.obatom_to_smina_type(atom)
            if smina_type == "NumTypes":
                smina_type_int = len(self.atom_type_data)
            else:
                smina_type_int = self.atom_types.index(smina_type)

            pdb_type = atom.residue.OBResidue.GetAtomID(atom.OBAtom).strip()
            res_number = atom.residue.idx
            res_type = seq1(atom.residue.name) # convert AA 3 to 1 letter code

            # excluding H in the pdb annotation --> this might have to be changed
            if pdb_type != 'H':
                xs.append(atom.coords[0])
                ys.append(atom.coords[1])
                zs.append(atom.coords[2])

                pdb_types.append(pdb_type)
                res_numbers.append(res_number)
                res_types.append(res_type)

                types.append(smina_type_int)

        df = pd.DataFrame()
        df["x"] = xs
        df["y"] = ys
        df["z"] = zs
        df["types"] = types
        df["pdb_type"] = pdb_types
        df["res_type"] = res_types
        df["res_number"] = res_numbers
        
        if return_occupancy_value:
            pdb_df = PandasPdb().read_pdb(str(inf)).df["ATOM"]
            pdb_df = pdb_df.set_index(["x_coord", "y_coord", "z_coord"])
            for i in range(len(df)):
                row = df.iloc[i]

                occupancy = int(
                    pdb_df.loc[row["x"], row["y"], row["z"]]["occupancy"]
                )
                occupancy_values.append(occupancy)
        
        return types, occupancy_values
                
        

    def run(
        self,
        inf: str,
        return_occupancy_value: bool = True,
        add_hydrogens: bool = True,
    ):
        """Run typer on pdb file.

        Args:
            inf (str): Path to pdb file
            return_occupancy_value (bool, optional): Save the value in columns 55-58 of the pdb.
                Defaults to True.
            add_hydrogens (bool, optional): Add hydrogens using OpenBabel.
                Defaults to True

        Raises:
            ValueError: If file reading fails, raises ValueError

        """
        openbabel.obErrorLog.SetOutputLevel(openbabel.obError)
        mol = self.read_file(inf, add_hydrogens)

        if type(mol) == int:
            raise ValueError(f"{inf} returned with error code {mol}")
        
        # run typing on molecule
        types, occupancies = self.get_types(
            mol, inf, return_occupancy_value
        )
        return types, occupancies
