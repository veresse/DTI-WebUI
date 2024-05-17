from rdkit import Chem
import sys
from gat import GAT
from transformer import *
import torch
import numpy as np
import os
from hyperparameter import hyperparameter

hp = hyperparameter()
sys.path.append('..')
sys.path.append('..')
num_atom_feat = 34
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,explicit_H=False,use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),                                           [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=600):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def preprocess_data1(drug, protein):
    smiles, sequence = drug, protein
    smi_id = torch.unsqueeze(torch.from_numpy(label_smiles(smiles, CHARISOSMISET)), dim=0)
    prot_id = torch.unsqueeze(torch.from_numpy(label_sequence(sequence, CHARPROTSET)), dim=0)
    atom_feature, adj = mol_features(smiles)
    atom_feature = torch.unsqueeze(torch.FloatTensor(atom_feature), dim=0)
    adj = torch.unsqueeze(torch.FloatTensor(adj), dim=0)
    if len(atom_feature.shape) == 2:
        atom_feature = atom_feature.reshape(1, *atom_feature.shape)
    if len(adj.shape) == 2:
        adj = adj.reshape(1, *adj.shape)
    return [atom_feature, adj, smi_id, prot_id]


class Predictor(nn.Module):
    def __init__(self, dropout):
        super(Predictor, self).__init__()

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.dropout = dropout
        self.hid_dim = hp.hid_dim
        self.pro_em = hp.pro_emb
        self.smi_em = hp.smi_emb
        self.prot_embed = nn.Embedding(self.pro_em, self.hid_dim, padding_idx=0)
        self.smi_embed = nn.Embedding(self.smi_em, self.hid_dim, padding_idx=0)
        self.smi_tf = transformer(self.hid_dim, self.hid_dim)
        self.pro_tf = transformer(self.hid_dim, self.hid_dim)
        self.gat = GAT()
        self.sigmoid = nn.Sigmoid()
        self.att_layer = torch.nn.Linear(64, 64)

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)

        prot_embed = self.prot_embed(prot_ids)
        smi_embed = self.smi_embed(smi_ids)

        smi_tf = self.smi_tf(smi_embed)
        pro_tf = self.pro_tf(prot_embed)

        smi_att = self.att_layer(smi_tf)
        pro_att = self.att_layer(pro_tf)

        smi_attss = torch.cat([smi_att, drug_gat], dim=1)

        protein = torch.unsqueeze(pro_att, 1).repeat(1, smi_attss.shape[-2], 1, 1)
        drug = torch.unsqueeze(smi_attss, 2).repeat(1, 1, pro_tf.shape[-2], 1)
        Atten_matrix = self.att_layer(self.relu(protein + drug))
        smi_atts = self.sigmoid(torch.mean(Atten_matrix, 2))
        pro_atts = self.sigmoid(torch.mean(Atten_matrix, 1))

        smi_tfs = 0.5 * smi_attss + smi_attss * smi_atts
        pro_tfs = 0.5 * pro_tf + pro_tf * pro_atts

        smi = smi_tfs.mean(dim=1)
        pro = pro_tfs.mean(dim=1)

        out = torch.cat([smi, pro], dim=-1)
        return self.out(out)
    def __call__(self, data, train=True):
        compound, adj, correct_interaction, smi_ids, prot_ids, atom_num, protein_num = data
        predicted_interaction = self.forward(compound, adj, smi_ids, prot_ids)
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        return  predicted_labels

    def load_model(self, model_path):
        device = next(self.parameters()).device
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

def pack(atoms, adjs, smi_ids, prot_ids):
    atoms_len = 0
    proteins_len = 0
    N = 1
    atom_num, protein_num = [], []
    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    if atoms_len>hp.MAX_DRUG_LEN: atoms_len = hp.MAX_DRUG_LEN
    atoms_new = torch.zeros((N,atoms_len,34))
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len>atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len>atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1
    if proteins_len>hp.MAX_PROTEIN_LEN: proteins_len = hp.MAX_PROTEIN_LEN
    proteins_new = torch.zeros((N, proteins_len, 100))
    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)
    if smi_id_len>hp.MAX_DRUG_LEN: smi_id_len = hp.MAX_DRUG_LEN
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long)
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len>smi_id_len: t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)
    if prot_id_len>hp.MAX_PROTEIN_LEN: prot_id_len = hp.MAX_PROTEIN_LEN
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long)
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len>prot_id_len: t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]
    return atoms_new, adjs_new, smi_ids_new, prot_ids_new


def test1(atom_feature, adj, smi_id, prot_id, model_path):
    predictor = Predictor(dropout=hp.dropout)
    predictor.load_model(model_path)
    atoms_new, adjs_new, smi_ids_new, prot_ids_new = pack(atom_feature, adj, smi_id, prot_id)
    predicted_labels = predictor.forward(atoms_new, adjs_new, smi_ids_new, prot_ids_new)
    return predicted_labels

# def DTI_model(drug_smiles, protein_sequence,model_path):
#     atom_feature, adj, smi_id, prot_id = preprocess_data(drug_smiles, protein_sequence)
#     predicted_labels = test1(atom_feature, adj, smi_id, prot_id, model_path)
#     interactions = (predicted_labels[:, 1] > 0).tolist()
#     interaction_predictions = ["无法发生交互作用" if not interact else "可以发生交互作用" for interact in interactions]
#     return interaction_predictions
#
# drug="[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])CC[C@@H]2O"
# protein="MQKRAIYPGTFDPITNGHIDIVTRATQMFDHVILAIAASPSKKPMFTLEERVALAQQATAHLGNVEVVGFSDLMANFARNQHATVLIRGLRAVADFEYEMQLAHMNRHLMPELESVFLMPSKEWSFISSSLVKEVARHQGDVTHFLPENVHQALMAKLA"
# model_path ="D:/Program Files/PycharmProject/DTI_Webui/model/model_1.pt"
# output = DTI_model(drug,protein,model_path)
# print(output)





