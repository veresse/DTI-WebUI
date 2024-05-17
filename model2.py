from rdkit import Chem
import sys
from gat import GAT
from transformer import *
from interformer import Decoder
from sw_tf import SwinTransformerModel
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

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=256):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1024):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def preprocess_data2(drug, protein):
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


class sw_tf(nn.Module):
    def  __init__(self,hid_dim, dropout):
        super(sw_tf,self).__init__()
        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb
        self.dropout = dropout
        self.pro_ln = nn.Linear(1024,32, dtype=torch.float32)
        self.sw_pro_embed = nn.Embedding(self.pro_emb, 64, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 64, padding_idx=0)
        self.sw_tf = SwinTransformerModel()
        self.gat = GAT()
        self.out = nn.Sequential(
            nn.Linear(576, 512),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)  # (8, 41, 64)
        sw_pro_embed = self.sw_pro_embed(prot_ids)  # (8，1024，32)
        sw_smi_embed = self.sw_smi_embed(smi_ids)  # (8，256，32)
        sw_smi = self.sw_tf(sw_smi_embed)  # (8,16,256)
        sw_pro = self.sw_tf(sw_pro_embed)  # (8,64,256)
        sw_smi = sw_smi.max(dim=1)[0]
        sw_pro = sw_pro.max(dim=1)[0]
        gat_smi = drug_gat.max(dim=1)[0]
        out_fc = torch.cat([sw_smi, sw_pro,gat_smi], dim=1)
        return self.out(out_fc)

class in_tf(nn.Module):
    def __init__(self,hid_dim, dropout):
        super(in_tf,self).__init__()
        self.hid_dim = hid_dim
        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb
        self.dropout = dropout
        self.pro_ln = nn.Linear(1024,32, dtype=torch.float32)
        self.sw_pro_embed =   nn.Embedding(self.pro_emb, 32, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 32, padding_idx=0)
        self.interformer = Decoder(32, 1, 8, 512, 0.2)
        self.out = nn.Sequential(
            nn.Linear(64, 512),
            nn.Linear(512, 2)
        )

    def forward(self, smi_ids, prot_ids):
        sw_pro_embed = self.sw_pro_embed(prot_ids)  # (8，1024，32)
        sw_smi_embed = self.sw_smi_embed(smi_ids)
        inf_pro, inf_smi = self.interformer(sw_pro_embed,sw_smi_embed)#smi(8,256,32) pro(8,1024 32)
        inf_smi = inf_smi.max(dim=1)[0]
        inf_pro = inf_pro.max(dim=1)[0]
        out_fc = torch.cat([inf_smi, inf_pro], dim=1)
        return self.out(out_fc)

class cnn(nn.Module):
    def __init__(self,hid_dim, dropout):
        super(cnn,self).__init__()
        self.hid_dim = hid_dim
        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb
        self.dropout = dropout
        self.pro_ln = nn.Linear(1024,32, dtype=torch.float32)
        self.sw_pro_embed = nn.Embedding(self.pro_emb, 64, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 64, padding_idx=0)

        self.smi_cnn = nn.Sequential(nn.Conv1d(in_channels=64,out_channels=40,kernel_size=4),
                                     nn.BatchNorm1d(40),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=40,out_channels=80,kernel_size=6),
                                     nn.BatchNorm1d(80),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=80,out_channels=160,kernel_size=8))
        self.smi_maxpool = nn.MaxPool1d(241)
        self.pro_cnn = nn.Sequential(nn.Conv1d(in_channels=64,out_channels=40,kernel_size=4),
                                     nn.BatchNorm1d(40),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=40,out_channels=80,kernel_size=8),
                                     nn.BatchNorm1d(80),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=80,out_channels=160,kernel_size=12))
        self.pro_maxpool = nn.MaxPool1d(1003)
        self.gat = GAT()
        self.out = nn.Sequential(
            nn.Linear(384, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)  # (8, 41, 64)

        pro_embed = self.sw_pro_embed(prot_ids)  # (8，1024，32)
        smi_embed = self.sw_smi_embed(smi_ids)
        pro_embed = pro_embed.permute(0, 2, 1)
        smi_embed = smi_embed.permute(0, 2, 1)
        cnn_pro = self.pro_cnn(pro_embed)
        cnn_pro = self.pro_maxpool(cnn_pro).squeeze(2)
        cnn_smi = self.smi_cnn(smi_embed)
        cnn_smi = self.smi_maxpool(cnn_smi).squeeze(2)
        gat_smi = drug_gat.max(dim=1)[0]
        out_fc= torch.cat([cnn_pro, cnn_smi,gat_smi], dim=1)
        return self.out(out_fc)

class trans(nn.Module):
    def __init__(self,hid_dim, dropout):
        super(trans,self).__init__()
        self.hid_dim = hid_dim
        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb
        self.dropout = dropout
        self.sw_pro_embed = nn.Embedding(self.pro_emb, 32, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 32, padding_idx=0)
        self.smi_cnn = transformer(32,32)
        self.pro_cnn = transformer(32,32)
        self.gat = GAT()
        self.out = nn.Sequential(
            nn.Linear(64, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, smi_ids, prot_ids):
        sw_pro_embed = self.sw_pro_embed(prot_ids)  # (8，1024，32)
        sw_smi_embed = self.sw_smi_embed(smi_ids)
        cnn_pro = self.pro_cnn(sw_pro_embed)
        cnn_smi = self.smi_cnn(sw_smi_embed)
        inf_smi = cnn_smi.max(dim=1)[0]
        inf_pro = cnn_pro.max(dim=1)[0]
        out_fc = torch.cat([inf_smi, inf_pro], dim=1)
        return self.out(out_fc)

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.hid_dim = hp.hid_dim
        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb
        self.dropout = hp.dropout
        self.model1 = sw_tf(self.hid_dim,self.dropout)
        self.model2 = in_tf(self.hid_dim,self.dropout)
        self.model3 = cnn(self.hid_dim,self.dropout)
        self.model4 = trans(self.hid_dim,self.dropout)
        self. out = nn.Sequential(
            nn.Linear(8, 512),
            nn.Dropout(hp.dropout),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(hp.dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    def forward(self, compound, adj, smi_ids, prot_ids):
        model1 = self.model1(compound, adj, smi_ids, prot_ids)
        model2 = self.model2(smi_ids, prot_ids)
        model3 = self.model3(compound, adj, smi_ids, prot_ids)
        model4 = self.model4(smi_ids, prot_ids)
        out = torch.cat([model1,model2,model3,model4], dim=1)
        return self.out(out)
    def __call__(self, data, train=True):
        compound, adj, correct_interaction, smi_ids, prot_ids, atom_num, protein_num = data
        predicted_interaction = self.forward(compound, adj, smi_ids, prot_ids)
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        return  predicted_labels

    def load_model(self, model_path):
        device = next(self.parameters()).device  # 获取模型所在设备
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


def test2(atom_feature, adj, smi_id, prot_id, model_path):
    predictor = model2()
    predictor.load_model(model_path)
    atoms_new, adjs_new, smi_ids_new, prot_ids_new = pack(atom_feature, adj, smi_id, prot_id)
    predicted_labels = predictor.forward(atoms_new, adjs_new, smi_ids_new, prot_ids_new)
    return predicted_labels



# def DTI_model(drug_smiles, protein_sequence,model_path):
#     atom_feature, adj, smi_id, prot_id = preprocess_data(drug_smiles, protein_sequence)
#     predicted_labels = test2(atom_feature, adj, smi_id, prot_id, model_path)
#     interactions = (predicted_labels[:, 1] > 0).tolist()
#     interaction_predictions = ["无法发生交互作用" if not interact else "可以发生交互作用" for interact in interactions]
#     return interaction_predictions
#
# drug="FC1=CC=CC=C1C1=CN2C=CN=C2C(NCC2=CC=CN=C2)=N1"
# protein="MVPHAILARGRDVCRRNGLLILSVLSVIVGCLLGFFLRTRRLSPQEISYFQFPGELLMRMLKMMILPLVVSSLMSGLASLDAKTSSRLGVLTVAYYLWTTFMAVIVGIFMVSIIHPGSAAQKETTEQSGKPIMSSADALLDLIRNMFPANLVEATFKQYRTKTTPVVKSPKVAPEEAPPRRILIYGVQEENGSHVQNFALDLTPPPEVVYKSEPGTSDGMNVLGIVFFSATMGIMLGRMGDSGAPLVSFCQCLNESVMKIVAVAVWYFPFGIVFLIAGKILEMDDPRAVGKKLGFYSVTVVCGLVLHGLFILPLLYFFITKKNPIVFIRGILQALLIALATSSSSATLPITFKCLLENNHIDRRIARFVLPVGATINMDGTALYEAVAAIFIAQVNNYELDFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWALDRFRTMINVLGDALAAGIMAHICRKDFARDTGTEKLLPCETKPVSLQEIVAAQQNGCVKSVAEASELTLGPTCPHHVPVQVEQDEELPAASLNHCTIQISELETNV"
# model_path ="D:/Program Files/PycharmProject/DTI_Webui/model/model_2.pt"
# output = DTI_model(drug,protein,model_path)
# print(output)
