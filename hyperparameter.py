class hyperparameter():
    def __init__(self):
        self.model_name = "DrugBank"
        self.hid_dim = 64
        self.n_layers = 3 
        self.n_heads = 8
        self.pf_dim = 256 
        self.dropout = 0.1
        self.batch = 32 
        self.lr = 1e-4 #1e-4
        self.weight_decay = 1e-6 #1e-6
        self.iteration = 100 
        self.n_folds = 5
        self.seed = 2021
        self.save_name = "test"
        self.MAX_PROTEIN_LEN = 1024
        self.MAX_DRUG_LEN = 256
        self.conv = 34
        self.pro_emb = 9050
        self.smi_emb = 82