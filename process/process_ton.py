import numpy as np
import os
from torch.utils.data import Dataset
from utils.config_lib import set_flags

out_path = 'data/TonIoT'
class toniot_train_dec(Dataset):
    def __init__(self, idx, repeat=1):
        super(toniot_train_dec, self).__init__()

        nb_shot = 52
        nb_node = 43
        nb_fea = 71
        n = idx

        save_path_n = os.path.join(out_path, f'agent{n}')

        self.csc_matrix = np.load(os.path.join(save_path_n, 'csc_matrix.npy')).reshape(nb_shot, nb_node, nb_node)
        self.cvc_matrix = np.load(os.path.join(save_path_n, 'cvc_matrix.npy')).reshape(nb_shot, nb_node, nb_node)
        self.exist_masks = np.load(os.path.join(save_path_n, 'exist_masks.npy')).reshape(nb_shot, nb_node, nb_node)
        self.cvc_mask1 = np.load(os.path.join(save_path_n, 'cvc_mask1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.cvc_mask2 = np.load(os.path.join(save_path_n, 'cvc_mask2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.train_masks = np.load(os.path.join(save_path_n, 'train_masks.npy')).reshape(nb_shot, nb_node, nb_node)
        self.val_masks = np.load(os.path.join(save_path_n, 'val_masks.npy')).reshape(nb_shot, nb_node, nb_node)
        self.dec_mask_agents = np.load(os.path.join(save_path_n, 'dec_mask_agents.npy')).reshape(nb_shot, nb_node, nb_node)

        save_path_tr = os.path.join(save_path_n, 'tr')
        self.X1 = np.load(os.path.join(save_path_tr, 'X1.npy')).reshape(nb_shot, nb_node, nb_fea)
        self.B1_p1 = np.load(os.path.join(save_path_tr, 'B1_p1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p2 = np.load(os.path.join(save_path_tr, 'B1_p2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p3 = np.load(os.path.join(save_path_tr, 'B1_p3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p4 = np.load(os.path.join(save_path_tr, 'B1_p4.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p5 = np.load(os.path.join(save_path_tr, 'B1_p5.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p6 = np.load(os.path.join(save_path_tr, 'B1_p6.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e1 = np.load(os.path.join(save_path_tr, 'B1_e1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e2 = np.load(os.path.join(save_path_tr, 'B1_e2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e3 = np.load(os.path.join(save_path_tr, 'B1_e3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e_adj = np.load(os.path.join(save_path_tr, 'B1_e_adj.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_E2N_adj = np.load(os.path.join(save_path_tr, 'B1_E2N_adj.npy')).reshape(nb_shot, nb_node, nb_node*nb_node)
        save_path_tr_b = os.path.join(save_path_n, 'tr_b')
        self.X2 = np.load(os.path.join(save_path_tr_b, 'X2.npy')).reshape(nb_shot, nb_node, nb_fea)
        self.B2_p1 = np.load(os.path.join(save_path_tr_b, 'B2_p1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_p2 = np.load(os.path.join(save_path_tr_b, 'B2_p2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_p3 = np.load(os.path.join(save_path_tr_b, 'B2_p3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_p4 = np.load(os.path.join(save_path_tr_b, 'B2_p4.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_p5 = np.load(os.path.join(save_path_tr_b, 'B2_p5.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_p6 = np.load(os.path.join(save_path_tr_b, 'B2_p6.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_e1 = np.load(os.path.join(save_path_tr_b, 'B2_e1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_e2 = np.load(os.path.join(save_path_tr_b, 'B2_e2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_e3 = np.load(os.path.join(save_path_tr_b, 'B2_e3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_e_adj = np.load(os.path.join(save_path_tr_b, 'B2_e_adj.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B2_E2N_adj = np.load(os.path.join(save_path_tr_b, 'B2_E2N_adj.npy')).reshape(nb_shot, nb_node,
                                                                                        nb_node * nb_node)

        self.repeat = repeat
        self.lenth = len(self.X1)

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.X1) * self.repeat
        return data_len

    def __getitem__(self, i):
        index = i % self.lenth
        # subnetwork1
        X1 = self.X1[index]
        B1_p1 = self.B1_p1[index]
        B1_p2 = self.B1_p2[index]
        B1_p3 = self.B1_p3[index]
        B1_p4 = self.B1_p4[index]
        B1_p5 = self.B1_p5[index]
        B1_p6 = self.B1_p6[index]
        B1_e1 = self.B1_e1[index]
        B1_e2 = self.B1_e2[index]
        B1_e3 = self.B1_e3[index]
        B1_e_adj = self.B1_e_adj[index]
        B1_E2N_adj = self.B1_E2N_adj[index]

        # subnetwork2
        X2 = self.X2[index]
        B2_p1 = self.B2_p1[index]
        B2_p2 = self.B2_p2[index]
        B2_p3 = self.B2_p3[index]
        B2_p4 = self.B2_p4[index]
        B2_p5 = self.B2_p5[index]
        B2_p6 = self.B2_p6[index]
        B2_e1 = self.B2_e1[index]
        B2_e2 = self.B2_e2[index]
        B2_e3 = self.B2_e3[index]
        B2_e_adj = self.B2_e_adj[index]
        B2_E2N_adj = self.B2_E2N_adj[index]

        csc_matrix = self.csc_matrix[index]
        cvc_matrix = self.cvc_matrix[index]
        exist_masks = self.exist_masks[index]
        cvc_mask1 = self.cvc_mask1[index]
        cvc_mask2 = self.cvc_mask2[index]
        train_masks = self.train_masks[index]
        val_masks = self.val_masks[index]
        dec_mask_agents = self.dec_mask_agents[index]

        return X1, B1_p1, B1_p2, B1_p3, B1_p4, B1_p5, B1_p6, B1_e1, B1_e2, B1_e3, B1_e_adj, B1_E2N_adj, \
               X2, B2_p1, B2_p2, B2_p3, B2_p4, B2_p5, B2_p6, B2_e1, B2_e2, B2_e3, B2_e_adj, B2_E2N_adj, \
               csc_matrix, cvc_matrix, exist_masks, cvc_mask1, cvc_mask2, train_masks, val_masks, dec_mask_agents


class toniot_test_dec(Dataset):
    def __init__(self, idx, repeat):
        super(toniot_test_dec, self).__init__()

        nb_shot = 52
        nb_node = 43
        nb_fea = 71
        n = idx

        save_path_n = os.path.join(out_path, f'agent{n}')

        save_path_tr = os.path.join(save_path_n, 'tr')
        self.X1 = np.load(os.path.join(save_path_tr, 'X1.npy')).reshape(nb_shot, nb_node, nb_fea)
        self.B1_p1 = np.load(os.path.join(save_path_tr, 'B1_p1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p2 = np.load(os.path.join(save_path_tr, 'B1_p2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p3 = np.load(os.path.join(save_path_tr, 'B1_p3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p4 = np.load(os.path.join(save_path_tr, 'B1_p4.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p5 = np.load(os.path.join(save_path_tr, 'B1_p5.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_p6 = np.load(os.path.join(save_path_tr, 'B1_p6.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e1 = np.load(os.path.join(save_path_tr, 'B1_e1.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e2 = np.load(os.path.join(save_path_tr, 'B1_e2.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e3 = np.load(os.path.join(save_path_tr, 'B1_e3.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_e_adj = np.load(os.path.join(save_path_tr, 'B1_e_adj.npy')).reshape(nb_shot, nb_node, nb_node)
        self.B1_E2N_adj = np.load(os.path.join(save_path_tr, 'B1_E2N_adj.npy')).reshape(nb_shot, nb_node,
                                                                                        nb_node * nb_node)

        save_path_ts = os.path.join(save_path_n, 'ts')
        nb_shot_ts = 37
        self.ts_mask = np.load(os.path.join(save_path_ts, 'ts_mask.npy'))
        self.ts_X = np.load(os.path.join(save_path_ts, 'ts_X.npy')).reshape(nb_shot_ts, nb_node, nb_fea)
        self.ts_B_p1 = np.load(os.path.join(save_path_ts, 'ts_B_p1.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_B_p2 = np.load(os.path.join(save_path_ts, 'ts_B_p2.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_B_p3 = np.load(os.path.join(save_path_ts, 'ts_B_p3.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_B_p4 = np.load(os.path.join(save_path_ts, 'ts_B_p4.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_B_p5 = np.load(os.path.join(save_path_ts, 'ts_B_p5.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_B_p6 = np.load(os.path.join(save_path_ts, 'ts_B_p6.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_e1 = np.load(os.path.join(save_path_ts, 'ts_e1.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_e2 = np.load(os.path.join(save_path_ts, 'ts_e2.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_e3 = np.load(os.path.join(save_path_ts, 'ts_e3.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_e_adj = np.load(os.path.join(save_path_ts, 'ts_e_adj.npy')).reshape(nb_shot_ts, nb_node, nb_node)
        self.ts_E2N_adj = np.load(os.path.join(save_path_ts, 'ts_E2N_adj.npy')).reshape(nb_shot_ts, nb_node,
                                                                                        nb_node * nb_node)
        
        self.repeat = repeat
        self.lenth = len(self.X1)

        self.y_true = np.load(os.path.join(save_path_n, 'y_true.npy'))
        self.tr_mask = np.load(os.path.join(save_path_n, 'tr_mask.npy'))
        self.dec_mask_ts = np.load(os.path.join(save_path_n, 'dec_mask_ts.npy'))


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.X1) * self.repeat
        return data_len

    def __getitem__(self, i):
        index = i % self.lenth
        return self.X1, self.B1_p1, self.B1_p2, self.B1_p3, self.B1_p4, self.B1_p5, self.B1_p6, self.B1_e1, self.B1_e2, \
               self.B1_e3, self.B1_e_adj, self.B1_E2N_adj, self.ts_X, self.ts_mask, self.ts_B_p1, self.ts_B_p2, \
               self.ts_B_p3, self.ts_B_p4, self.ts_B_p5, self.ts_B_p6, self.ts_e1, self.ts_e2, self.ts_e3, \
               self.ts_e_adj, self.ts_E2N_adj, self.tr_mask, self.y_true, self.dec_mask_ts
