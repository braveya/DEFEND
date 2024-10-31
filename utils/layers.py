from absl import flags
import torch.nn as nn
import torch
import torch.nn.functional as F
FLAGS = flags.FLAGS

# SiameseNetwork
class SiameseNetwork(nn.Module):
    def __init__(self, ft_size, nb_nodes, nb_attribute=3, hid_units= [16, 8], hid_units_e=[4, 2], mp_att_size=128,
                 attn_heads=8, nb_paths=6, dropout_rate=0.5):
        super(SiameseNetwork, self).__init__()

        self.gat = EGAT(ft_size, nb_nodes, nb_attribute, hid_units, hid_units_e,
                         mp_att_size,attn_heads, nb_paths, dropout_rate)

    def forward_once(self, input):
        output = self.gat(input)
        return output

    def forward(self, input1, input2):
        view1a, view1b, embed1_pred = self.forward_once(input1)
        view2a, view2b, embed2_pred = self.forward_once(input2)
        return [view1a, view1b, embed1_pred], [view2a, view2b, embed2_pred]

# Hierarchical Graph Attention Network for Representation Learning
class EGAT(nn.Module):
    def __init__(self, ft_size, nb_nodes, nb_attribute=3, hid_units= [16, 8], hid_units_e=[4, 2], mp_att_size=128,
                 attn_heads=8, nb_paths=6, dropout_rate=0.5):
        super(EGAT, self).__init__()
        self.nb_nodes = nb_nodes
        self.dropout_rate = dropout_rate
        self.nb_paths = nb_paths
        self.Natt = NodeAttention(hid_units, attn_heads, nb_paths, dropout_rate, ft_size)
        self.Eatt = EdgeAttention(hid_units, hid_units_e, attn_heads, nb_attribute, dropout_rate, ft_size)

        path_output_dim = ft_size
        for i in range(len(hid_units)):
            path_output_dim += hid_units[i] * attn_heads

        self.Patt = PathAttention(mp_att_size, path_output_dim)
        self.Patt_N = PathAttention(mp_att_size, path_output_dim)
        self.Patt_E = PathAttention(mp_att_size, path_output_dim)

    def forward(self, inputs):
        X, A_list, E_list, E2N_adj = inputs
        X_chge = X
        A_chge = [A for A in A_list]
        E_chge = [E for E in E_list]
        E2N_adj_chge = E2N_adj

        node_attention = self.Natt([X_chge, A_chge[0:self.nb_paths]])
        edge_attention = self.Eatt([X_chge, A_chge[self.nb_paths:], E_chge, E2N_adj_chge])

        # embed of view1
        dropout_N = F.dropout(node_attention, self.dropout_rate, self.training)
        path_attention_N = self.Patt_N(dropout_N)

        # embed of view2
        dropout_E = F.dropout(edge_attention, self.dropout_rate, self.training)
        path_attention_E = self.Patt_E(dropout_E)

        NE_attention = torch.cat([path_attention_N.unsqueeze(1), path_attention_E.unsqueeze(1)], dim=1)
        path_attention = self.Patt(NE_attention)

        return path_attention_N, path_attention_E, path_attention


# Contrastive Loss
class Task:
    def __init__(self, meta_model, reg_param, data, nb_nodes, tau, device):
        self.fmodel = meta_model
        self.input1, self.input2, self.csc_matrix, self.cvc_matrix, self.exist_masks, self.cvc_mask1, \
        self.cvc_mask2, self.train_masks, self.val_masks, self.dec_masks  = data
        self.nb_nodes = nb_nodes
        self.tau = tau
        self.reg_param = reg_param
        self.device = device

    def predict(self, task_model):
        output1, output2 = task_model(self.input1, self.input2)
        return output1, output2

    def train_loss_f(self, meta_para, task_para):
        [view1a, view1b, embed1_pred], [view2a, view2b, embed2_pred] = self.predict(self.fmodel)

        # csc
        sim_mtx = cosine_metrix(embed1_pred, embed2_pred)
        loss1 = torch.mean(
            torch.square(sim_mtx - self.csc_matrix) * self.exist_masks * self.train_masks * self.dec_masks)

        # cvc
        mtx_st1_ab = tempcoe_metrix(view1a, view1b, self.tau).to(self.device)
        mtx_st1_ba = mtx_st1_ab.t()
        mtx_st2_ab = tempcoe_metrix(view2a, view2b, self.tau).to(self.device)
        mtx_st2_ba = mtx_st2_ab.t()
        mtx_st1_ab = mtx_st1_ab / torch.sum(mtx_st1_ab, dim=1).view(-1, 1) + 1e-8
        lori_1a = torch.sum(torch.log(torch.sum(torch.mul(mtx_st1_ab, self.cvc_matrix) * self.cvc_mask1 * self.train_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st1_ba = mtx_st1_ba / torch.sum(mtx_st1_ba, dim=1).view(-1, 1) + 1e-8
        lori_1b = torch.sum(torch.log(torch.sum(torch.mul(mtx_st1_ba, self.cvc_matrix) * self.cvc_mask1 * self.train_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st2_ab = mtx_st2_ab / torch.sum(mtx_st2_ab, dim=1).view(-1, 1) + 1e-8
        lori_2a = torch.sum(torch.log(torch.sum(torch.mul(mtx_st2_ab, self.cvc_matrix) * self.cvc_mask2 * self.train_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st2_ba = mtx_st2_ba / torch.sum(mtx_st2_ba, dim=1).view(-1, 1) + 1e-8
        lori_2b = torch.sum(torch.log(torch.sum(torch.mul(mtx_st2_ba, self.cvc_matrix) * self.cvc_mask2 * self.train_masks * self.dec_masks, dim=-1) + 1e-8))
        loss2 = (lori_1a + lori_1b + lori_2a + lori_2b) / 4
        loss_bias = bias_reg_f(task_para, meta_para)
        loss2 = loss2 + self.reg_param * loss_bias

        return loss1+loss2

    def val_loss_f(self, task_model):

        [view1a, view1b, embed1_pred], [view2a, view2b, embed2_pred] = self.predict(task_model)

        sim_mtx = cosine_metrix(embed1_pred, embed2_pred)
        loss1 = torch.mean(torch.square(sim_mtx - self.csc_matrix) * self.exist_masks * self.val_masks * self.dec_masks)

        mtx_st1_ab = tempcoe_metrix(view1a, view1b, self.tau).to(self.device)
        mtx_st1_ba = mtx_st1_ab.t()
        mtx_st2_ab = tempcoe_metrix(view2a, view2b, self.tau).to(self.device)
        mtx_st2_ba = mtx_st2_ab.t()
        mtx_st1_ab = mtx_st1_ab / torch.sum(mtx_st1_ab, dim=1).view(-1, 1) + 1e-8
        lori_1a = torch.sum(torch.log(torch.sum(torch.mul(mtx_st1_ab, self.cvc_matrix) * self.cvc_mask1 * self.val_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st1_ba = mtx_st1_ba / torch.sum(mtx_st1_ba, dim=1).view(-1, 1) + 1e-8
        lori_1b = torch.sum(torch.log(torch.sum(torch.mul(mtx_st1_ba, self.cvc_matrix) * self.cvc_mask1 * self.val_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st2_ab = mtx_st2_ab / torch.sum(mtx_st2_ab, dim=1).view(-1, 1) + 1e-8
        lori_2a = torch.sum(torch.log(torch.sum(torch.mul(mtx_st2_ab, self.cvc_matrix) * self.cvc_mask2 * self.val_masks * self.dec_masks, dim=-1) + 1e-8))
        mtx_st2_ba = mtx_st2_ba / torch.sum(mtx_st2_ba, dim=1).view(-1, 1) + 1e-8
        lori_2b = torch.sum(torch.log(torch.sum(torch.mul(mtx_st2_ba, self.cvc_matrix) * self.cvc_mask2 * self.val_masks * self.dec_masks, dim=-1) + 1e-8))
        loss2 = (lori_1a + lori_1b + lori_2a + lori_2b) / 4

        return loss1+loss2


class NodeAttention(nn.Module):
    def __init__(self,
                 hid_units = [16, 8],
                 attn_heads=8,
                 nb_paths=6,
                 dropout_rate=0.5,
                 input_dim=10,
                 skip_connection=True,
                 use_bias=True,
                 ):
        super(NodeAttention, self).__init__()

        self.hid_units = hid_units
        self.attn_heads = attn_heads
        self.nb_paths = nb_paths
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.use_bias = use_bias
        self.kernels_dict = nn.ParameterDict()
        self.biases_dict = nn.ParameterDict()
        self.attn_kernels_dict = nn.ParameterDict()

        for path in range(self.nb_paths):
            input_dim = self.input_dim
            for hid in range(len(self.hid_units)):
                # Initialize weights for each attention head
                for head in range(self.attn_heads):
                    kernel = nn.Parameter(torch.randn(input_dim, self.hid_units[hid]))
                    nn.init.xavier_uniform_(kernel, gain=1.414)
                    self.kernels_dict.update({'path{}_hid{}_head{}'.format(path, hid, head):kernel})
                    # Layer bias
                    if self.use_bias:
                        bias = nn.Parameter(torch.zeros(self.hid_units[hid],))
                        self.biases_dict.update({'path{}_hid{}_head{}'.format(path, hid, head): bias})
                    # Attention kernels
                    attn_kernel_self = nn.Parameter(torch.randn(self.hid_units[hid], 1))
                    attn_kernel_neighs = nn.Parameter(torch.randn(self.hid_units[hid], 1))
                    nn.init.xavier_uniform_(attn_kernel_self, gain=1.414)
                    nn.init.xavier_uniform_(attn_kernel_neighs, gain=1.414)
                    self.attn_kernels_dict.update({'path{}_hid{}_head{}_self'.format(path, hid, head): attn_kernel_self})
                    self.attn_kernels_dict.update({'path{}_hid{}_head{}_neighs'.format(path, hid, head): attn_kernel_neighs})
                input_dim = self.attn_heads * self.hid_units[hid]

    def forward(self, inputs):
        X = inputs[0]
        A = inputs[1]
        paths_out = []
        for path in range(self.nb_paths):
            h_list = []
            X_path = X
            A_path = A[path]
            h_list.append(X_path)
            X_head = X_path
            for hid in range(len(self.hid_units)):
                attns = []
                for head in range(self.attn_heads):
                    kernel = self.kernels_dict['path{}_hid{}_head{}'.format(path, hid, head)]
                    # Compute inputs to attention network
                    features = torch.matmul(X_head, kernel)
                    # Compute feature combinations
                    attn_for_self = torch.matmul(features, self.attn_kernels_dict['path{}_hid{}_head{}_self'.format(path, hid, head)])
                    attn_for_neighs = torch.matmul(features, self.attn_kernels_dict['path{}_hid{}_head{}_neighs'.format(path, hid, head)])
                    dense = attn_for_self + attn_for_neighs.t()
                    dense = nn.LeakyReLU(0.2)(dense)
                    dense += A_path
                    dense = F.softmax(dense, dim=-1)
                    dropout_attn = F.dropout(dense, self.dropout_rate, self.training)
                    dropout_feat = F.dropout(features, self.dropout_rate, self.training)
                    # Linear combination with neighbors' features
                    node_features = torch.matmul(dropout_attn, dropout_feat)
                    if self.use_bias:
                        node_features = torch.add(node_features, self.biases_dict['path{}_hid{}_head{}'.format(path, hid, head)])
                    node_features = F.elu(node_features)
                    # Add output of attention head to final output
                    attns.append(node_features)
                # Aggregate the heads' output according to the reduction method
                h1 = torch.cat(attns,dim=1)
                X_head = h1
                h_list.append(h1)
            h_concat = torch.cat(h_list, dim=-1)
            if self.skip_connection:
                paths_out.append(h_concat.unsqueeze(1))
            else:
                paths_out.append(h1.unsqueeze(1))
        multi_embed = torch.cat(paths_out, dim=1)
        return multi_embed


class EdgeAttention(nn.Module):
    def __init__(self,
                 hid_units=[16, 8],
                 hid_units_e=[4, 2],
                 attn_heads=8,
                 nb_attribute=3,
                 dropout_rate=0.5,
                 input_dim=10,
                 skip_connection=True,
                 use_bias=True,
                 ):
        super(EdgeAttention, self).__init__()
        self.hid_units = hid_units
        self.hid_units_e = hid_units_e
        self.attn_heads = attn_heads
        self.nb_attribute = nb_attribute
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection
        self.use_bias = use_bias
        self.input_dim = input_dim
        self.kernels_dict = nn.ParameterDict()
        self.biases_dict = nn.ParameterDict()
        self.attn_kernels_dict = nn.ParameterDict()
        self.kernels_e2n = nn.ParameterDict()
        self.biases_e2n = nn.ParameterDict()
        self.attn_kernels_e2n = nn.ParameterDict()
        self.kernels_ct = nn.ParameterDict()
        self.biases_ct = nn.ParameterDict()

        for attr in range(self.nb_attribute):
            input_dim = self.input_dim
            for hid in range(len(self.hid_units)):

                # Initialize weights for each attention head
                for head in range(self.attn_heads):
                    kernel = nn.Parameter(torch.randn(input_dim, self.hid_units[hid]))
                    nn.init.xavier_uniform_(kernel, gain=1.414)
                    self.kernels_dict.update({'attr{}_hid{}_head{}'.format(attr, hid, head): kernel})
                    if self.use_bias:
                        bias = nn.Parameter(torch.zeros(self.hid_units[hid], ))
                        self.biases_dict.update({'attr{}_hid{}_head{}'.format(attr, hid, head): bias})

                    # Attention kernels
                    attn_kernel_self = nn.Parameter(torch.randn(self.hid_units[hid], 1))
                    attn_kernel_neighs = nn.Parameter(torch.randn(self.hid_units[hid], 1))
                    nn.init.xavier_uniform_(attn_kernel_self, gain=1.414)
                    nn.init.xavier_uniform_(attn_kernel_neighs, gain=1.414)
                    self.attn_kernels_dict.update(
                        {'attr{}_hid{}_head{}_self'.format(attr, hid, head): attn_kernel_self})
                    self.attn_kernels_dict.update(
                        {'attr{}_hid{}_head{}_neighs'.format(attr, hid, head): attn_kernel_neighs})
                input_dim = self.attn_heads * (self.hid_units[hid])

        input_dim_edge = self.nb_attribute

        for hid in range(len(self.hid_units_e)):
            for head in range(self.attn_heads):
                kernel = nn.Parameter(torch.randn(input_dim_edge, self.hid_units_e[hid]))
                nn.init.xavier_uniform_(kernel, gain=1.414)
                self.kernels_e2n.update({'hid{}_head{}'.format(hid, head): kernel})
                kernels_ct = nn.Parameter(torch.randn(self.hid_units[hid] + self.hid_units_e[hid], self.hid_units[hid]))
                nn.init.xavier_uniform_(kernels_ct, gain=1.414)
                self.kernels_ct.update({'hid{}_head{}'.format(hid, head): kernels_ct})

                # Layer bias
                if self.use_bias:
                    bias = nn.Parameter(torch.zeros(self.hid_units_e[hid], ))
                    self.biases_e2n.update({'hid{}_head{}'.format(hid, head): bias})
                    bias_ct = nn.Parameter(torch.zeros(self.hid_units[hid], ))
                    self.biases_ct.update({'hid{}_head{}'.format(hid, head): bias_ct})

                #Attention kernels for edge aggregation
                attn_kernel_e2n_self = nn.Parameter(torch.randn(self.hid_units[hid], 1))
                nn.init.xavier_uniform_(attn_kernel_e2n_self, gain=1.414)
                attn_kernel_e2n_neighs = nn.Parameter(torch.randn(self.hid_units_e[hid], 1))
                nn.init.xavier_uniform_(attn_kernel_e2n_neighs, gain=1.414)
                self.attn_kernels_e2n.update({'hid{}_head{}_self'.format(hid, head): attn_kernel_e2n_self})
                self.attn_kernels_e2n.update({'hid{}_head{}_neighs'.format(hid, head): attn_kernel_e2n_neighs})

            input_dim_edge = self.attn_heads * self.hid_units_e[hid]

    def forward(self, inputs):
        X = inputs[0]
        A = inputs[1]
        E = inputs[2]
        e2n_adj = inputs[3]
        E_e2n = torch.cat([t.view(-1, 1) for t in E], dim=1)
        attrs_out = []
        for attr in range(self.nb_attribute):
            h_list = []
            X_attr = X
            E_attr = E[attr]
            A_attr = A[attr]
            h_list.append(X_attr)

            X_head = X_attr
            X_edge = E_e2n
            E_info = E_attr
            for hid in range(len(self.hid_units)):
                attns = []
                edge_feas = []
                for head in range(self.attn_heads):
                    kernel = self.kernels_dict['attr{}_hid{}_head{}'.format(attr, hid, head)]
                    features = torch.matmul(X_head, kernel)
                    kernel_e2n = self.kernels_e2n['hid{}_head{}'.format(hid, head)]
                    features_edge = torch.matmul(X_edge, kernel_e2n)
                    attn_for_self = torch.matmul(features, self.attn_kernels_e2n['hid{}_head{}_self'.format(hid, head)])
                    attn_for_edges = torch.matmul(features_edge, self.attn_kernels_e2n['hid{}_head{}_neighs'.format(hid, head)])

                    dense_e2n = attn_for_self + attn_for_edges.t()
                    dense_e2n = nn.LeakyReLU(0.2)(dense_e2n)
                    dense_e2n += e2n_adj
                    dense_e2n = F.softmax(dense_e2n, dim=-1)
                    dropout_attn_e2n = F.dropout(dense_e2n, self.dropout_rate, self.training)
                    dropout_edge = F.dropout(features_edge, self.dropout_rate, self.training)
                    node_features_e2n = torch.matmul(dropout_attn_e2n, dropout_edge)

                    if self.use_bias:
                        node_features_e2n = torch.add(node_features_e2n, self.biases_e2n['hid{}_head{}'.format(hid, head)])

                    edge_feas.append(features_edge)
                    features = torch.cat([features, node_features_e2n], dim=-1)
                    kernels_ct = self.kernels_ct['hid{}_head{}'.format(hid, head)]
                    features = torch.matmul(features, kernels_ct)

                    if self.use_bias:
                        features = torch.add(features, self.biases_ct['hid{}_head{}'.format(hid, head)])

                    # Compute feature combinations
                    attn_for_self = torch.matmul(features, self.attn_kernels_dict['attr{}_hid{}_head{}_self'.format(attr, hid, head)])
                    attn_for_neighs = torch.matmul(features, self.attn_kernels_dict['attr{}_hid{}_head{}_neighs'.format(attr, hid, head)])
                    dense = attn_for_self + attn_for_neighs.t()
                    dense = torch.matmul(dense, E_info)

                    # Add nonlinearty
                    dense = nn.LeakyReLU(0.2)(dense)

                    dense += A_attr
                    dense = F.softmax(dense, dim=-1)

                    # Apply dropout to features and attention coefficients
                    dropout_attn = F.dropout(dense, self.dropout_rate, self.training)
                    dropout_feat = F.dropout(features, self.dropout_rate, self.training)

                    # Linear combination with neighbors' features
                    node_features = torch.matmul(dropout_attn, dropout_feat)

                    if self.use_bias:
                        node_features = torch.add(node_features, self.biases_dict['attr{}_hid{}_head{}'.format(attr, hid, head)])

                    node_features = F.elu(node_features)

                    # Add output of attention head to final output
                    attns.append(node_features)

                # Aggregate the heads' output according to the reduction method
                h1 = torch.cat(attns, dim=1)
                h_list.append(h1)
                X_head = h1
                X_edge = torch.cat(edge_feas ,dim=1)
                E_info = dense

            h_concat = torch.cat(h_list, dim=-1)


            if self.skip_connection :
                attrs_out.append(h_concat.unsqueeze(1))
            else:
                attrs_out.append(h1.unsqueeze(1))

        multi_embed = torch.cat(attrs_out, dim=1)
        return multi_embed


class PathAttention(nn.Module):
    def __init__(self,
                 mp_att_size=128,
                 input_dim=10,
                 ):
        super(PathAttention, self).__init__()
        self.mp_att_size = mp_att_size
        self.input_dim = input_dim
        hidden_size = self.input_dim
        self.kernel = nn.Parameter(torch.randn(hidden_size, self.mp_att_size))
        nn.init.xavier_uniform_(self.kernel, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(self.mp_att_size,))
        self.u_omega = nn.Parameter(torch.zeros(self.mp_att_size,))


    def forward(self, inputs):
        v = torch.tanh(torch.tensordot(inputs, self.kernel, dims=1) + self.bias)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alphas = F.softmax(vu, dim=-1)
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)
        return output




def tempcoe_metrix(x1, x2, tau):

    x1_norm = torch.norm(x1, dim=-1, keepdim=True)
    x2_norm = torch.norm(x2, dim=-1, keepdim=True)

    dot_numerator = torch.matmul(x1, x2.t())
    dot_denominator = torch.matmul(x1_norm, x2_norm.t())

    sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)

    return sim_matrix


def cosine_metrix(x1, x2):
    x1_norm = torch.norm(x1, dim=-1, keepdim=True)
    x2_norm = torch.norm(x2, dim=-1, keepdim=True)
    dot_numerator = torch.matmul(x1, x2.t())
    dot_denominator = torch.matmul(x1_norm, x2_norm.t())
    cosin = dot_numerator / dot_denominator

    return cosin


def bias_reg_f(task_para, meta_para):
    return sum([((b - p) ** 2).sum() for b, p in zip(task_para, meta_para)])


