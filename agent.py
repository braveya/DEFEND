import copy
from utils.layers import SiameseNetwork, Task
from utils.functions import *
from itertools import repeat

class agent(object):
    def __init__(self):
        self.local_parameters = None

    def initia(self, id, device, ft_size, nb_nodes):

        self.id = id
        self.device = device
        self.N = 5
        self.batch_size = 16
        self.theta_lr = 1e-4
        self.reg_param = 1e-4
        self.lag_lamda_lr = 1e-4
        self.params_lr = 1e-4
        self.ft_size = ft_size
        self.nb_nodes = nb_nodes
        self.nb_attribute = 3
        self.hid_units = [16, 8]
        self.hid_units_e = [4, 2]
        self.mp_att_size = 128
        self.n_attn_heads = 8
        self.nb_paths = 6
        self.dropout_rate = 0.5
        self.tau = 0.5
        self.errta = 1e-4
        self.inner_lr = 1e-4
        self.lag_init = 1e-4
        self.outter_lr = 1e-4
        self.val_loss = 0
        self.meta_model = SiameseNetwork(ft_size, nb_nodes, self.nb_attribute, self.hid_units, self.hid_units_e,
                                         self.mp_att_size, self.n_attn_heads, self.nb_paths, self.dropout_rate)
        self.outer_opt = torch.optim.Adam(params=self.meta_model.parameters(), lr=self.outter_lr, weight_decay=1e-4)
        self.con_hpx = []
        self.con_hpy = []
        self.con_c = []
        self.lag_all = []
        self.W = []
        self.x = [[] for i in range(self.N)]
        self.new_x = []
        self.theta = [[[] for i in range(self.N)] for j in range(self.N)]
        self.new_theta = [[[] for i in range(self.N)] for j in range(self.N)]
        self.aver_x = []
        for p in self.meta_model.parameters():
            self.aver_x.append(torch.zeros_like(p))

    def dual_val(self, hparams, theta):
        W = torch.Tensor(self.W)
        zero = torch.zeros_like(W)
        one = torch.ones_like(W)
        w_mask = torch.where(W > 0, one, zero)
        sum = 0
        for i in range(self.N):
            sum = sum + (w_mask[self.id][i] * ((cat_list_to_tensor(hparams) - cat_list_to_tensor(
                self.x[i])) * cat_list_to_tensor(theta[i]))).sum()
        return sum

    def loc_upd(self, batch_data, active=True):

        self.outer_opt.zero_grad()

        if active is True:

            for d in range(len(batch_data)):
                batch_data[d] = batch_data[d].to(self.device)

            X1, B1_p1, B1_p2, B1_p3, B1_p4, B1_p5, B1_p6, B1_e1, B1_e2, B1_e3, B1_e_adj, B1_E2N_adj, \
            X2, B2_p1, B2_p2, B2_p3, B2_p4, B2_p5, B2_p6, B2_e1, B2_e2, B2_e3, B2_e_adj, B2_E2N_adj, \
            csc_matrix, cvc_matrix, exist_masks, cvc_mask1, cvc_mask2, train_masks, val_masks, dec_masks =  batch_data

            self.meta_model.train()
            self.meta_model.gat.train()

            for i in range(len(self.aver_x)):
                self.aver_x[i] = torch.zeros_like(self.aver_x[i]).to(self.device)
                for j in range(self.N):
                    self.aver_x[i] = self.aver_x[i] + self.W[self.id][j] * self.x[j][i]
            for (p, q) in zip(self.meta_model.parameters(), self.aver_x):
                p.data = q.data.detach().clone()

            meta_model_h = copy.deepcopy(self.meta_model)

            val_loss_batch = 0
            for batch in range(self.batch_size):
                task_model = copy.deepcopy(self.meta_model).to(self.device)

                theta = []

                for i in range(self.N):
                    theta.append([p.detach().clone().requires_grad_(True) for p in self.theta[self.id][i]])

                input1 = [X1[batch], [B1_p1[batch], B1_p2[batch], B1_p3[batch], B1_p4[batch], B1_p5[batch], B1_p6[batch],
                                      B1_e_adj[batch], B1_e_adj[batch], B1_e_adj[batch]],
                                      [B1_e1[batch], B1_e2[batch], B1_e3[batch]], B1_E2N_adj[batch]]

                input2 = [X2[batch], [B2_p1[batch], B2_p2[batch], B2_p3[batch], B2_p4[batch], B2_p5[batch], B2_p6[batch],
                                      B2_e_adj[batch], B2_e_adj[batch], B2_e_adj[batch]],
                                      [B2_e1[batch], B2_e2[batch], B2_e3[batch]], B2_E2N_adj[batch]]

                task = Task(task_model, self.reg_param, [input1, input2,
                                                         csc_matrix[batch], cvc_matrix[batch],
                                                         exist_masks[batch], cvc_mask1[batch],
                                                         cvc_mask2[batch], train_masks[batch],
                                                         val_masks[batch], dec_masks[batch]],
                                                         self.nb_nodes, self.tau, self.device)

                hparams = [p for p in meta_model_h.parameters()]
                val_loss_i = task.val_loss_f(task_model)
                o_loss = val_loss_i + self.dual_val(hparams, theta)
                val_loss_batch += val_loss_i.item()

                # cutting plane
                if len(self.con_hpx) > 0:
                    stackx = torch.stack(self.con_hpx).to(self.device)
                    stacky = torch.stack(self.con_hpy).to(self.device)
                    conc = torch.tensor(self.con_c).to(self.device)
                    x0 = cat_list_to_tensor(p for p in self.meta_model.parameters()).to(self.device)
                    y0 = cat_list_to_tensor(p.detach().clone().requires_grad_(True) for p in
                                            task_model.parameters()).to(self.device)
                    lag_bias = torch.matmul(stacky, y0) + torch.matmul(stackx, x0) + conc
                    lag_lamda = cat_list_to_tensor(self.lag_all).detach().clone().requires_grad_(True).to(self.device)
                    o_loss = o_loss + torch.matmul(lag_bias, lag_lamda)
                    grad_lamda = torch.autograd.grad(o_loss, lag_lamda, allow_unused=True, retain_graph=True)
                    grad_lamda = cat_list_to_tensor(grad_lamda)
                    lag_lamda = lag_lamda + self.lag_lamda_lr * grad_lamda
                    for num, lag in enumerate(self.lag_all):
                        if lag_lamda[num] > 0:
                            self.lag_all[num] = lag_lamda[num]
                        else:
                            self.lag_all[num] = torch.tensor(0.).to(self.device)

                o_loss.backward(retain_graph=True)
                grad_outer_w = torch.autograd.grad(o_loss, task_model.parameters())

                for g, p in zip(grad_outer_w, task_model.parameters()):
                    if g is None:
                        continue
                    p.data = p.data - self.params_lr * g.data

                xapp = [p.detach().clone().requires_grad_(True) for p in meta_model_h.parameters()]
                yapp = [p.detach().clone().requires_grad_(True) for p in task_model.parameters()]
                inner_opt = GD(task.train_loss_f, self.inner_lr)
                phix = inner_opt(xapp, xapp, create_graph=True)
                hxy = [torch.abs(i - j).sum() + ((i - j) ** 2).sum() for i, j in zip(phix, yapp)]
                hpx, hpy = get_outer_gradients(hxy, xapp, yapp)
                hpx = cat_list_to_tensor(hpx).to(self.device)
                hpy = cat_list_to_tensor(hpy).to(self.device)
                x0 = cat_list_to_tensor(xapp).to(self.device)
                y0 = cat_list_to_tensor(yapp).to(self.device)
                hxy = cat_list_to_tensor(hxy).to(self.device)
                new_c = hxy.sum() - self.errta ** 2 - torch.matmul(hpx, x0) - torch.matmul(hpy, y0)

                self.con_c.append(new_c)
                self.con_hpx.append(hpx)
                self.con_hpy.append(hpy)
                self.lag_all.append(torch.tensor(self.lag_init).to(self.device))

            self.val_loss = val_loss_batch


            # delete inactive cutting planes
            for num in reversed(range(len(self.lag_all))):
                if abs(self.lag_all[num]) < 1e-7:
                    del self.lag_all[num]
                    del self.con_hpx[num]
                    del self.con_hpy[num]
                    del self.con_c[num]

            W = torch.Tensor(self.W)
            zero = torch.zeros_like(W)
            one = torch.ones_like(W)
            w_mask = torch.where(W > 0, one, zero)
            for i in range(self.N):
                grad_theta = torch.autograd.grad(o_loss, theta[i], retain_graph=True)
                for j in range(len(theta[i])):
                    if grad_theta[j] is None:
                        continue
                    theta[i][j] = theta[i][j] + w_mask[self.id][i] * self.theta_lr * grad_theta[j]
                self.new_theta[self.id][i] = [p.detach().clone() for p in theta[i]]

            self.outer_opt.step()
            self.outer_opt.zero_grad()
            self.new_x = [p.data.detach().clone() for p in self.meta_model.parameters()]

            temp_opt = torch.optim.Adam(params=meta_model_h.parameters(), lr=0.001, weight_decay=1e-4)
            temp_opt2 = torch.optim.Adam(params=task_model.parameters(), lr=0.001, weight_decay=1e-4)
            temp_opt.zero_grad()
            temp_opt2.zero_grad()
            temp_opt.step()
            temp_opt2.step()

        # for inactive agent
        else:
            for i in range(len(self.aver_x)):
                self.aver_x[i] = torch.zeros_like(self.aver_x[i]).to(self.device)
                for j in range(self.N):
                    self.aver_x[i] = self.aver_x[i] + self.W[self.id][j] * self.x[j][i]

            for (p, q) in zip(self.meta_model.parameters(), self.aver_x):
                p.data = q.data.detach().clone()

            self.new_x = [p.data.detach().clone() for p in self.meta_model.parameters()]

            for i in range(self.N):
                self.new_theta[self.id][i] = [p.detach().clone() for p in self.theta[self.id][i]]

    def update_all_msg(self, W, x, theta):
        self.W = copy.deepcopy(W)
        self.x = copy.deepcopy(x)
        self.theta = copy.deepcopy(theta)

    def send_msg_x(self):
        return [p.detach().clone() for p in self.new_x]

    def send_msg_theta(self):
        new_t = []
        for i in range(self.N):
            new_t.append(self.new_theta[self.id][i])
        return new_t




class Optimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(
                data_or_iter)
        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult - 1)])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class GD(Optimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GD, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)

        def gd_step(params, loss, step_size, create_graph=True):
            grads = torch.autograd.grad(loss, params, create_graph=create_graph)
            return [w - step_size * g for w, g in zip(params, grads)]

        return gd_step(params, loss, sz, create_graph=create_graph)

def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])