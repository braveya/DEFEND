from absl import app
from utils.mygraph import *
from utils.config_lib import set_flags
from utils.layers import SiameseNetwork
from utils.functions import *
from agent import agent
import random
from load_data import load_data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(aseed):
    num_worker, ft_size, nb_nodes, tr_dataiterator_list, ts_dataiterator_list =  load_data()

    x = [[] for i in range(num_worker)]
    theta = [[[] for i in range(num_worker)] for j in range(num_worker)]
    
    # the Siamese Network
    model0 = SiameseNetwork(ft_size, nb_nodes).to(device)
    
    for i in range(num_worker):
        for p in model0.parameters():
            x[i].append(p.data.detach().clone())
    for i in range(num_worker):
        for j in range(num_worker):
            for p in model0.parameters():
                theta[i][j].append(torch.zeros_like(p.data))
    
    # init 
    agents = []
    msg_net = ER_random(num_worker)
    for i in range(num_worker):
        w = agent()
        w.initia(i, device, ft_size, nb_nodes)
        w.update_all_msg(msg_net, x, theta)
        agents.append(w)


    # each agent becomes active at least once every T_asyn rounds
    act_list_iter = sample_with_constraints(list(range(num_worker)), FLAGS.T_asyn, FLAGS.epochs,
                                            int(num_worker*(1-FLAGS.stuggle_rate)))
    for epoch in range(FLAGS.epochs):
        act_id_list = act_list_iter[epoch]
        print(f'epoch={epoch}, active agent={act_id_list}')
        for j in range(num_worker):
            # If this agent is active
            if j in act_id_list:
                batch_data = tr_dataiterator_list[j].next()
                batch_data = data_a_batch(batch_data)
                # active agent update
                agents[j].loc_upd(batch_data, active=True)
            else:
                agents[j].loc_upd(None, active=False)

            # send message
            x[j] = agents[j].send_msg_x()
            theta[j] = agents[j].send_msg_theta()

        for j in range(num_worker):
            agents[j].update_all_msg(msg_net, x, theta)

        msg_net = ER_random(num_worker)



if __name__ == '__main__':
    # ==========================================================
    #                          参数
    # ==========================================================
    FLAGS = set_flags()
    app.run(main)