import networkx as nx

def ER_random(num_worker):
    p=0.5
    g = nx.random_graphs.erdos_renyi_graph(num_worker,p)
    e = [[] for i in range(num_worker)]
    msg_net=[[] for i in range(num_worker)]
    for p in g.edges():
        e[p[0]].append(p[1])
        e[p[1]].append(p[0])
    for i in range(num_worker):
        val = 1/(1+len(e[i]))
        k = 0
        for j in range(num_worker):
            if i==j:
                msg_net[i].append(val)
            elif (k<len(e[i]) and e[i][k]==j):
                msg_net[i].append(val)
                k = k+1
            else:
                msg_net[i].append(0)
    return msg_net
