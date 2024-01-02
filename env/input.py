import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def PCR(y=None):
    G = nx.DiGraph()
    for i in range(8):
        G.add_node(i, htype='ds', type='ds' + str(i + 1), pos=(i + 1, 0), dur=20, finished=False)
    for i in range(4):
        G.add_node(i + 8, htype='conf', type='mix', pos=[0, 1], finished=False)
        G.add_edges_from([(2 * i, i + 8), (2 * i + 1, i + 8)])
    G.add_node(12, htype='conf', type='mix', pos=[0, 2], finished=False)
    G.add_node(13, htype='conf', type='mix', pos=[0, 2], finished=False)
    G.add_node(14, htype='conf', type='mix', pos=[0, 3], finished=False)
    G.add_edges_from([(8, 12), (9, 12), (10, 13), (11, 13), (12, 14), (13, 14)])
    G.name = 'PCR'
    G.porttyp = False
    return G

def in_vitro(inp=(2,3)):
    p, q = inp
    G = nx.DiGraph()
    pin = 0
    qjn = p * q
    m = 2 * p * q
    d = 3 * p * q
    dss = ['S1', 'S2', 'S3', 'S4']
    dsr = ['R1', 'R2', 'R3', 'R4']
    mix = ['M1', 'M2', 'M3', 'M4']
    det = ['D1', 'D2', 'D3', 'D4']
    for i in range(p):
        for j in range(q):
            G.add_node(pin, type=dss[i], htype='ds', dur=10)
            G.add_node(qjn, type=dsr[j], htype='ds', dur=10)
            G.add_node(m, htype='conf', type=mix[i])
            G.add_node(d, htype='nonc', type=det[j], dur=20)
            G.add_edges_from([(pin, m), (qjn, m), (m, d)])
            pin += 1
            qjn += 1
            m += 1
            d += 1
    G.porttyp = True
    return G

def create_dispense(width, length, n):
    arr = np.zeros((width, length))
    boundary = [(0, i) for i in range(length)] + [(i, 0) for i in range(1, width)] + [(width - 1, i) for i in
                                                                                      range(length)] + [(i, length - 1)
                                                                                                        for i in
                                                                                                        range(1,
                                                                                                              width - 1)]
    chosen = []
    while len(chosen) < n:
        pos = np.random.choice(len(boundary))
        if all(abs(boundary[pos][0] - c[0]) + abs(boundary[pos][1] - c[1]) > 2 or (
                abs(boundary[pos][0] - c[0]) != abs(boundary[pos][1] - c[1]) and abs(boundary[pos][0] - c[0]) + abs(
            boundary[pos][1] - c[1]) == 2) for c in chosen):
            chosen.append(boundary.pop(pos))
    for pos in chosen:
        arr[pos] = -1
    return arr, chosen


mix_p = [0.29, 0.58, 0.1, -0.5]


def check_op2start(op, chip, drops):
    return True


class DAG(nx.DiGraph):
    def __init__(self, G):
        super().__init__(G)
        self.dispensers = [node for node in G.nodes() if G.in_degree(node) == 0]
        self.copy = G
        self.noconfig = [node for node in G.nodes() if G.nodes[node]['htype'] == 'noc'] + self.dispensers
        self.initial_candidate_list()

    def assign_ports(self, ports, typ=False):
        if self.copy.porttyp:
           tports={}
           for node in self.dispensers:
               ty=self.nodes[node]['type']
               if ty not in tports:
                   tports[ty]=ports.pop()
               self.nodes[node]['pos'] = tports[ty]
        else:
            i = 0
            for node in self.dispensers:
                self.nodes[node]['pos'] = ports[i]
                i += 1
        self.out = ports[-1]
        # self._add_trans_nodes(ports[-1])

    def _add_trans_nodes(self, out):
        edges_to_remove = []
        for u, v in self.edges():
            new_node_id = self.number_of_nodes()
            self.add_node(new_node_id, htype='T1', type='trans', pos=list(u.pos))
            self.add_edge(u, new_node_id)
            self.add_edge(new_node_id, v)
            edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            self.remove_edge(u, v)
        for node in self.nodes():
            if self.out_degree(node) == 0:
                new_node_id = self.number_of_nodes()
                self.add_node(new_node_id, htype='T1', type='out', pos=out)
                self.add_edge(node, new_node_id)

    def initial_candidate_list(self):
        self.non_dis=self.copy.copy()
        self.non_dis.remove_nodes_from(self.dispensers)
        G = self.non_dis
        self.CL = [node for node in G.nodes() if G.in_degree(node) == 0]
        return

    def check_CL(self, droplets, chip):
        CL = self.CL
        G = self
        for i in CL[:]:  # [:]防止删一个跳一个
            op = G.nodes[i]
            if check_op2start(op, chip, droplets):
                CL.remove(i)
                # 混合or 稀释操作
                if op['htype'] == 'conf':
                    d = []
                    for father in G.predecessors(i):  # 两个父节点
                        fop = G.nodes[father]
                        # 非分配操作一定产生 store，父节点记录其完成后store的液滴
                        if 'store' in fop:
                            store = fop['store']
                            d.append(store.pos)
                            store.ref -= 1
                            if store.ref <= 0:
                                droplets.remove(store)
                        # 说明是分配操作
                        else:
                            d.append(list(fop['pos']))
                    droplets.addcp(d[0], d[1], opid=i)
                elif op['htype'] == 'nonc':
                    father = list(G.predecessors(i))[0]  # 只有一个父节点
                    fop = G.nodes[father]
                    # if 'store' in fop:
                    #     store = fop['store']
                    #     start = store.pos
                    #     store.ref -= 1
                    #     if store.ref <= 0:
                    #         droplets.remove(store)
                    # else:
                    #     start = fop['pos']
                    # des = op['pos']
                    # droplets.add(0, list(start), des, opid=i)
                    if 'store' in fop:
                        store = fop['store']
                        store.duration = op['dur']
                        store.opid = i

    def update_droplets(self, droplets, t, chip):
        G = self
        CL = self.CL
        ned_remove = []
        for d in droplets:
            # if d.type == 2:
            #     continue
            if d.finished:
                ned_remove.append(d)
                if d.opid != 0: # 输出和 store opid=0
                    op = G.nodes[d.opid]
                    if d.type == 0:
                        if op['htype'] == 'conf':
                            droplets.add(1, d.pos, opid=d.opid)
                            if d.partner:
                                droplets.remove(d.partner)
                        elif op['htype'] == 'nonc':
                            op['stop'] = t + op['dur']
                    if d.type == 1:
                        # 完成了就删除，G.non_dis只记录即将要执行的操作
                        G.non_dis.remove_node(d.opid)
                        # 叶子结点的话
                        if G.out_degree(d.opid) == 0:
                            droplets.add(0, d.pos, G.out)
                        else:
                            # 完成后先记录到store里
                            store = droplets.add(2, d.pos)
                            op['store'] = store
                            # 看看后继操作
                            for s in G.successors(d.opid):
                                store.ref += 1 # 有一个后继+1个需要用的
                                if G.non_dis.in_degree(s) == 0: # 所有前面都完成了 加到CL里
                                    CL.append(s)
                    elif d.type == 2:
                        droplets.add(0, d.pos, G.out)
        for d in ned_remove:
            droplets.remove(d)
        self.check_CL(droplets, chip)
