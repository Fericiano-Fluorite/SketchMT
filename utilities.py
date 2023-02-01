import os
import math
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gromovWassersteinAveraging as gwa
import hashlib


def draw_H(H, path, coefficient=True):
    plt.figure(figsize=(15, 10))
    plt.matshow(H, fignum=False)
    if H.shape[1] / H.shape[0] >= 20:
        plt.yticks(range(0, H.shape[0], 2), range(0, H.shape[0], 2))
    if H.shape[1] >= 100:
        plt.xticks(range(0, H.shape[1], 10), range(0, H.shape[1], 10))
    # plt.colorbar(orientation="horizontal")
    if coefficient:
        plt.savefig(fname=os.path.join(path, "coefficient-matrix.jpg"))
    else:
        plt.savefig(fname=os.path.join(path, "sketch-error.jpg"))
    plt.close()


def tree_colors(T):
    n = T.number_of_nodes()
    if not hasattr(T, "root"):
        return ["r" for i in range(n)]

    dist_to_root = nx.single_source_dijkstra_path_length(T, T.root)
    colors = ["#00FF00" if dist_to_root[i] < 1e-6 else "r" for i in range(len(dist_to_root))]
    return colors


def djs_find(f, x):
    if f[x] != x:
        f[x] = djs_find(f, f[x])
    return f[x]


def coordinate_matrix(T, par=1):
    tmp = np.zeros((T.number_of_nodes(), par))
    for e, n in enumerate(T.nodes()):
        node = T.nodes[n]
        tmp[e][0] = node['height']
    return tmp


def is_tree(T):
    return nx.is_tree(T)


def has_near(l, x, thres=0.5):
    for i in l:
        if isinstance(i, str) and ((len(i) < 1) or i[0] == "<"):
            continue
        if math.fabs(float(i)-x)<thres:
            return True
    return False


# If we need to eliminate internal nodes to simplify the tree, we have to specify the root of the tree first
# Thus, when we pass parameters to eliminate_internal_nodes, it should be the original tree that T is sketched from
# Otherwise, determining the root manually can be a dangerous step
def tree_simplified(T: nx.Graph, p,
                    lambda_persistence=0.0,
                    # lambda_zero=1.0, lambda_limit=1.0,
                    # eliminate_internal_nodes=False, eliminate_fans=False,
                    T_o=None, retry=True):
    p = np.asarray(p).reshape(-1, )
    assert is_tree(T)
    num_nodes = T.number_of_nodes()
    if num_nodes <= 1:
        return T, [1] if p is not None else None

    if p is not None:
        assert num_nodes == len(p)

    f = {}
    for i in range(num_nodes):
        f[list(T.nodes())[i]] = list(T.nodes())[i]

    if T_o is not None:
        assert T_o in T.nodes()

    # limit = None
    # diam = -1
    #
    # dist_to_0 = nx.single_source_dijkstra_path_length(T, list(T.nodes())[0])
    # farthest_to_0 = None
    # for e in dist_to_0:
    #     if dist_to_0[e] > diam:
    #         diam = dist_to_0[e]
    #         farthest_to_0 = e
    # assert farthest_to_0 is not None
    # dist_to_farthest_to_0 = nx.single_source_dijkstra_path_length(T, farthest_to_0)
    # for e in dist_to_farthest_to_0:
    #     diam = max(diam, dist_to_farthest_to_0[e])
    # zero = diam / (num_nodes * num_nodes * lambda_zero)
    # limit = diam / (num_nodes * lambda_limit)
    # assert limit is not None

    removal_flag = False

    for u, vs in T.adjacency():
        for v, e in vs.items():
            if e['weight'] <= lambda_persistence and (len(list(T.neighbors(v))) <= 2 or len(list(T.neighbors(u))) <= 2):
                removal_flag = True
                fu = djs_find(f, u)
                fv = djs_find(f, v)
                if fu != fv:
                    if v > u:
                        f[fv] = fu
                    else:
                        f[fu] = fv

    for node in T.nodes:
        djs_find(f, node)

    s = {}
    cnt = 0
    f_order = list(f.values())
    f_order.sort()
    for i in f_order:
        if i not in s:
            s[i] = cnt
            cnt += 1

    T_sim = nx.Graph()
    if p is not None:
        for i in range(num_nodes):
            fa = f[list(T.nodes())[i]]
            if s[fa] not in T_sim.nodes():
                if "height" in T.nodes[fa]:
                    T_sim.add_node(s[fa], id=s[fa], prob=p[i], height=T.nodes[fa]["height"])
                else:
                    T_sim.add_node(s[fa], id=s[fa], prob=p[i])
            else:
                T_sim.nodes[s[fa]]["prob"] += p[i]

        for i in range(T_sim.number_of_nodes()):
            if i not in T_sim.nodes():
                print(s, f_order, f)

    for u, vs in T.adjacency():
        for v, e in vs.items():
            fu = djs_find(f, u)
            fv = djs_find(f, v)
            if fu != fv:
                T_sim.add_edge(s[fu], s[fv], weight=e['weight'])

    if T_o is not None:
        T_sim.root = s[djs_find(f, T_o)]
        assert T_sim.root in T_sim.nodes()

    if T_sim.number_of_nodes() <= 1:
        if T_sim.number_of_nodes() == 0:
            T_sim.add_node(T.nodes()[0])
        return T_sim, [1] if p is not None else None

    # if eliminate_internal_nodes:
    #     if not hasattr(T_sim, "root"):
    #         T_sim = get_root(T_sim)
    #     while True:
    #         found_internal_nodes = False
    #         for u, vs in T_sim.adjacency():
    #             if u == T_sim.root:
    #                 continue
    #             if len(vs.items()) == 2:
    #                 v0, e0 = list(vs.items())[0]
    #                 v1, e1 = list(vs.items())[1]
    #                 T_sim.add_edge(v0, v1, weight=e0["weight"] + e1["weight"])
    #                 if p is not None:
    #                     T_sim.nodes[v0]["prob"] += T_sim.nodes[u]["prob"] / 2
    #                     T_sim.nodes[v1]["prob"] += T_sim.nodes[u]["prob"] / 2
    #                 T_sim.remove_node(u)
    #                 found_internal_nodes = True
    #                 break
    #         if not found_internal_nodes:
    #             break
    #     assert is_tree(T_sim)

    def graph_labels(G):
        pp = {}
        for n in G.nodes():
            node = G.nodes[n]
            assert "prob" in node
            pp[n] = node["prob"]
        return list(pp.values())

    p_sim = None if p is None else graph_labels(T_sim)
    if p is not None:
        assert abs(sum(p_sim) - sum(p)) < 1e-10

    if removal_flag and retry:
        T_sim, p_sim = tree_simplified(T_sim, p_sim, lambda_persistence,
                                       T_o=T_sim.root if hasattr(T_sim, "root") else None, retry=False)

    return T_sim, p_sim


def get_dist2parent_distribution(T: nx.Graph, root: int, scalar_name: str):
    dist2parent = {}
    li = list()
    node = root
    dist2parent[node] = 0
    li.append(node)
    heights = []
    while len(li) > 0:
        node = li.pop(-1)
        heights.append(T.nodes[node][scalar_name])
        for u, vs in T.adjacency():
            if u != node:
                continue
            for v, e in vs.items():
                if v not in dist2parent:
                    dist2parent[v] = e['weight']
                    li.append(v)

    dists = np.asarray(list(dist2parent.values()), dtype=float)
    dist2parent[root] = np.max(heights) - np.min(heights)
    prob_weight = [dist2parent[key] for key in sorted(dist2parent.keys())]
    prob_weight /= np.sum(prob_weight)
    return np.asarray(prob_weight, dtype=float)


def get_distance_and_distribution(tree, distribution="uniform", weight_mode="shortestpath", **params):
    """
    Required field for the strategy choice in **params
    ---------------------------------------------------
    distribution="uniform": None

    distribution="ancestor": params["root"], int, the id of the root node in the tree
                             params["scalar_name"], str, the name for the scalar function of nodes in the tree

    weight_mode="shortestpath": params["edge_weight_name"], str, the name for the edge weight in the tree

    weight_mode="lca": params["root"], int, the id of the root node in the tree
                       params["scalar_name"], str, the name for the scalar function of nodes in the tree
    """
    num_of_nodes = tree.number_of_nodes()
    if distribution == "uniform":
        p = np.ones((num_of_nodes,)) / num_of_nodes
    elif distribution == "ancestor":
        assert "root" in params
        assert "scalar_name" in params
        p = get_dist2parent_distribution(tree, params["root"], params["scalar_name"])
    else:
        p = np.ones((num_of_nodes,)) / num_of_nodes

    if weight_mode == "shortestpath":
        assert "edge_weight_name" in params
        weight_str = params["edge_weight_name"]
    elif weight_mode == "lca":
        assert "scalar_name" in params
        assert "root" in params
        C = np.zeros((num_of_nodes, num_of_nodes))
        lca_matrix = lca(tree, params["root"])
        for node_a in tree.nodes:
            for node_b in tree.nodes:
                lca_node = lca_matrix[node_a, node_b]
                C[node_a][node_b] = tree.nodes[lca_node][params["scalar_name"]]
        return C, p
    else:
        assert "edge_weight_name" in params
        weight_str = params["edge_weight_name"]

    D = list(nx.all_pairs_dijkstra_path_length(tree, weight=weight_str))
    C = np.zeros((num_of_nodes, num_of_nodes))
    for ii in range(num_of_nodes):
        dist_zip = zip(*sorted(D[ii][1].items()))
        dist = list(dist_zip)
        C[ii, :] = dist[1]
    return C, p


def lca(T, root):
    num = T.number_of_nodes()
    lca_mat = np.zeros((num, num), dtype=int) - 1
    ff = {}
    col = {}
    ancestor = {}
    for node in T.nodes():
        ff[node] = node
        col[node] = False
        ancestor[node] = node

    TarjanOLCA(T, root, None, ff, col, ancestor, lca_mat)
    return lca_mat


def TarjanOLCA(T, u, parent, ff, col, ancestor, lca_mat):
    for neighbor in T.neighbors(u):
        if parent is not None and neighbor == parent:
            continue
        TarjanOLCA(T, neighbor, u, ff, col, ancestor, lca_mat)
        fu = djs_find(ff, u)
        fv = djs_find(ff, neighbor)
        if fu != fv:
            ff[fv] = fu
        fu = djs_find(ff, u)
        ancestor[fu] = u

    col[u] = True
    for node in T.nodes():
        if col[node] and lca_mat[u, node] < 0:
            fv = djs_find(ff, node)
            lca_mat[u, node] = lca_mat[node, u] = ancestor[fv]


def graph_labels(G, pos_mat=None):
    colors = {}
    pos = {}
    for n in G.nodes():
        node = G.nodes[n]
        if 'color' in node:
            colors[n] = node['color']
        else:
            colors[n] = "r"
        if pos_mat is None:
            pos[n] = (node['x'], node['y'])
        else:
            pos[n] = (pos_mat[n][0], pos_mat[n][1])

    return colors.values(), pos


def get_root(T, T_ref=None, root_ref=None, C_T=None, p_T=None, C_ref=None, p_ref=None):
    if hasattr(T, "root"):
        return T

    if T_ref is None:
        C, _ = get_distance_and_distribution(T.number_of_nodes(), T)
        T.root = list(T.nodes())[np.argmin(np.sum(C, axis=1))]
        return T
    else:
        if root_ref is None:
            raise ValueError

        if C_T is None or p_T is None or C_ref is None or p_ref is None:
            return get_root(T)

        opt_coup, _ = gwa.gromov_wasserstein_asym(C_T, C_ref, p_T, p_ref)

        T.root = list(T.nodes())[np.argmax(opt_coup[:, root_ref])]
        if not hasattr(T, "root"):
            print ("Failed to find a corresponding root from the reference. Return the balanced root instead")
            return get_root(T)
        return T


def is_sorted(l):
    if len(l) <= 1:
        return True
    for i in range(len(l)-1):
        if l[i] >= l[i+1]:
            return False
    return True


def mst(mat, T_=None, verbose=0):
    N, M = mat.shape
    assert N == M

    edges = [[mat[i, j], i, j] for i in range(N) for j in range(M) if i < j]
    unused_edges = []
    f = []
    for i in range(N):
        f.append(i)

    edges.sort()
    if T_ is not None:
        T = T_
    else:
        T = nx.Graph()
        for i in range(N):
            T.add_node(i)

    for w, x, y in edges:
        fx = djs_find(f, x)
        fy = djs_find(f, y)
        if fx != fy:
            f[fx] = fy
            T.add_edge(x, y, weight=w)
        else:
            unused_edges.append((w, x, y))

    if verbose == 0:
        return T
    else:
        return T, unused_edges


def get_trees_from_matrices(vecs, p_vecs,
                            num_vecs, vec_size,
                            lambda_persistence=0.0,
                            # lambda_zero=1.0, lambda_limit=1.0,
                            weight_mode="shortestpath", ref=None,
                            # eliminate_internal_nodes=False, eliminate_fans=False,
                            no_simp=False):
    Ts = []
    ps = []
    T_nosimps = []
    mats = []
    for ii in range(num_vecs):
        vec = vecs[:, ii]
        p_vec = None if p_vecs is None else p_vecs[ii]
        if p_vec is not None:
            assert len(p_vec) == vec_size
        mat = np.zeros((vec_size, vec_size)) + sys.float_info.max
        tmp = 0
        vec[vec < 0] = 0
        for x in range(vec_size):
            for y in range(vec_size):
                if x <= y:
                    if abs(vec[tmp]) < 1e-12:
                        vec[tmp] = 0
                    mat[x][y] = vec[tmp]
                    tmp += 1
                else:
                    mat[x][y] = mat[y][x]

        mats.append(mat)

        if weight_mode == "shortestpath":
            T = mst(mat)
        else:
            assert weight_mode == "lca"
            T, _ = ultra2tree(mat, average_fix=True)

        if ref is not None:
            T.root = ref[ii]

        if no_simp:
            T_nosimps.append(copy.deepcopy(T))
            ps.append(copy.deepcopy(p_vec))
        else:
            if ref is not None:
                T, p = tree_simplified(T, p_vec, lambda_persistence,
                                       # lambda_zero=lambda_zero, lambda_limit=lambda_limit,
                                       # eliminate_internal_nodes=eliminate_internal_nodes,
                                       # eliminate_fans=eliminate_fans,
                                       T_o=ref[ii])
            else:
                T, p = tree_simplified(T, p_vec, lambda_persistence)
                                       # lambda_zero=lambda_zero, lambda_limit=lambda_limit)
            Ts.append(T)
            ps.append(np.asarray(p))

    if not no_simp:
        return Ts, mats, ps
    else:
        return T_nosimps, mats, ps


def draw_vecs(vecs, path=None, filenames=[], log_scale=False):
    num_vecs = len(vecs)
    length_color_map = 15
    if path is None:
        return
    wid = 0.9
    assert num_vecs == len(filenames)
    for e, vec in enumerate(vecs):
        fig = plt.figure(figsize=(20, 8))
        fig.add_axes([0.05, 0.05, 0.9, 0.8])
        plt.matshow(vec, fignum=False)
        if vec.shape[0] == 1:
            plt.yticks([])
        fig.add_axes([0.05, 0.9, 0.9, 0.05])
        mat_min = max(min([np.min(each) for each in vec]), 0)
        mat_max = max([np.max(each) for each in vec])
        interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
        step = 2
        while (interval < 1e-4) and (step < 4) :
            step += 1
            interval = round((mat_max - mat_min) / (length_color_map - 1), step)
        if log_scale:
            interval = (mat_max - mat_min) / (length_color_map - 1)
        color_map_mat = np.matrix([i * interval + mat_min for i in range(length_color_map)], dtype=float).reshape(
            (1, length_color_map))
        color_map_list = []
        if log_scale:
            color_map_list = [str(round(np.exp(i) - 1, step)) if ee % 2 == 0 else "" for ee, i in
                              enumerate(color_map_mat.tolist()[0])]
            # color_map_list = ["e^" + str(round(i, step)) + "-1" if ee % 2 == 0 else "" for ee, i in
            #                   enumerate(color_map_mat.tolist()[0])]
        else:
            color_map_list = [str(round(i, step)) if ee%2==0 else "" for ee, i in enumerate(color_map_mat.tolist()[0])]
        plt.matshow(color_map_mat, fignum=False)
        plt.yticks([])
        plt.xticks(np.arange(0, length_color_map), color_map_list)
        plt.savefig(os.path.join(path, filenames[e]))
        plt.close()


def mkdir(path, sysstr):
    if not os.path.exists(path):
        if sysstr == "Windows":
            path = path.replace("/", "\\")
            os.system("md " + path)
        else:
            path = path.replace("\\", "/")
            os.system("mkdir " + path)
    return path


def hashcode(args: list):
    for e in range(len(args)):
        args[e] = str(args[e]).lower().replace("/", "_")
    hash_string = "-".join(args)
    return hash_string


# recover the tree structure from an ultra matrix
# Strategy: convert the ultra matrix into a pairwise distance matrix, and run MST
def ultra2tree(C, g=None, average_fix=False, verbose=False):
    N = C.shape[0]
    assert C.shape[1] == N
    if g is not None:
        assert g.shape[0] == N

    dgn = np.diag(C)
    T = nx.Graph()
    root = np.min(dgn)
    T.root = root
    for i in range(N):
        if g is None:
            T.add_node(i, height=C[i][i])
        else:
            raise NotImplementedError

    # Two ideas:
    # 1. using the original "ultra" matrix C (usually illegal after sketching)
    # 2. using a fixed ultra matrix (symmetrical by the main diagonal)
    Cm = np.zeros(C.shape)
    for i in range(N):
        for j in range(N):
            Cm[i, j] = Cm[j, i] = (C[i][j] + C[j][i]) / 2

    # Changing the "ultra" matrix into a pairwise dist matrix
    # assuming lca(a,b) == c
    # distance(a,b) = | f(a) - f(c) | + | f(b) - f(c) |
    # f(a) = dgn(a), f(b) = dgn(b), f(c) = C[a,b]

    D = np.zeros(C.shape)
    if average_fix:
        for i in range(N):
            for j in range(N):
                if i >= j:
                    continue
                D[i, j] = D[j, i] = abs(dgn[i] - Cm[i, j]) + abs(dgn[j] - Cm[i, j])
    else:
        for i in range(N):
            for j in range(N):
                D[i, j] = abs(dgn[i] - C[i, j]) + abs(dgn[j] - C[i, j])

    return mst(D, T, verbose), root






