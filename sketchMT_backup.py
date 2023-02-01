import matplotlib.pyplot as plt
import networkx as nx

from weighted_hierarchy_pos import weighted_hierarchy_pos
from sklearn.decomposition import NMF
import readMergeTree as rmt
import column_selection as csel
import platform
from utilities import *

# ------------------------- Hyperparameters ------------------------------- #
# Path of input dataset (which includes .txt data files)
# dataset_path = "more_data/redSeaNew"
# dataset_name = "redSeaNew"
# dataset_path = "more_data/CornerFlow/MT_TXTFormat"
# dataset_name = "CornerFlow"
# dataset_path = "more_data/boussinessq2D/step1/"
# dataset_name = "heatedCylinder_Ycomponent"
# dataset_path = "more_data/Gaussian_rotated/"
# dataset_name = "Gaussian_rotated"
# dataset_path = "more_data/pania/MT-Temperature/"
# dataset_name = "material_science"
dataset_path = "more_data/Jacob_MT/"
dataset_name = "Jacob_MT"

# Output prefix, generally the root directory you want your output to be placed. For example, "./TestResult1/"
# We assume that the prefix ends with "/"
output_prefix = "./05-04-2021/"
if output_prefix[-1] != "/":
    output_prefix += "/"

# The maximum size of blow-up graph in GW framework, usually 2 or 3 times of the size of the largest input tree.
# We only modify the multiplier
# If we do not set the maximum size, the blow-up graph can be exponentially large depending on the input data
lambda_budget = 3

lambda_zero = 2.0
lambda_limit = 2.0

# 0.5, 2.0 for CornerFlow
# 2.0, 2.0 for redSeaNew
# 2.0, 1.0 for heatedFlow
# 1024.0, 8.0 for Gaussian_rotated

iteration = 50
# Use sampling_step > 1 if you do not want consecutive trees from the input to be used
sampling_step = 1

# "mst" for MST or "iter" for LSST
spanning_tree_mode = "mst"

# "IFS" for CSS-IFS, "LSS" for CSS-LSS, "NMF" for NMF
sketching_mode = "NMF"


# switches to include what kinds of outputs
output_mapping = True
output_loss_log = True
output_W = False
output_H = True
output_basis = True
output_main_results = True
output_GW_dist = True
output_selected_trees = (sketching_mode in {"LSS", "IFS"}) and output_basis

# random seed
seed = 31

# We provide results with following numbers of basis trees
# num_basis_trees_list = [10, 15, 20, 30]
# num_basis_trees_list = [15, 30]
num_basis_trees_list = [3, 4, 5, 7, 10]
# # num_basis_trees_list = range(2, 3)
# num_basis_trees_list = range(5, 31, 5)

retest = False
coupling_reordering = False

# ------------------------- Hyperparameters Complete ------------------------------- #

sysstr = platform.system()

CList = []
pList = []
trees = []
for root, _, files in os.walk(dataset_path):

    def key(s):
        try:
            int(s)
            return int(s)
        except ValueError:
            return len(files) + 1

    files.sort(key=lambda x: key(x.split(".")[0].split("_")[-1]))
    for file in files:
        trees.extend(rmt.get_trees(os.path.join(root, file)))
trees = [i for e, i in enumerate(trees[:]) if e % sampling_step == 0]

# _, pos = graph_labels(trees[0])
# nx.draw_networkx(trees[0], pos=pos)
# plt.show()
# exit()

# nx.draw_networkx(trees[0], pos=weighted_hierarchy_pos(trees[0], trees[0].root))
# plt.show()

num_trees = len(trees)
trees_reselect_root = []
print "Num_trees = " + str(num_trees)
largest_tree_idx = -1
tree_max_size = -1

for e, T in enumerate(trees):
    if T.number_of_nodes() > tree_max_size:
        tree_max_size = T.number_of_nodes()
        largest_tree_idx = e
    T_reselect_root = copy.deepcopy(T)
    delattr(T_reselect_root, 'original')
    for node in T_reselect_root.nodes():
        if 'height' in T_reselect_root.nodes[node]:
            T_reselect_root.nodes[node].pop('height')
    trees_reselect_root.append(T_reselect_root)
    C, p = get_distance_and_distribution(T.number_of_nodes(), T)
    CList.append(C)
    pList.append(p)

for T in trees_reselect_root:
    T = get_root(T)

# Compute C, p for average graph using GW framework
budget = np.max([T.number_of_nodes() for T in trees]) * lambda_budget
print "budget = " + str(budget)

CnBase_path = 'tmp/CnBase_' + dataset_path.replace("/", "_") + '_' + str(budget) + '_' + str(sampling_step) + '.txt'
pnBase_path = 'tmp/pnBase_' + dataset_path.replace("/", "_") + '_' + str(budget) + '_' + str(sampling_step) + '.txt'
Frechet_path = 'tmp/Frechet_' + dataset_path.replace("/", "_") + '_' + str(budget) + '_' + str(sampling_step) + '.txt'
if retest or not os.path.exists(CnBase_path):
    base_index = 1
    CList_toCompare = copy.deepcopy(CList[:base_index])
    CList_toCompare.extend(CList[base_index + 1:])
    pList_toCompare = copy.deepcopy(pList[:base_index])
    pList_toCompare.extend(pList[base_index + 1:])
    CnBase, pnBase, Frechet_Loss = gwa.network_karcher_mean_armijo_sched_compress(
        CList[base_index], pList[base_index], CList_toCompare, pList_toCompare, budget, iteration)
    # CnBase, pnBase, Frechet_Loss = gwa.network_karcher_mean_armijo_sched(
    #     CList[base_index], pList[base_index], CList_toCompare, pList_toCompare, iteration)
    np.savetxt(CnBase_path, CnBase, delimiter=",")
    np.savetxt(pnBase_path, pnBase, delimiter=",")
    np.savetxt(Frechet_path, Frechet_Loss, delimiter=",")

else:
    CnBase = np.loadtxt(CnBase_path, delimiter=",")
    pnBase = np.loadtxt(pnBase_path, delimiter=",")
    Frechet_Loss = np.loadtxt(Frechet_path, delimiter=",")

opt_couplings = [[] for i in range(num_trees)]
C_blowups = [[] for i in range(num_trees)]
p_blowups = [[] for i in range(num_trees)]
meanC_blowups = [[] for i in range(num_trees)]
meanp_blowups = [[] for i in range(num_trees)]
root_blowups = [[] for i in range(num_trees)]
tangent_vecs = [[] for i in range(num_trees)]

# X0, X1, v.shape = (num_lsst_nodes, num_lsst_nodes), to be flattened

N_blowup = -1
for i in range(num_trees):
    CnBase_copy = copy.deepcopy(CnBase)
    pnBase_copy = copy.deepcopy(pnBase)
    X0, X1, p0, p1, v, oc, log = gwa.log_map_ignore(CnBase_copy, CList[i], pnBase_copy, pList[i],
                                                    G0=(None if coupling_reordering or (i<1) or (len(pList[i]) != len(pList[i-1]))
                                                        else opt_couplings[i-1]))
    root_blowups[i] = np.argmax(oc[:, trees[i].root])

    opt_couplings[i] = oc
    # if (i > 3) and (i < 6):
    #     print root_blowups[i]
    #     print trees[i].root, oc[:, trees[i].root]
    #     np.savetxt(str(i)+".txt", oc, fmt="%.3f", delimiter=",")
    #     np.savetxt("X1_" + str(i) + ".txt", X1, fmt="%.3f", delimiter=",")
    #     np.savetxt("CList_" + str(i) + ".txt", CList[i], fmt="%.3f", delimiter=",")
    #     print "log = ", log['gw_dist']
    # print X0.shape, X1.shape, oc.shape
    C_blowups[i] = X1 if not coupling_reordering else rearrange_mat(X1, oc)
    assert X1.shape[0] == X1.shape[1]
    if N_blowup == -1:
        N_blowup = X1.shape[0]
    else:
        assert N_blowup == X1.shape[0]

    p_blowups[i] = p1
    meanC_blowups[i] = X0
    meanp_blowups[i] = p0
    tangent_vecs[i] = v

# exit()
# We only get the upper triangular part of each matrix and flatten it into a vector
trees_vecs = np.array([[C_blowups[i][x][y] for x in range(N_blowup) for y in range(N_blowup) if x <= y] for i in range(num_trees)], dtype=float)
trees_vecs = np.matrix(trees_vecs).transpose()

mapping_matrix = trees_vecs
T_mappings, mats_mapping, T_mappings_nosimp = get_trees_from_matrices(mapping_matrix, num_trees, N_blowup, "Mapping",
                                                                      lambda_zero=lambda_zero,
                                                                      lambda_limit=lambda_limit,
                                                                      mode=spanning_tree_mode,
                                                                      eliminate_internal_nodes=True,
                                                                      eliminate_fans=True, ref=root_blowups, no_simp=True)

# for t in range(num_trees):
#     T_mapping = T_mappings[t]
#     C_mapping, p_mapping = get_distance_and_distribution(T_mapping.number_of_nodes(), T_mapping)
#     opt_coup, _ = gwa.gromov_wasserstein_asym(CList[t], C_mapping, pList[t], p_mapping)
#     T_mapping = get_root(T_mapping)  # get_root(T_mapping, trees[t], opt_coup)
#     if not hasattr(T_mapping, 'root'):
#         T_mapping.root = T_mapping.nodes.keys()[np.argmax(opt_coup[trees[t].root])]
#     T_mappings.append(T_mapping)

# Matrix decomposed by NMF
# Find the num_basis_trees that produce minimum GW_dist(Ti, ^Ti)
# ^Ti: approximated tree from W0*H0
num_basis_trees = -1
best_NMF_loss = float('inf')
W = None
H = None
H_showMin = -5
H_showMax = 5
output_file_mode = "LSST" if spanning_tree_mode == "iter" else "MST"

subfolder = output_prefix + dataset_name + "/" + output_file_mode + "/" + sketching_mode
if sysstr == "Windows":
    output_prefix_backslash = output_prefix.replace("/", "\\")
    subfolder_backslash = subfolder.replace("/", "\\")
    os.system("md " + output_prefix_backslash + dataset_name)
    os.system("md " + output_prefix_backslash + dataset_name + "\\" + output_file_mode)
    os.system("md " + subfolder_backslash)
else:
    os.system("mkdir " + output_prefix + dataset_name)
    os.system("mkdir " + output_prefix + dataset_name + "/" + output_file_mode)
    os.system("mkdir " + subfolder)

f_gwloss_log = open(subfolder + "/gwloss.txt", "w")
f_solveloss_log = open(subfolder + "/solveloss.txt", "w")


for e, attempt_num_basis_trees in enumerate(num_basis_trees_list):
    W0 = None
    H0 = None
    selected_index = None
    solve_loss = None
    # NMF
    if sketching_mode == "NMF":
        NMF_model = NMF(n_components=attempt_num_basis_trees, init='nndsvda', random_state=seed)
        W0 = NMF_model.fit_transform(trees_vecs)
        H0 = NMF_model.components_
    # Column Selection strat
    elif (sketching_mode == "IFS") or (sketching_mode == "LSS"):
        W0, selected_index = csel.column_selection(trees_vecs, num_trees, attempt_num_basis_trees, mode=sketching_mode, seed=seed)
        H0, solve_loss, _, _ = np.linalg.lstsq(W0, trees_vecs, rcond=None)
    else:
        raise NotImplementedError

    if W0.shape[1] < attempt_num_basis_trees:
        print "Too many basis trees, column selection may fail."
        break
    approx_trees_vecs = W0.dot(H0)
    assert (approx_trees_vecs.shape == trees_vecs.shape)
    NMF_loss = 0

    print "Decomposition Done"

    # We generate ^Ti by generating LSST from the vectors in W0*H0
    T_approx_trees, mats_approx, T_approx_nosimp = get_trees_from_matrices(approx_trees_vecs, num_trees, N_blowup,
                                                                           "Approx",
                                                                           lambda_zero=lambda_zero,
                                                                           lambda_limit=lambda_limit,
                                                                           mode=spanning_tree_mode,
                                                                           eliminate_internal_nodes=True,
                                                                           eliminate_fans=True, ref=root_blowups, no_simp=True)

    T_basis_trees, mats_basis = get_trees_from_matrices(W0, attempt_num_basis_trees, N_blowup, "Basis",
                                                        lambda_zero=lambda_zero,
                                                        lambda_limit=lambda_limit,
                                                        mode=spanning_tree_mode,
                                                        eliminate_internal_nodes=True, eliminate_fans=True,
                                                        ref=None)

    for t in range(num_trees):
        T_approx_trees[t] = get_root(T_approx_trees[t])
        T_mappings[t] = get_root(T_mappings[t])

    for t in range(attempt_num_basis_trees):
        T_basis_trees[t] = get_root(T_basis_trees[t])

    if sysstr == "Windows":
        output_prefix_backslash = output_prefix.replace("/", "\\")
        subfolder_backslash = subfolder.replace("/", "\\")
        os.system("md " + subfolder_backslash + "\\numBasis_" + str(attempt_num_basis_trees))
        os.system("md " + subfolder_backslash + "\\numBasis_" + str(attempt_num_basis_trees) + "\\maps")
        os.system("md " + subfolder_backslash + "\\numBasis_" + str(attempt_num_basis_trees) + "\\basis")
        if output_selected_trees:
            os.system("md " + subfolder_backslash + "\\numBasis_" + str(attempt_num_basis_trees) + "\\selected_trees")
    else:
        os.system("mkdir " + subfolder + "/numBasis_" + str(attempt_num_basis_trees))
        os.system("mkdir " + subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/maps")
        os.system("mkdir " + subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/basis")
        if output_selected_trees:
            os.system("mkdir " + subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/selected_trees")

    if output_W:
        print "Outputting basis vectors..."
        draw_vecs([trees_vecs, approx_trees_vecs, W0],
                  path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/basis/",
                  filenames=["A.jpg", "A_head.jpg", "W.jpg"])
        print "Outputting basis vectors done"

    if output_loss_log:
        print "Outputting loss..."
        if selected_index is not None:
            np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/selected_index.txt",
                       selected_index, fmt="%d")
            np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/solve_loss.txt",
                       solve_loss, fmt="%.2f", delimiter="\n")
            np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/total_solve_loss.txt",
                       np.array([np.sum(solve_loss)]), fmt="%.2f", delimiter="\n")
            draw_vecs([solve_loss], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/", filenames=["solve_loss.jpg"])
            log_solve_loss = np.log(1 + solve_loss)
            draw_vecs([log_solve_loss], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/",
                      filenames=["solve_loss_log.jpg"], log_scale=True)
            print >> f_solveloss_log, "Num_basis=" + str(attempt_num_basis_trees), ", loss=" + str(np.sum(solve_loss))
        else:
            diff = trees_vecs - approx_trees_vecs
            norm = np.linalg.norm(diff, ord=None, axis=0)
            norm = np.multiply(norm, norm)
            print trees_vecs.shape, norm.shape
            np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/solve_loss.txt",
                       norm, fmt="%.2f", delimiter="\n")
            np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/total_solve_loss.txt",
                       np.array([np.sum(norm)]), fmt="%.2f", delimiter="\n")
            draw_vecs([np.matrix(norm)], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/",
                      filenames=["solve_loss.jpg"])
            log_norm = np.log(1 + norm)
            draw_vecs([np.matrix(log_norm)], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/",
                      filenames=["solve_loss_log.jpg"], log_scale=True)
            print >> f_solveloss_log, "Num_basis=" + str(attempt_num_basis_trees), ", loss=" + str(np.sum(norm))
        print "Outputting loss done"

    if output_H:
        print "Outputting weight matrix..."
        plt.figure(figsize=(15, 10))
        H0_toShow = copy.deepcopy(H0)

        H0_toShow = np.where(H0_toShow < H_showMin, H_showMin, H0_toShow)
        H0_toShow = np.where(H0_toShow > H_showMax, H_showMax, H0_toShow)
        H0_min = np.min(H0_toShow)
        H0_max = np.max(H0_toShow)
        H0_length_color_map = 15
        H0_interval = round((H0_max - H0_min) / (H0_length_color_map - 1), 2)
        H0_range = [H0_min + H0_interval * i for i in range(H0_length_color_map)]
        H0_range.sort()
        H0_labels = [str(round(i, 2)) if ee%2==0 else "" for ee, i in enumerate(H0_range)]
        if has_near(H0_labels, H_showMin, 0.005 * H0_length_color_map):
            H0_labels[0] = "<="+H0_labels[0]
        if has_near(H0_labels, H_showMax, 0.005 * H0_length_color_map):
            H0_labels[-1] = ">="+H0_labels[-1]
        plt.axes([0.1, 0.05, 0.8, 0.8])
        plt.matshow(H0_toShow, fignum=False)
        plt.axes([0.3, 0.9, 0.4, 0.05])
        plt.matshow(np.matrix(H0_range, dtype=float), fignum=False)
        plt.xticks(np.arange(0, len(H0_range)), H0_labels)
        plt.yticks([], [])
        plt.savefig(fname=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/basis/Y.jpg")
        plt.close()

        np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/basis/Y.txt", H0, fmt="%.2f", delimiter=",")
        print "Outputting weight matrix done"

    if output_basis:
        print "Outputting basis trees..."
        if output_selected_trees:
            os.system("rm " + subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/selected_trees/selected_idx=*.jpg")
            for selected_id in selected_index:
                plt.figure(figsize=(8, 10))
                # plt.axis([0.05, 1.05, 995.7, 1000.3])
                nx.draw_networkx(trees[selected_id], pos=weighted_hierarchy_pos(trees[selected_id], root=trees[selected_id].root))
                plt.suptitle("Tree " + str(selected_id).zfill(2))
                plt.savefig(fname=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/selected_trees/selected_idx=" + str(selected_id).zfill(2) + ".jpg")
                plt.close()

        for i in range(attempt_num_basis_trees):

            fig = plt.figure(figsize=(7, 11))
            # plt.subplot(131)
            # plt.matshow(mats_basis[i], fignum=False)
            # plt.title("Map_basis[" + str(i) + "] (size=" + str(len(mats_basis[i])) + ")")
            # plt.subplot(132)
            # if attempt_num_basis_trees == 3:
            #     T_basis_trees[0].root = 13
            #     T_basis_trees[1].root = 19
            #     T_basis_trees[2].root = 16
            for cp in range(attempt_num_basis_trees):
                while True:
                    found_internal_nodes = False
                    for u, vs in T_basis_trees[cp].adjacency():
                        if u == T_basis_trees[cp].root:
                            continue
                        if len(vs.items()) == 2:
                            v0, e0 = vs.items()[0]
                            v1, e1 = vs.items()[1]
                            T_basis_trees[cp].add_edge(v0, v1, weight=e0['weight'] + e1['weight'])
                            T_basis_trees[cp].remove_node(u)
                            found_internal_nodes = True
                            break
                    if not found_internal_nodes:
                        break
            nx.draw_networkx(T_basis_trees[i], pos=weighted_hierarchy_pos(T_basis_trees[i], root=T_basis_trees[i].root))
            plt.title("T_basis[" + str(i) + "] (root=" + str(T_basis_trees[i].root) + ")")
            # plt.subplot(133)
            C_basis, _ = get_distance_and_distribution(T_basis_trees[i].number_of_nodes(), T_basis_trees[i])
            # plt.matshow(C_basis, fignum=False)
            # plt.title("Map_T_basis[" + str(i) + "] (size=" + str(len(C_basis)) + ")")

            # length_color_map = 15
            # fig.add_axes([0.13, 0.20, 0.22, 0.05])
            # mat_min = np.min(mats_basis[i])
            # mat_max = np.max(mats_basis[i])
            # interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            # color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
            #                           dtype=float).reshape((1, length_color_map))
            # color_map_list = [str(round(r, 2)) if e%2==0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            # plt.matshow(color_map_mat, fignum=False)
            # plt.yticks([])
            # plt.xticks(np.arange(0, length_color_map), color_map_list)
            #
            # fig.add_axes([0.68, 0.20, 0.22, 0.05])
            # mat_min = np.min(mats_basis[i])
            # mat_max = np.max(mats_basis[i])
            # interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            # color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
            #                           dtype=float).reshape((1, length_color_map))
            # color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            # plt.matshow(color_map_mat, fignum=False)
            # plt.yticks([])
            # plt.xticks(np.arange(0, length_color_map), color_map_list)

            plt.savefig(fname=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/basis/basis_" + str(i).zfill(2) + ".jpg")
            plt.close()

        print "Outputting basis trees done"

    if output_main_results:

        print "Outputting main results..."
        for i in range(num_trees):
            pos_approx = weighted_hierarchy_pos(T_approx_trees[i], root=T_approx_trees[i].root)
            pos_approx_min = min(pos_approx.values(), key=lambda x: x[1])[1]

            y_axis_boundary = min(pos_approx_min, -trees[i].height_difference)
            plt.figure(figsize=(40, 20))
            plt.subplot(221)
            _, pos = graph_labels(trees[i])
            nx.draw_networkx(trees[i], pos=pos)
            plt.title("T_original (xy-layout)")

            plt.subplot(223)
            nx.draw_networkx(trees[i], pos=weighted_hierarchy_pos(trees[i], root=trees[i].root))
            plt.title("T_original (root=" + str(trees[i].root) + ")")

            # plt.subplot(142)
            # plt.axis([-0.05, 1.05, y_axis_boundary*1.2, trees[i].height_difference*0.05])
            # colors = tree_colors(T_mappings_nosimp[i])
            # nx.draw_networkx(T_mappings_nosimp[i],
            #                  pos=weighted_hierarchy_pos(T_mappings_nosimp[i], root=T_mappings_nosimp[i].root), node_color=colors)
            # plt.title("T_balance (root=" + str(T_mappings_nosimp[i].root) + ")")
            #
            # plt.subplot(143)
            # plt.axis([-0.05, 1.05, y_axis_boundary * 1.2, trees[i].height_difference * 0.05])
            # T_approx_nosimp[i] = get_root(T_approx_nosimp[i], True)
            # colors = tree_colors(T_approx_nosimp[i])
            # nx.draw_networkx(T_approx_nosimp[i], pos=weighted_hierarchy_pos(T_approx_nosimp[i], T_approx_nosimp[i].root), node_color=colors)
            # plt.title("T_sketched_no_compression (root=" + str(T_approx_nosimp[i].root) + ")")

            # plt.subplot(154)
            # plt.axis([-0.05, 1.05, y_axis_boundary*1.2, trees[i].height_difference*0.05])
            # nx.draw_networkx(T_approx_trees[i], pos=pos_approx)
            # plt.title("T_sketched (root=" + str(T_approx_trees[i].root) + ")")

            plt.subplot(224)
            # plt.axis([-0.05, 1.05, y_axis_boundary*1.2, trees[i].height_difference * 0.05])
            T_approx_trees[i] = get_root(T_approx_trees[i], False, trees[i])
            nx.draw_networkx(T_approx_trees[i], pos=weighted_hierarchy_pos(T_approx_trees[i], root=T_approx_trees[i].root))
            plt.title("T_sketched (root=" + str(T_approx_trees[i].root) + ")")

            plt.suptitle("Tree " + str(i))
            plt.savefig(fname=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/output_numBasis=" + str(
                attempt_num_basis_trees) + "_tree_" + str(i).zfill(2) + ".jpg")
            plt.close()

            C_approx, p_approx = get_distance_and_distribution(T_approx_trees[i].number_of_nodes(), T_approx_trees[i])
            opt_coup_approx, _ = gwa.gromov_wasserstein_asym(C_approx, CList[i], p_approx, pList[i])
            C_rearrange_approx = rearrange_mat(C_approx, opt_coup_approx)

            C_approx_nosimp, p_approx_nosimp = get_distance_and_distribution(T_approx_nosimp[i].number_of_nodes(), T_approx_nosimp[i])
            fig = plt.figure(figsize=(15, 15))
            length_color_map = 15
            fig.add_axes([0.04, 0.63, 0.28, 0.28])
            plt.matshow(CList[i],fignum=False)
            plt.title("Map_original")
            fig.add_axes([0.04, 0.54, 0.28, 0.04])
            mat_min = np.min(CList[i])
            mat_max = np.max(CList[i])
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            fig.add_axes([0.36, 0.63, 0.28, 0.28])
            plt.matshow(C_blowups[i],fignum=False)
            plt.title("Map_blowup")
            fig.add_axes([0.36, 0.54, 0.28, 0.04])
            mat_min = np.min(C_blowups[i])
            mat_max = np.max(C_blowups[i])
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            fig.add_axes([0.68, 0.63, 0.28, 0.28])
            plt.matshow(mats_approx[i], fignum=False)
            plt.title("Map_approx")
            fig.add_axes([0.68, 0.54, 0.28, 0.04])
            mat_min = np.min(mats_approx[i])
            mat_max = np.max(mats_approx[i])
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            fig.add_axes([0.04, 0.13, 0.28, 0.28])
            plt.matshow(C_approx_nosimp, fignum=False)
            plt.title("Map_approx_" + output_file_mode + " (no compression)")
            fig.add_axes([0.04, 0.04, 0.28, 0.04])
            mat_min = np.min(C_approx_nosimp)
            mat_max = np.max(C_approx_nosimp)
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            fig.add_axes([0.36, 0.13, 0.28, 0.28])
            plt.matshow(C_approx, fignum=False)
            plt.title("Map_approx_" + output_file_mode + "")
            fig.add_axes([0.36, 0.04, 0.28, 0.04])
            mat_min = np.min(C_approx)
            mat_max = np.max(C_approx)
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            fig.add_axes([0.68, 0.13, 0.28, 0.28])
            plt.matshow(C_rearrange_approx, fignum=False)
            plt.title("Map_rearrange_approx")
            fig.add_axes([0.68, 0.04, 0.28, 0.04])
            mat_min = np.min(C_rearrange_approx)
            mat_max = np.max(C_rearrange_approx)
            interval = round((mat_max - mat_min) / (length_color_map - 1), 2)
            color_map_mat = np.matrix([r * interval + mat_min for r in range(length_color_map)],
                                      dtype=float).reshape((1, length_color_map))
            color_map_list = [str(round(r, 2)) if e % 2 == 0 else "" for e, r in enumerate(color_map_mat.tolist()[0])]
            plt.matshow(color_map_mat, fignum=False)
            plt.yticks([])
            plt.xticks(np.arange(0, length_color_map), color_map_list)

            plt.suptitle("Tree " + str(i))
            # plt.tight_layout()
            plt.savefig(fname=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/maps/output_numBasis=" + str(
                attempt_num_basis_trees) + "_tree_" + str(i).zfill(2) + "_mat.jpg")
            plt.close()

        print "Outputting main results done"

    if output_GW_dist:

        print "Outputting GW-distance results..."
        losses = []
        for ii in range(num_trees):
            C_approx_tree, p_approx_tree = get_distance_and_distribution(T_approx_trees[ii].number_of_nodes(),
                                                                         T_approx_trees[ii])
            _, log = gwa.gromov_wasserstein_asym(C_approx_tree, CList[ii], p_approx_tree, pList[ii])
            losses.append(log['gw_dist'])
            NMF_loss += log['gw_dist']

        np.savetxt(subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/GW_Dist.txt",
                   np.asarray(losses, dtype=float), fmt="%.5f", delimiter="\n")
        draw_vecs([np.matrix(losses)], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/", filenames=["GW_Dist.jpg"])
        draw_vecs([np.matrix(np.log(np.asarray(losses) + 1))], path=subfolder + "/numBasis_" + str(attempt_num_basis_trees) + "/",
                  filenames=["GW_Dist_log.jpg"], log_scale=True)

        print >> f_gwloss_log, "Num_basis=" + str(attempt_num_basis_trees), ", loss=" + str(NMF_loss)
        if NMF_loss < best_NMF_loss and best_NMF_loss / NMF_loss >= 1.01:
            best_NMF_loss = NMF_loss
            num_basis_trees = attempt_num_basis_trees
            W = W0.copy()
            H = H0.copy()

        print "Outputting GW-distance results done"

# if output_GW_dist:
#     print >> f_gwloss_log, "Best num of basis trees = " + str(num_basis_trees), ", min GW_distance = " + str(best_NMF_loss)

f_gwloss_log.close()
f_solveloss_log.close()

exit()