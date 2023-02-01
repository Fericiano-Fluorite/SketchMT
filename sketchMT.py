from weighted_hierarchy_pos import weighted_hierarchy_pos
from sklearn.decomposition import NMF
import readMergeTree as rmt
import column_selection as csel
import platform
from utilities import *


class GWMergeTree:
    def __init__(self, tree: nx.Graph, root: int):
        """ Initialization of a merge tree object

        Parameters
        ----------
        tree:  networkx.Graph
               the merge tree and its data
        root:  int
               the id of the root of the merge tree
        """
        self.tree = tree
        self.root = root

        try:
            assert nx.is_tree(self.tree)
            assert self.root in self.tree.nodes()
        except AssertionError:
            print("The tree data is incorrect! "
                  "Either the object is not a tree, or the root is not a valid point in the tree.")
            raise ValueError

    def label_validation(self, coordinate_labels: list, scalar_name="height", edge_weight_name="weight"):
        try:
            for node in self.tree.nodes():
                assert scalar_name in self.tree.nodes[node]
                for label in coordinate_labels:
                    assert label in self.tree.nodes[node]
        except AssertionError:
            print("The tree data is incorrect! "
                  "Either the scalar function name is not valid, or the label names for nodes are not valid.")
            raise KeyError

        try:
            for u, vs in self.tree.adjacency():
                for v, e in vs.items():
                    assert edge_weight_name in e
        except AssertionError:
            print("The edge weight name is incorrect!")
            raise KeyError


class SketchMT:
    def __init__(self,
                 trees: list,
                 dataset_name: str,
                 scalar_name="height",
                 edge_weight_name="weight",
                 weight_mode="shortestpath",
                 prob_distribution="uniform",
                 budget=3.0,
                 **params):

        """ Initialization for the pFGW feature tracking framework

        Parameters
        ----------
        trees : list[GWMergeTree],
                a list of GWMergeTree for feature tracking
        dataset_name: str,
                      It is required to identify the dataset, so that we can save the Frechet Mean in files
        scalar_name: str, default="height"
                     the name for the scalar field in GWMergeTree
        edge_weight_name: str, default="weight"
                     the name for the weight of edges in GWMergeTree
        weight_mode : str, default="shortestpath"
                      declare the strategy to generate the weight matrix W for the measure network.
                      Options are ["shortestpath", "lca"]
        prob_distribution: str, default="uniform"
                           declare the strategy to assign probability vector p to nodes
                           Options are ["uniform", "ancestor"]
        budget: float, default=3.0
                parameter for the max size of the Frechet Mean, which equals to budget * max(|Tree_i|)

        References
        ----------
        .. [1]
        """

        self.trees = [x.tree for x in trees]
        self.roots = [x.root for x in trees]

        self.max_tree_size = np.max([T.number_of_nodes() for T in self.trees])
        self.max_tree_id = np.argmax([T.number_of_nodes() for T in self.trees])
        print("Max Tree Size, ID = {}, {}".format(str(self.max_tree_size), str(self.max_tree_id)))

        for tree in trees:
            tree.label_validation([], scalar_name, edge_weight_name)

        self.dataset = dataset_name

        # The maximum size of blow-up graph in GW framework, usually 2 to 3 times of the size of the largest input tree.
        # We only modify the multiplier
        # If we do not set the maximum size, the blow-up graph can be exponentially large depending on the input data
        try:
            self.budget = float(budget)
        except TypeError:
            print("budget must be a number!")

        # parameters to simplify merge trees
        # 0.5, 2.0 for CornerFlow
        # 2.0, 2.0 for redSeaNew
        # 2.0, 1.0 for heatedFlow
        # 1024.0, 8.0 for Gaussian_rotated
        try:
            # self.lambda_zero = 2.0 if "lambda_zero" not in params else float(params["lambda_zero"])
            # self.lambda_limit = 2.0 if "lambda_limit" not in params else float(params["lambda_limit"])
            self.lambda_persistence = 0.0 if "lambda_persistence" not in params else float(params["lambda_persistence"])
        except TypeError:
            print("lambda_zero and lambda_limit must be float numbers!")

        try:
            self.GWIteration = 50 if "GWIteration" not in params else int(params["GWIteration"])
        except TypeError:
            print("GWIteration must be an integer!")

        self.STMode = "MST" if "spanning_tree_mode" not in params else params["spanning_tree_mode"]
        if self.STMode not in {"MST", "LSST"}:
            print("Spanning tree mode has to be either MST or LSST! Using MST as default.")
            self.STMode = "MST"

        self.weight_mode = weight_mode
        if self.weight_mode is not None and self.weight_mode not in {"shortestpath", "lca"}:
            print("Weight matrix mode undefined! Use function value difference as default.")
            self.weight_mode = None

        self.prob_distribution = prob_distribution
        if self.prob_distribution not in {"uniform", "ancestor"}:
            print("Probability Distribution of nodes has to be \'uniform\' or \'ancestor\'! "
                  "Using uniform distribution by default.")
            self.prob_distribution = "uniform"

        try:
            self.seeds = [31] if "seeds" not in params else list(params["seeds"])
        except TypeError:
            print("seeds have to be iterables of integers!")

        try:
            # self.sample = False if "sample" not in params else bool(params["sample"])
            # self.horizontal_edge = False if "horizontal_edge" not in params else bool(params["horizontal_edge"])
            self.retest = False if "retest" not in params else bool(params["retest"])
        except TypeError:
            print("retest, sample, and horizontal_edge parameters have to be boolean values!")

        self.scalar_name = scalar_name
        self.edge_weight_name = edge_weight_name

        self.num_trees = len(self.trees)

        self.CList = []
        self.pList = []

        self.CnBase = None
        self.pnBase = None
        self.Frechet_Loss = None

        self.trees_vecs = None
        self.p_vecs = None

        self.initialize_measure_network()
        self.compute_frechet_mean()
        self.compute_blowup()
        self.get_trees_vecs()

    def initialize_measure_network(self):
        max_C = None
        min_C = None
        # Initialize the measure network intrinsic weight & probability distribution
        for e in range(self.num_trees):
            T = self.trees[e]

            # C refers to the measure network intrinsic weight matrix. shape: (N, N)
            # p refers to the probability distribution across nodes. shape: (N, )
            C, p = get_distance_and_distribution(T,
                                                 distribution=self.prob_distribution,
                                                 weight_mode=self.weight_mode,
                                                 root=self.roots[e],
                                                 scalar_name=self.scalar_name,
                                                 edge_weight_name=self.edge_weight_name
                                                 )
            if max_C is None:
                max_C = np.max(C)
            else:
                max_C = max(max_C, np.max(C))
            if min_C is None:
                min_C = np.min(C)
            else:
                min_C = min(min_C, np.min(C))

            self.CList.append(C)
            self.pList.append(p)

        # print(self.CList[0])
        # for e in range(self.num_trees):
        #     self.CList[e] /= (max_C - min_C)
        # print(self.CList[0])

    def compute_frechet_mean(self):
        print("Computing Frechet Mean...")
        # Compute C, p for the Frechet Mean graph using GW framework
        budget = int(self.max_tree_size * self.budget)
        print("budget = " + str(budget))

        base_path = "./tmp/"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        hash_string_lst = [self.dataset, self.prob_distribution, self.weight_mode, str(budget)]
        hash_string = hashcode(hash_string_lst) + ".txt"

        CnBase_path = os.path.join(base_path, 'CnBase-' + hash_string)
        pnBase_path = os.path.join(base_path, 'pnBase-' + hash_string)
        Frechet_path = os.path.join(base_path, 'Frechet-' + hash_string)

        if self.retest or not os.path.exists(CnBase_path):
            base_index = self.max_tree_id

            CList_toCompare = copy.deepcopy(self.CList)
            pList_toCompare = copy.deepcopy(self.pList)

            # CList_toCompare = copy.deepcopy(self.CList[:base_index])
            # CList_toCompare.extend(self.CList[base_index + 1:])
            # pList_toCompare = copy.deepcopy(self.pList[:base_index])
            # pList_toCompare.extend(self.pList[base_index + 1:])

            self.CnBase, self.pnBase, self.Frechet_Loss = gwa.network_karcher_mean_armijo_sched_compress(
                self.CList[base_index], self.pList[base_index], CList_toCompare, pList_toCompare, budget, self.GWIteration)

            np.savetxt(CnBase_path, self.CnBase, delimiter=",")
            np.savetxt(pnBase_path, self.pnBase, delimiter=",")
            np.savetxt(Frechet_path, self.Frechet_Loss, delimiter=",")
        else:
            self.CnBase = np.loadtxt(CnBase_path, delimiter=",")
            self.pnBase = np.loadtxt(pnBase_path, delimiter=",")
            self.Frechet_Loss = np.loadtxt(Frechet_path, delimiter=",")
        print("Computing Frechet Mean Done")

    def compute_blowup(self):
        print("Computing Blowups...")
        if self.CnBase is None:
            self.compute_frechet_mean()

        self.opt_couplings = [[] for i in range(self.num_trees)]
        self.blowup_couplings = [[] for i in range(self.num_trees)]
        self.C_blowups = [[] for i in range(self.num_trees)]
        self.p_blowups = [[] for i in range(self.num_trees)]
        self.meanC_blowups = [[] for i in range(self.num_trees)]
        self.meanp_blowups = [[] for i in range(self.num_trees)]
        self.root_blowups = [[] for i in range(self.num_trees)]
        self.tangent_vecs = [[] for i in range(self.num_trees)]

        # X0, X1, v.shape = (num_lsst_nodes, num_lsst_nodes), to be flattened
        self.N_blowup = -1
        for i in range(self.num_trees):
            CnBase_copy = copy.deepcopy(self.CnBase)
            pnBase_copy = copy.deepcopy(self.pnBase)

            # WARNING: log_map does reorder all nodes based on the matching
            X0, X1, p0, p1, v, oc, log = gwa.log_map_ignore(CnBase_copy, self.CList[i],
                                                            pnBase_copy, self.pList[i],
                                                            G0=(None if (i < 1) or
                                                                (len(self.pList[i]) != len(self.pList[i - 1]))
                                                                else self.opt_couplings[i - 1]))

            self.root_blowups[i] = np.argmax(oc[:, self.roots[i]])

            self.opt_couplings[i] = oc

            self.C_blowups[i] = X1
            self.p_blowups[i] = p1

            assert X1.shape[0] == X1.shape[1]
            if self.N_blowup == -1:
                self.N_blowup = X1.shape[0]
            else:
                assert self.N_blowup == X1.shape[0]

            self.meanC_blowups[i] = X0
            self.meanp_blowups[i] = p0
            self.tangent_vecs[i] = v

        print("Computing Blowups Done")

    def get_trees_vecs(self):
        print("Getting Trees Vectors...")
        if self.trees_vecs is not None and self.p_vecs is not None:
            return self.trees_vecs, self.p_vecs

        # We only get the upper triangular part of each matrix and flatten it into a vector
        self.trees_vecs = np.array([[self.C_blowups[i][x][y] for x in range(self.N_blowup)
                                     for y in range(self.N_blowup) if x <= y] for i in range(self.num_trees)],
                                   dtype=float)
        self.trees_vecs = np.asmatrix(self.trees_vecs).transpose()
        self.p_vecs = np.asmatrix(self.p_blowups).transpose()

        print("Getting Trees Vectors Done")
        return self.trees_vecs, self.p_vecs

    def get_trees_from_blowups(self, trees_vecs, p_vecs,
                               num_trees=0, N_blowup=0, ref=None, no_simp=True):
        if num_trees == 0:
            num_trees = self.num_trees
        if N_blowup == 0:
            N_blowup = self.N_blowup
        if ref is None:
            ref = self.root_blowups

        return get_trees_from_matrices(trees_vecs, p_vecs,
                                       num_trees, N_blowup,
                                       lambda_persistence=self.lambda_persistence,
                                       # lambda_zero=self.lambda_zero,
                                       # lambda_limit=self.lambda_limit,
                                       weight_mode=self.weight_mode,
                                       # eliminate_internal_nodes=False,
                                       # eliminate_fans=True,
                                       ref=ref, no_simp=no_simp)

    def sketching(self, sketching_mode, num_basis_list, seed=31):
        if self.trees_vecs is None or self.p_vecs is None:
            self.get_trees_vecs()

        sketch_losses = []
        GW_losses = []
        selected_indices = []
        H0s = []
        basis_trees = []
        for e, attempt_num_basis_trees in enumerate(num_basis_list):
            frame_name = attempt_num_basis_trees
            print("Sketching: ", frame_name)
            selected_index = None

            # NMF
            if sketching_mode == "NMF":
                NMF_model = NMF(n_components=attempt_num_basis_trees, init='nndsvda', random_state=seed)
                W0 = NMF_model.fit_transform(self.trees_vecs)
                H0 = NMF_model.components_
                approx_trees_vecs = W0.dot(H0)
                diff = self.trees_vecs - approx_trees_vecs
                norm = np.linalg.norm(diff, ord=None, axis=0)
                sketch_loss = np.multiply(norm, norm)
            # Column Selection strat
            elif (sketching_mode == "CSS-IFS") or (sketching_mode == "CSS-LSS"):
                W0, selected_index = csel.column_selection(self.trees_vecs, self.num_trees, attempt_num_basis_trees,
                                                               mode=sketching_mode, seed=seed)
                H0, sketch_loss, _, _ = np.linalg.lstsq(W0, self.trees_vecs, rcond=None)
                approx_trees_vecs = W0.dot(H0)
            else:
                raise NotImplementedError

            if W0.shape[1] < attempt_num_basis_trees:
                print("Too many basis trees, column selection may fail.")
                sketch_losses.append(np.asarray(sketch_loss, dtype=float).reshape(-1, ))
                continue

            H0s.append(H0)

            # print("Decomposition Done")

            # Compute the sketch loss
            sketch_losses.append(np.asarray(sketch_loss, dtype=float).reshape(-1, ))

            # Compute the GW loss
            T_blowup_trees, mats_blowup_trees, p_blowup_trees = self.get_trees_from_blowups\
                                                                              (approx_trees_vecs, self.p_blowups,
                                                                               self.num_trees, self.N_blowup,
                                                                               ref=None, no_simp=True)
            # plt.subplot(121)
            # nx.draw_networkx(self.trees[0], pos=weighted_hierarchy_pos(self.trees[0], self.roots[0]))
            # plt.subplot(122)
            # nx.draw_networkx(T_blowup_trees[0], pos=weighted_hierarchy_pos(T_blowup_trees[0], T_blowup_trees[0].root))
            # plt.show()

            GW_loss = 0
            for i in range(self.num_trees):
                # print(T_blowup_trees[i].number_of_nodes(), T_blowup_trees[i].root)
                # for edges in T_blowup_trees[i].adjacency():
                #     print(edges)
                # C_recover, p_recover = get_distance_and_distribution(T_blowup_trees[i],
                #                                                      distribution=self.prob_distribution,
                #                                                      weight_mode=self.weight_mode,
                #                                                      root=T_blowup_trees[i].root,
                #                                                      scalar_name="height",
                #                                                      edge_weight_name="weight"
                #                                                      )
                # print(p_blowup_trees[i])
                # print(sum(self.pList[i]), sum(self.p_blowups[i]))
                _, gw_dist = gwa.gromov_wasserstein_asym(self.CList[i],mats_blowup_trees[i],self.pList[i],p_blowup_trees[i])
                GW_loss += gw_dist['gw_dist']
            GW_losses.append(GW_loss)

            # Compute the basis trees
            selected_indices.append(selected_index)

            # Only works for NMF
            if sketching_mode == "NMF":
                T_basis_trees, mats_basis, _ = self.get_trees_from_blowups(W0, None,
                                                                           attempt_num_basis_trees,
                                                                           self.N_blowup,
                                                                           ref=None, no_simp=False)
                for t in range(attempt_num_basis_trees):
                    T_basis_trees[t] = get_root(T_basis_trees[t])
                basis_trees.append((T_basis_trees, mats_basis))

        if sketching_mode == "NMF":
            return sketch_losses, H0s, basis_trees, GW_losses
        else:
            return sketch_losses, H0s, selected_indices, GW_losses

    def set_root(self, id: int, new_root: int):
        if id >= self.num_trees:
            print("ID out of boundary")
            return

        if new_root not in self.trees[id].nodes():
            print("New root outside of the tree")
            return

        self.trees[id].root = new_root

    def pairwise_instance_dist(self):
        dist_instance = np.zeros((self.num_trees, self.num_trees))
        for ei, i in enumerate(self.trees):
            for ej, j in enumerate(self.trees):
                if ei >= ej:
                    continue
                CBase = copy.deepcopy(self.CList[ei])
                pBase = copy.deepcopy(self.pList[ei])
                CCompare = copy.deepcopy(self.CList[ej])
                pCompare = copy.deepcopy(self.pList[ej])

                _, log = gwa.gromov_wasserstein_asym(CBase, CCompare,
                                                     pBase, pCompare,
                                                     G0=None)

                dist_instance[ei][ej] = dist_instance[ej][ei] = log['gw_dist']
        return dist_instance
       
    def tree_vec_residual(self, idx, idy):
        if (idx >= self.num_trees) or (idy >= self.num_trees):
            raise OverflowError

        vec_x = self.trees_vecs[:, idx]
        vec_y = self.trees_vecs[:, idy]
        res = vec_x - vec_y
        x_sum = np.sum(np.dot(vec_x.T, vec_x))
        y_sum = np.sum(np.dot(vec_y.T, vec_y))
        res_sum = np.sum(np.dot(res.T, res))
        print(x_sum, y_sum, res_sum)
