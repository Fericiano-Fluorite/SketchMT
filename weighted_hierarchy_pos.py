import networkx as nx
import random
import copy

def weighted_hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use weighted_hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    subtree_size = {}

    def _get_subtree_length(G, root, parent=None):
        sum_weight = 1
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and (parent is not None):
            children.remove(parent)
        if len(children) != 0:
            for child in children:
                sub_weight = _get_subtree_length(G, child, parent=root)
                sum_weight += sub_weight

        subtree_size[root] = sum_weight
        return sum_weight

    _get_subtree_length(G, root)

    def _weighted_hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0,
                    xcenter = 0.5, rootpos = None,
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        duplicates = []
        distances = {}
        parents = {}

        if not isinstance(G, nx.DiGraph) and parent is not None and parent in children:
            children.remove(parent)
        if len(children) != 0:
            if len(children) <= (3 if parent is not None else 2):
                children.sort(key=lambda x: G.get_edge_data(root, x)['weight'])
                children.sort(key=lambda x: subtree_size[x])
            else:
                children.sort(key=lambda x: G.get_edge_data(root, x)['weight'])
                def rearrage(children):
                    l = len(children)
                    order_up = [i for i in range(0, l, 2)]
                    order_down = [i+1 for i in range(0, l, 2)]
                    if l%2:
                        order_down.pop()
                    order_down.reverse()
                    order_up.extend(order_down)
                    assert len(order_up) == l
                    children = [children[order_up[i]] for i in range(l)]
                    return children
                children = rearrage(children)

            children_copy = copy.deepcopy(children)

            def find_duplicates(child, root):
                edge_weight = G.get_edge_data(root, child)['weight']
                height = vert_loc - edge_weight
                if 'height' in G.nodes[child]:
                    height = - G.nodes[child]['height']

                if abs(height - vert_loc) < 1e-6:
                    duplicates.append(child)
                    if child in children:
                        children.remove(child)

                    neighbors = list(G.neighbors(child))
                    for neighbor in neighbors:
                        if neighbor != root:
                            find_duplicates(neighbor, child)
                else:
                    if child not in children:
                        parents[child] = root
                        distances[child] = abs(height - vert_loc)
                        children.append(child)

            for child in children_copy:
                find_duplicates(child, root)

            if len(children) < 1:
                leaf_count = 1
                leafpos[root] = (leftmost, vert_loc)
            else:
                rootdx = width / len(children)
                nextx = xcenter - width / 2 - rootdx / 2
                for child in children:
                    if child in distances:
                        edge_weight = distances[child]
                    else:
                        edge_weight = G.get_edge_data(root, child)['weight']
                    height = vert_loc - edge_weight
                    if 'height' in G.nodes[child]:
                        height = - G.nodes[child]['height']

                    nextx += rootdx
                    rootpos, leafpos, newleaves = _weighted_hierarchy_pos(G,child, leftmost+leaf_count*leafdx,
                                        width=rootdx, leafdx=leafdx,
                                        vert_gap = 0, vert_loc = height,
                                        xcenter=nextx,
                                        rootpos=rootpos, leafpos=leafpos, parent = root if child not in parents else parents[child])

                    leaf_count += newleaves

                leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
                rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
                leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)

        if len(duplicates) > 0:
            for dup in duplicates:
                rootpos[dup] = rootpos[root]
                leafpos[dup] = leafpos[root]
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    if 'height' in G.nodes[root]:
        vert_loc = -G.nodes[root]['height']
    if leafcount == 0:
        return [(0, 0)]
    _get_subtree_length(G, root)
    rootpos, leafpos, leaf_count = _weighted_hierarchy_pos(G, root, 0, width,
                                                    leafdx=width*1./leafcount,
                                                    vert_gap=vert_gap,
                                                    vert_loc = vert_loc,
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        try:
            len(leafpos[node][1])
            pos[node] = (
                leaf_vs_root_factor * leafpos[leafpos[node][1][1]][0] + (1 - leaf_vs_root_factor) * rootpos[leafpos[node][1][1]][0] - 0.01, leafpos[node][1][0])
        except TypeError:
            pos[node] = (
                leaf_vs_root_factor * leafpos[node][0] + (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][1])

#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())

    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])

    return pos