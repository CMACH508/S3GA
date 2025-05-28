# Import the graph_tool module for visualization.
from graph_tool.all import * 

# Import matplotlib 
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import subgraph, remove_self_loops
import numpy as np

def construct_graph_between_clus(node_index1, node_index2, edge_index, output=None):
    node_index1 = node_index1.to(edge_index.device)
    node_index2 = node_index2.to(edge_index.device)
    all_nodes = torch.cat([node_index1, node_index2], dim=0)
    num_nodes = all_nodes.shape[0]
    g = Graph()
    g.add_vertex(num_nodes)
    edges, _ = subgraph(subset=all_nodes, edge_index=edge_index,
                         num_nodes= max(all_nodes.max(), edge_index.max())+1, relabel_nodes=True)
    g.add_edge_list(edges.transpose(0,1).cpu().numpy())
    temp = ["black" for _ in range(g.num_vertices())]
    temp[node_index1.shape[0]:] = ["red"] * node_index2.shape[0]
    node_colours = g.new_vertex_property("string",temp)

    c_map = plt.get_cmap('autumn')
    pos = sfdp_layout(g)
    graph_draw(g, pos=pos, output_size=(1000, 1000), 
               vertex_fill_color=node_colours, 
                edge_pen_width=1.0,
                edge_color = [0.0, 0, 0, 1.0],
                vcmap=c_map,
                output= f'{output}.png')

def construct_graph_(node_index, edge_index, output=None):
    num_nodes = node_index.size(0)
    g = Graph()
    g.add_vertex(num_nodes)
    edges, _ = subgraph(subset=node_index, edge_index=edge_index, 
                        num_nodes= max(node_index.max(), edge_index.max())+1, relabel_nodes=True)
    g.add_edge_list(edges.transpose(0,1).cpu().numpy())

    temp = ["black" for x in range(g.num_vertices())]
    node_colours = g.new_vertex_property("string",temp)
    c_map = plt.get_cmap('autumn')
    pos = sfdp_layout(g)
    print(pos)
    graph_draw(g, pos=pos, output_size=(1000, 1000), 
               vertex_fill_color=node_colours, 
                edge_pen_width=1.0,
                edge_color = [0.0, 0, 0, 1.0],
                vcmap=c_map,
                output= f'{output}.png')

def construct_graph(nodes, node_index, 
                    acc_index,
                    edge_index, N, output=None):
    num_nodes = nodes.size(0)
    g = Graph()
    g.add_vertex(num_nodes)

    edges, _ = subgraph(subset=node_index, edge_index=edge_index, num_nodes=N, relabel_nodes=True)
    g.add_edge_list(edges.transpose(0,1).numpy())
    # print(edges)
    v_embbeddings = nodes.numpy()
    vemb = g.new_vertex_property("vector<float>", v_embbeddings)

    temp = ["black" for x in range(g.num_vertices())]
    index = torch.nonzero(acc_index.unsqueeze(1)== node_index, as_tuple=True)[1]
    assert index.size(0) == acc_index.size(0), "acc is not correct!"
    for i in index:
        temp[i.item()] = 'red'
    node_colours = g.new_vertex_property("string",temp)

    c_map = plt.get_cmap('autumn')
    pos = sfdp_layout(g)
    print(pos)
    graph_draw(g, pos=pos, output_size=(1000, 1000), 
               vertex_fill_color=node_colours, 
                edge_pen_width=1.0,
                edge_color = [0.0, 0, 0, 1.0],
                vcmap=c_map,
                output= f'{output}.png')
    

def construct_graph_neighbor(nodes, node_index, node_ori_index,
                             ori_acc_index, neig_acc_index,
                             edge_index, N, output=None):
    num_nodes = nodes.size(0)
    g = Graph()
    g.add_vertex(num_nodes)

    edges, _ = subgraph(subset=node_index, edge_index=edge_index, num_nodes=N, relabel_nodes=True)
    # edges, _ = remove_self_loops(edge_index=edges)

    g.add_edge_list(edges.transpose(0,1).numpy())
    # print(edges)
    v_embbeddings = nodes.numpy()
    vemb = g.new_vertex_property("vector<float>", v_embbeddings)
    # all nodes: black
    # original acc nodes: red
    # neighbor acc nodes: green
    temp = ["black" for x in range(g.num_vertices())] 
    ori_acc_index = set(ori_acc_index.tolist())
    neig_acc_index = set(neig_acc_index.tolist()) - ori_acc_index

    for i in node_ori_index:
        temp[i.item()] = 'yellow'
    for i in ori_acc_index:
        temp[i] = 'red'
    for i in neig_acc_index:
        temp[i] = 'green'
    node_colours = g.new_vertex_property("string",temp)

    c_map = plt.get_cmap('autumn')
    pos = sfdp_layout(g)
    graph_draw(g, pos=pos, output_size=(1000, 1000), 
               vertex_fill_color=node_colours, 
                edge_pen_width=1.0,
                edge_color = [0.0, 0, 0, 1.0],
                vcmap=c_map,
                output= f'{output}.png')
    

def shortest_node_path(source_nodes, target_nodes, all_edge_index, num_nodes, clus):
    g = Graph()
    g.add_vertex(num_nodes)
    source_nodes = clus[source_nodes].numpy()
    target_nodes = clus[target_nodes].numpy()
    print(source_nodes.shape[0], target_nodes.shape[0])

    g.add_edge_list(all_edge_index.transpose(0,1).numpy())
    neig_count = {}
    for i in source_nodes:
        target=[g.vertex(index) for index in target_nodes if index != i]
        path = shortest_distance(g, source=g.vertex(i), target=target)
        indices = np.argmin(np.array(path))
        
        # print(i, np.min(np.array(path)), target[indices])
        vlist, _ = shortest_path(g, g.vertex(i), g.vertex(target[indices]))
        v_list = [int(v) for v in vlist]
        # print(i, v_list, )
        # count the number of times a 1-hop neighbor appears in the shortest path.
        if np.min(np.array(path)) == 2:
            if v_list[1] in neig_count:
                neig_count[v_list[1]] += 1
            else:
                neig_count[v_list[1]] = 1
       
    # print(neig_count)
    return neig_count