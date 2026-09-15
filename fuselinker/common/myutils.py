import numpy as np
import torch
import dgl
import random

#Tính norm= 1/ bậc của node với đầu vào là 1 ma trận kề của đồ thị
def compute_degree_norm(g: dgl.DGLGraph):
    g = g.local_var()
    # g.number_of_nodes(): Trả về số lượng node trong đồ thị
    # range(g.number_of_nodes()):trả về danh sách node từ 0->N-1
    # g.in_degrees(..) : Với mỗi node v, trả về số cạnh đi vào v(bao nhiêu cạnh có dạng u->v)
    in_degs = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = np.zeros_like(in_degs, dtype=np.float32)
    np.divide(1.0, in_degs, out=norm, where=in_degs != 0)
    return norm #mảng chứa norm của mỗi node

#Xây dựng đồ thị DGL từ danh sách các triple 
# Trả về đồ thị(node,cạnh), danh sách các quan hệ và norm của node
def build_graph_from_triples(num_nodes, num_rels, triples):
    g = dgl.DGLGraph() # Tạo đồ thị rỗng 
    g.add_nodes(num_nodes) #Thêm num_nodes vào đồ thị, lúc này đồ thị sẽ có các node 0,1,2,3,...num_nodes-1
    src, rel, dst = triples #(src,rel,dst): có kích thước bằng nhau ,rel: quan hệ đc lặp lại nhiều lần
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src)) #Thêm cạnh ngược để quá trình lan truyền thông tin 2 chiều
    rel = np.concatenate((rel, rel + num_rels)) #Thêm quan hệ ngược để quá trình lan truyền thông tin 2 chiều
    edges = sorted(zip(dst, src, rel))  #[(dst, src, rel), (), ()...]
    dst, src, rel = np.array(edges).transpose()  # ma trận(N,3) biến thành ma trận (3,N)
    g.add_edges(src, dst) 
    nodes_norm = compute_degree_norm(g)  # 1./in-degree
    return g, rel.astype('int64'), nodes_norm.astype('float32')
    #Trả về đồ thị g, danh sách các quan hệ rel và norm của mỗi node

# Xây dựng đồ thị từ danh sách cạnh(họ tạo ra các triple từ danh sách cạnh)
def build_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose() # biến ma trận (N,3) thành ma trận (3,N)
    return build_graph_from_triples(num_nodes, num_rels, triples=(src, rel, dst))

# Duyệt từng cạnh (u->v) và gán edge.norm =norm(v) =1/ bậc(v) :Chuyển norm node thành norm cạnh
def node_norm_2_edge_norm(g: dgl.DGLGraph, node_norm):
    g = g.local_var() #Tạo ra bản sao của g 
    g.ndata['norm'] = node_norm #Gán norm làm thuộc tính của mỗi node, thuộc tính đó tên là norm
    #Duyệt qua từng cạnh của đồ thị, gán norm của cạnh = norm của node đích
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']

#Xây dựng danh sách kề từ các triples( ) của node :Danh sách này gồm num_nodes phần tử
#Kết quả trả về: Với mỗi node u, adj_list[u] chứa các cặp [edge_id,v] là hàng xóm của u
def get_adj(num_nodes, triplets):
    #Tạo 1 list rỗng cho từng node 
    adj_list = [[] for _ in range(num_nodes)]
    # Duyệt qua từng triple và thêm cạnh 
    for i, triple in enumerate(triplets): # i là idex của triple
        src, dst = triple[0], triple[2]
        # both directions have same id
        adj_list[src].append([i, dst])  # [edge_id, dst]
        adj_list[dst].append([i, src]) #Chuyển list thành numpy_array 

    adj_list = [np.array(n) for n in adj_list]
    return adj_list 

#Tạo ra negative sample bằng cách thay thế ngẫu nhiên head và tail thành các entity khác
#INPUT: pos_samples: mảng chứa các positive samples (B,3) với B là số lượng positive tripple trong batch
#num_entity: số lượng entity trong dataset, dùng id thuộc [0, num_entity-1]
def negative_sampling(pos_samples, num_entity, negative_rate):
    batch_size = len(pos_samples) #Số positive triple trong batch
    generate_num = batch_size * negative_rate # Tổng số negative cần sinh
    neg_samples = np.tile(pos_samples, (negative_rate, 1))  # [generate_num, 3]
    labels = np.zeros(batch_size * (negative_rate + 1), dtype=np.float32) # mảng nhãn này gồm cả nhãn cho negative sample và positive sample
    labels[:batch_size] = 1 # đây là  nhãn của positive sample, đặt các nhãn đó =1
    values = np.random.randint(num_entity, size=generate_num)  #SInh ra generate_num số nguyên, mỗi số nằm trong khoảng [0,num_entity-1]
    choices = np.random.uniform(size=generate_num) #Sinh ra generate_num số thực trong khoảng từ [0,1]
    sub = choices > 0.5 
    obj = choices <= 0.5
        # randomly replace sbj or obj
    neg_samples[sub, 0] = values[sub]
    neg_samples[obj, 2] = values[obj]
    # Lẫy những phần tử ở cột subject của những dòng mà sub == True
    return np.concatenate((pos_samples, neg_samples)), labels
    #Trả về mảng mẫu huấn luyện gồm các positive sample trước và negative sample sau
    # Vector nhãn tương ứng  

# n_triplets: tổng số triplet cạnh hiện có trong dataset
# sample_size: số lượng cạnh muốn lấy mẫu 
def sample_edge_uniform(n_triplets, sample_size):
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)
#Chọn ngẫu nhiên sample_size cạnh từ tất cả các cạnh trong dataset ,replace =False: Không lặp lại cạnh

# mỗi node u có 1 danh sách kề adj_list[] với mỗi phần tử adj_list[u] chứa các cặp [edge_id,v] là hàng xóm của u
#degrees: bậc của từng node 
# n_triplets: tổng số cạnh trong đồ thị
# sample_size: Số lượng cạnh muốn lấy mẫu 
#Mục đích hàm: Chọn sample_size cạnh từ đồ thị theo kiểu mở rộng dần các node được chọn
def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    #Tạo mảng edges có  độ dài sample_size để lưu các id cạnh được chọn
    edges = np.zeros((sample_size,), dtype=np.int32)
    #sample_counts: mảng chứa số lượng cạnh còn lại của mỗi node 
    sample_counts = np.array([d for d in degrees])
    #picked: mảng đánh dấu cạnh đã được chọn chưa -> picked[i] = True nếu cạnh i đã được chọn
    picked = np.array([False for _ in range(n_triplets)])
    #seen: mảng đánh dấu node đã được chọn hay chưa-> seen[i] = True nếu node i đã từng xuất hiện trong các cạnh của sample
    seen = np.array([False for _ in degrees])

    #Mỗi vòng lặp chọn 1 cạnh cho đến khi đủ sample_size cạnh
    for i in range(sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            # all nodes are unseen, pick one node uniformly
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        p = weights / np.sum(weights)
        #Chọn ngẫu nhiên 1 node trong tất cả các node  với xác suất p; chosen_vertex: node được chọn
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=p)
        # lấy ra danh sách các node kề với node được chọn
        chosen_adj_list = adj_list[chosen_vertex]
        
        #Chọn ngẫu nhiên 1 cạnh trong danh sách các node kề vs node được chọn
        #ví dụ: chonsen_adj_list = [[1,2],[3,4],[5,6]] -> chosen_edge = [1,2]
        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0])) 
        #Lấy ra cạnh được chọn
        chosen_edge = chosen_adj_list[chosen_edge]
        #Lât ra id cạnh được chọn
        edge_number = chosen_edge[0]
        
        #nếu cạnh đã được chọn rồi thì ta sẽ chọn cạnh khác
        while picked[edge_number]:
            # this edge is already picked
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number #Đưa id cạnh vào kết quả 
        other_vertex = chosen_edge[1]  # node còn lại trong cạnh được chọn
        picked[edge_number] = True  # Đánh dấu rằng cạnh này đã được chọn 1 lần rồi 
        sample_counts[chosen_vertex] -= 1  # in-degree of chosen-vertex minus one (this edge is deleted from graph
        sample_counts[other_vertex] -= 1 # Đánh dấu rằng bậc của node còn lại bị giảm đi 1
        seen[chosen_vertex] = True # Đánh dấu node được chọn đã xuất hiện trong sample
        seen[other_vertex] = True

    return edges #Mảng chứa chỉ số của sample_size cạnh được lấy mẫu


def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_rels, adj_list, degrees, negative_rate,
                                        sampler='uniform'):
        # biến sampler này quyết định xem là chọn ra sample_size cạnh ngẫu nhiên hay theo kiểu mở rộng
        if sampler == "uniform":
            edges = sample_edge_uniform(len(triplets), sample_size)  # edge id
        elif sampler == "neighbor":
            edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
        else:
            raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")
        
        #Lẩy ra các cạnh được chọn từ danh sách triples
        edges = triplets[edges]
        src, rel, dst = edges.transpose()
        # node_id ban đầu có thể rất lớn nên ta chuẩn hóa về các node liên tục
        #uniq_v: mảng chứa các node đã được chuẩn hóa 0123(đã sắp xếp và ko có trùng lặp).Không cùng độ dài với edges ban đầu nữa
        #edges: mảng cùng độ dài với edges ban đầu nhưng node_id đã được thay thế
        uniq_v, edges = np.unique([src, dst], return_inverse=True)
        # Chuyển đổi id thành src và dst liên tục
        src, dst = np.reshape(edges, (2, -1))
        #Đây là triple mới khi đã chuyển đổi node_id  
        relabeled_edges = np.stack([src, rel, dst]).transpose()  # [sample_size, 3]

        # Samples : gồm cả negative sample và positive sample
        # labels : nhãn của các sample
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v), negative_rate)

        
        #=> chia batch thành 2 phần: 1 phần dùng để x

        split_size = int(sample_size * split_size)
        #Chỉ số của các cạnh được chọn để build graph
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size)

        #Chỉ số  các src, rel, dst trong mảng edges được chọn để build graph
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]
       
        g, rel, norm = build_graph_from_triples(len(uniq_v), num_rels, (src, rel, dst))

        return g, uniq_v, rel, norm, samples, labels


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def perturb_and_get_raw_rank(emb, w, a, r, b, test_size, batch_size=100):
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    emb = emb.transpose(0, 1)
    w = w.transpose(0, 1)
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = emb[:,batch_a] * w[:,batch_r]
        emb_ar = emb_ar.unsqueeze(2)
        emb_c = emb.unsqueeze(1)

        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0).sigmoid()
        target = b[batch_start: batch_end]

        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return torch.cat(ranks)


def filter(triplets_to_filter, test_nodes, target_s, target_r, target_o, num_nodes, neg_sample_size_eval, filter_o=True):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)

    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]

    while len(candidate_nodes) < (neg_sample_size_eval + 1):
        # e = np.random.randint(0, num_nodes)
        e = random.choice(test_nodes)
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter and triplet not in candidate_nodes:
            candidate_nodes.append(e)
    return torch.LongTensor(candidate_nodes)


def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, neg_sample_size_eval, filter_o=True):
    num_nodes = emb.shape[0]
    ranks = []
    test_nodes = torch.unique(torch.cat((s, o))).tolist()
    for idx in range(test_size):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(triplets_to_filter, test_nodes, target_s, target_r,
                                 target_o, num_nodes, neg_sample_size_eval, filter_o=filter_o)
        if filter_o:
            emb_s = emb[target_s]  # Vector
            emb_o = emb[candidate_nodes]  # A set of vectors
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o  # Distmult
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks) 
    # Trả về chỉ số rank của từng cạnh thật trong mảng scores

#Tính chỉ số MRR,MR và list_hist
def _calc_mrr(emb, w, test_triplets, total_data, batch_size, neg_sample_size_eval, hits, filter=False):
    with torch.no_grad():
        s, r, o = test_triplets[:,0], test_triplets[:,1], test_triplets[:,2]
        test_size = len(s)

        if filter:
            triplets_to_filter = {tuple(triplet) for triplet in total_data.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, neg_sample_size_eval, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter, neg_sample_size_eval)
        else:
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mr = torch.mean(ranks.float()).item()
        mrr = torch.mean(1.0 / ranks.float()).item()
        hits_dict = dict()
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            hits_dict[hit] = avg_count

    return mr, mrr, hits_dict


def calc_mrr(emb, w, test_triplets, total_data, batch_size=100, neg_sample_size_eval=20, hits=[1, 3, 10], eval_p="filtered"):
    if eval_p == "filtered":
        mr, mrr, hits_dict = _calc_mrr(emb, w, test_triplets, total_data, batch_size, neg_sample_size_eval, hits, filter=True)
    else:
        mr, mrr, hits_dict = _calc_mrr(emb, w, test_triplets, total_data, batch_size, hits)
    return mr, mrr, hits_dict