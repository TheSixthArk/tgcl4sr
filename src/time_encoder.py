# Codes are based on GCL4SR
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from utils import ScaledDotProductAttention


class TiConv(pygnn.MessagePassing):
    def __init__(self, args, head=2, add_self_loop=True, normalize=True, bias=True):
        super(TiConv, self).__init__(aggr='add', node_dim=0)
        self.head = head
        self.device = args.device
        self.remain = 3 * args.hidden_channel // head
        self.in_channels = args.hidden_channel
        self.out_channels = args.hidden_channel
        self.time_channels = args.hidden_channel
        # 多头注意力机制
        self.w_q = nn.Linear(3 * args.hidden_channel, 3 * args.hidden_channel, bias=False)
        self.w_k = nn.Linear(3 * args.hidden_channel, 3 * args.hidden_channel, bias=False)
        self.w_v = nn.Linear(3 * args.hidden_channel, 3 * args.hidden_channel, bias=False)
        self.attn = ScaledDotProductAttention(temperature=np.power(args.hidden_channel, 0.5), attn_dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(3 * args.hidden_channel, 3 * args.hidden_channel),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(3 * args.hidden_channel)
        # 特征融合
        self.merge = nn.Sequential(
            nn.Linear(4 * args.hidden_channel, args.hidden_channel),
            nn.LeakyReLU(0.1),
            nn.Linear(args.hidden_channel, args.hidden_channel)
        )
        # 时间编码器
        self.basic_freq = torch.zeros([1, args.hidden_channel]).to(self.device)
        self.bias = torch.zeros([args.hidden_channel]).to(self.device)
        self.lin = nn.Linear(2 * args.hidden_channel, args.hidden_channel)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.normal_(self.w_q.weight, std=0.1)
        nn.init.normal_(self.w_k.weight, std=0.1)
        nn.init.normal_(self.w_v.weight, std=0.1)
        nn.init.normal_(self.ffn[0].weight, std=0.1)
        nn.init.normal_(self.merge[0].weight, std=0.1)
        nn.init.normal_(self.merge[2].weight, std=0.1)
        nn.init.normal_(self.basic_freq, std=0.1)
        nn.init.normal_(self.lin.weight, std=0.1)

    def func(self, ts):
        ts = ts.view(-1, 1)
        pos = ts * self.basic_freq
        pos += self.bias
        return torch.cos(pos)

    def time_encoder(self, attr):
        t2, t1 = attr[:, 1], attr[:, 0]
        t2_emb, t1_emb = self.func(t2), self.func(t1)
        a = self.lin(torch.cat([t1_emb, t2_emb], dim=1))
        return a

    def multi_attn(self, q, k, v, mask=None):
        residual = q
        q_size, k_size, v_size = q.shape[0], k.shape[0], v.shape[0]
        q, k, v = self.w_q(q).view(q_size, 1, self.head, self.remain), self.w_k(k).view(k_size, 1, self.head, self.remain), self.w_v(v).view(v_size, 1, self.head, self.remain)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.remain)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.remain)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.remain)
        output, attn = self.attn(q, k, v)
        output = output.view(self.head, k_size, 1, self.remain).permute(1, 2, 0, 3).contiguous().view(k_size, 1, -1)
        output = self.ffn(output)
        output = self.layer_norm(output + residual)
        return output, attn

    def attention(self, src_emb, src_t_emb, ngh_rmb, ngh_t_emb):
        q, k = torch.cat([src_emb, src_t_emb], dim=1).unsqueeze(1), torch.cat([ngh_rmb, ngh_t_emb], dim=1).unsqueeze(1)
        output, attn = self.multi_attn(q, k, k)
        output, attn = output.squeeze(), attn.squeeze()
        return output, attn

    def forward(self, x, edge_index, edge_attr, emb):
        x_l = x_r = x
        time, index = edge_attr[:, :2], edge_attr[:, 2].long()
        edge_attr = torch.cat([self.time_encoder(time), F.embedding(index, emb)], dim=1)
        output = self.propagate(x=(x_l, x_r), edge_index=edge_index, edge_attr=edge_attr)
        output = self.merge(torch.cat([x, output], dim=1))
        return output

    def message(self, x_j, x_i, edge_attr):
        j_t_emb, user_emb = edge_attr[:, :self.in_channels], edge_attr[:, self.in_channels:]
        i_t_emb = self.time_encoder(torch.zeros(x_i.shape[0], 2).to(self.device))
        j_t_emb = torch.cat([j_t_emb, user_emb], dim=1)
        i_t_emb = torch.cat([i_t_emb, user_emb], dim=1)
        output, alpha = self.attention(x_i, i_t_emb, x_j, j_t_emb)
        return output

class GlobalGNN(nn.Module):
    def __init__(self, args, time_graph):
        super(GlobalGNN, self).__init__()
        self.args = args
        self.device = args.device
        self.hidden_channel = args.hidden_channel
        self.time_graph = time_graph.to(self.device)
        in_channels = hidden_channels = self.hidden_channel
        self.num_layers = len(args.sample_size)
        self.dropout = nn.Dropout(0.5)
        self.gcn = TiConv(args)
        self.convs = nn.ModuleList()
        self.convs.append(pygnn.SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_layers - 1):
            self.convs.append(pygnn.SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs, emb):
        attr = self.time_graph.edge_attr

        xs = []
        x_all = x
        if self.num_layers > 1:
            for i, (edge_index, e_id, size) in enumerate(adjs):
                weight = attr[e_id].type(torch.float)
                x = x_all
                if len(list(x.shape)) < 2:
                    x = x.unsqueeze(0)
                x = self.gcn(x, edge_index, weight, emb)
                # sage
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
        else:
            # 只有1-hop的情況
            edge_index, e_id, size = adjs.edge_index, adjs.e_id, adjs.size
            x = x_all
            x = self.dropout(x)
            weight = attr[e_id].view(-1).type(torch.float)
            if len(list(x.shape)) < 2:
                x = x.unsqueeze(0)
            x = self.gcn(x, edge_index, weight)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[-1]((x, x_target), edge_index)
        xs.append(x)
        return torch.cat(xs, 0)