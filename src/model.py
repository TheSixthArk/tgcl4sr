# Codes are based on GCL4SR
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time_encoder import GlobalGNN
from torch_geometric.loader.neighbor_sampler import NeighborSampler

class MainModel(nn.Module):
    def __init__(self, args, bi_graph, time_graph):
        super(MainModel, self).__init__()
        self.args = args
        self.device = args.device
        self.hidden_channel = args.hidden_channel
        # 嵌入矩阵
        self.user_emb = nn.Embedding(args.user_num, args.hidden_channel)
        self.item_emb = nn.Embedding(args.item_num, args.hidden_channel)
        self.seq_pos_emb = nn.Embedding(args.max_length, args.hidden_channel)
        self.time_pos_emb = nn.Embedding(args.max_ts, args.hidden_channel)
        # 用于编码time_graph的TAT_model
        self.tat = GlobalGNN(args, time_graph)
        self.tat_dp = nn.Dropout(args.seq_dropout)
        # 用于编码交互序列的transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(args.hidden_channel, args.seq_head, 4*args.hidden_channel, args.seq_dropout)
        self.seq_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.seq_layers)
        self.LayerNorm = nn.LayerNorm(args.hidden_channel, eps=1e-12)
        self.seq_dp = nn.Dropout(args.seq_dropout)
        # 对序列特征编码进行融合
        self.seq_fuse = nn.Linear(3 * args.hidden_channel, args.hidden_channel, bias=False)
        # 对用户特征编码结果的注意力机制
        self.w_1 = nn.Parameter(torch.Tensor(2 * args.hidden_channel, args.hidden_channel))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_channel, 1))
        self.linear_1 = nn.Linear(args.hidden_channel, args.hidden_channel)
        self.linear_2 = nn.Linear(args.hidden_channel, args.hidden_channel, bias=False)
        # 配置损失函数
        self.pred_loss = nn.CrossEntropyLoss()
        # mmd参数
        self.w_g = nn.Linear(args.hidden_channel, 1)
        self.w_e = nn.Linear(args.hidden_channel, 1)
        # 初始化参数
        self.reset_parameter()
        # 配置优化器
        self.optim = optim.Adam(self.parameters(), lr=args.lr)

    def reset_parameter(self):
        nn.init.normal_(self.seq_fuse.weight, std=0.1)
        nn.init.normal_(self.linear_1.weight, std=0.1)
        nn.init.normal_(self.linear_2.weight.data, std=0.1)
        nn.init.normal_(self.w_1.data, std=0.1)
        nn.init.normal_(self.w_2.data, std=0.1)
        nn.init.normal_(self.user_emb.weight.data, std=0.1)
        nn.init.normal_(self.user_emb.weight.data, std=0.1)
        nn.init.normal_(self.item_emb.weight.data, std=0.1)
        nn.init.normal_(self.seq_pos_emb.weight.data, std=0.1)
        nn.init.normal_(self.time_pos_emb.weight.data, std=0.1)
        nn.init.normal_(self.w_e.weight, std=0.1)
        nn.init.normal_(self.w_g.weight, std=0.1)

    def genMask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1.0e4).masked_fill(mask == 1, float(0.0))
        return mask

    def seq_attention(self, mask, hidden):
        batch_size, length = hidden.shape[0], self.args.max_length
        seq_hidden = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, length, 1)

        pos_emb = self.seq_pos_emb.weight[:length].unsqueeze(0).repeat(batch_size, 1, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)

        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output

    def forward(self, user, seq, tts):
        item_id = seq.flatten()
        tg_hidden_a, tg_hidden_a_disrupted = self.tat_encode(item_id, sigma=self.args.sigma)
        tg_hidden_b, tg_hidden_b_disrupted = self.tat_encode(item_id, sigma=self.args.sigma)
        tg_hidden_a = tg_hidden_a.view(-1, self.args.max_length, self.hidden_channel)
        tg_hidden_b = tg_hidden_b.view(-1, self.args.max_length, self.hidden_channel)
        tg_hidden_a_disrupted = tg_hidden_a_disrupted.view(-1, self.args.max_length, self.hidden_channel)
        tg_hidden_b_disrupted = tg_hidden_b_disrupted.view(-1, self.args.max_length, self.hidden_channel)

        mask = (seq == 0).float().unsqueeze(-1)
        mask = 1.0 - mask
        attn_mask = self.genMask(self.args.max_length).to(seq.device)
        padding_mask = (seq == 0)

        seq_hidden_local = self.item_emb(seq)
        seq_hidden_local = seq_hidden_local + self.time_pos_emb(tts)

        seq_hidden_local = self.LayerNorm(seq_hidden_local) #
        seq_hidden_local = self.seq_dp(seq_hidden_local) #

        seq_hidden_local = seq_hidden_local.permute(1, 0, 2)
        seq_hidden_encode = self.seq_encoder(seq_hidden_local, mask=attn_mask, src_key_padding_mask=padding_mask)
        seq_hidden_encode = seq_hidden_encode.permute(1, 0, 2)

        user_seq_a = tg_hidden_a
        user_seq_b = tg_hidden_b
        user_hidden = self.seq_fuse(torch.cat([seq_hidden_encode, user_seq_a, user_seq_b], -1))
        return seq_hidden_encode, user_hidden, user_seq_a, user_seq_b, tg_hidden_a, tg_hidden_b, tg_hidden_a_disrupted, tg_hidden_b_disrupted, mask

    def tat_encode(self, items, sigma=0.1):
        subgraph_loaders = NeighborSampler(self.tat.time_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])
        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_emb(n_idxs).squeeze()
        g_hidden = self.tat(s_nodes, g_adjs, self.user_emb.weight)
        # 增加时间扰动的对比学习
        org_time, index = self.tat.time_graph.edge_attr[:, :2], self.tat.time_graph.edge_attr[:, 2].long().view(-1, 1)
        disrupt_mask = torch.rand(org_time.shape)
        disrupt = torch.randn(org_time.shape).masked_fill(disrupt_mask < 0.5, 0.0).to(org_time.device) * sigma
        self.tat.time_graph.edge_attr = torch.cat([org_time + disrupt, index], dim=1)
        g_hidden_disrupted = self.tat(s_nodes, g_adjs, self.user_emb.weight)
        self.tat.time_graph.edge_attr = torch.cat([org_time, index], dim=1)
        return g_hidden, g_hidden_disrupted

    def contrast(self, hidden1, hidden2, hidden_norm=True, temperature=0.5):
        batch_size = hidden1.shape[0]
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.cat([hidden1, hidden2], dim=0)
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            hidden1, hidden2 = torch.split(hidden, batch_size, dim=0)
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden1.device).long()
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size).astype(np.int64)).to(hidden1.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss

    def loss_function(self, user, seq, target, tts):
        seq_hidden_encode, user_hidden, user_seq_a, user_seq_b, tg_hidden_a, tg_hidden_b, tg_hidden_a_disrupted, tg_hidden_b_disrupted, mask = self.forward(user, seq, tts)
        user_hidden = self.seq_attention(mask, user_hidden)
        user_hidden = self.tat_dp(user_hidden) #
        item_emb = self.item_emb.weight[:self.args.item_num]
        # 预测损失
        logits = torch.matmul(user_hidden, item_emb.transpose(0, 1))
        pred_loss = self.pred_loss(logits, target)
        # 对比损失
        sum_a = torch.sum(tg_hidden_a * mask, dim=1) / torch.sum(mask.float(), 1)
        sum_b = torch.sum(tg_hidden_b * mask, dim=1) / torch.sum(mask.float(), 1)
        sum_a_d = torch.sum(tg_hidden_a_disrupted * mask, dim=1) / torch.sum(mask.float(), 1)
        sum_b_d = torch.sum(tg_hidden_b_disrupted * mask, dim=1) / torch.sum(mask.float(), 1)
        cl_loss = self.contrast(sum_a, sum_b) + 0.5 * (self.contrast(sum_a, sum_a_d, temperature=self.args.temp) + self.contrast(sum_b, sum_b_d, temperature=self.args.temp))
        # MMD损失
        seq_hidden_local = self.w_e(self.item_emb(seq)).squeeze().unsqueeze(0)
        user_seq_a = self.w_g(user_seq_a).squeeze()
        user_seq_b = self.w_g(user_seq_b).squeeze()
        mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)

        return pred_loss, cl_loss, mmd_loss

    def predict(self, user, seq, tts):
        _, user_hidden, _, _, _, _, _, _,mask = self.forward(user, seq, tts)
        user_hidden = self.seq_attention(mask, user_hidden)
        item_emb = self.item_emb.weight[:self.args.item_num]
        pred = torch.sigmoid(torch.matmul(user_hidden, item_emb.transpose(0, 1)))
        return pred

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        source = source.view(-1, self.args.max_length)
        target = target.view(-1, self.args.max_length)
        batch_size = int(source.size()[0])
        loss_all = []
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)
        loss_all.append(loss)
        return sum(loss_all) / len(loss_all)
