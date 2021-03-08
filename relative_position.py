class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


self.relative_position_k = RelativePosition(i, self.d_k, max_relative_position)
self.relative_position_v = RelativePosition(i, self.d_v, max_relative_position)

r_q = q.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*n_head, d_k)
r_k = self.relative_position_k(len_q, len_k)
attn_2 = torch.matmul(r_q, r_k.transpose(1, 2)).transpose(0, 1)
attn_2 = attn_2.contiguous().view(sz_b, self.n_head, len_k, len_k)

r_v = self.relative_position_v(len_q, len_v)
weight = attn.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*n_head, len_k)
weight = torch.matmul(weight, r_v)
weight = weight.transpose(0, 1).contiguous().view(sz_b, self.n_head, len_q, d_v)
