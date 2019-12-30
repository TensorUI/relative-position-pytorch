class relative_position(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super(relative_position, self).__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        # 将序列distance_mat小于-max_relative_position或者大于max_relative_position的值都设置为-+max_relative_position
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


r_k = self.relative_position(Q.size()[1], K_.size()[1])
outputs = outputs + torch.bmm(Q.permute(1, 0, 2), r_k.permute(0, 2, 1)).permute(1, 0, 2)
#outputs  = Q*K^T
                                                      
r_v = self.relative_position(Q.size()[1], V.size()[1])
outputs2 = outputs + torch.bmm(weights.permute(1, 0, 2), r_v).permute(1, 0, 2)
#the size of Q,K,V is [heads*batch,length,dim//heads]
#theoutputs is origin self-attention, weights is the self-attention not multiplied by V
#the implementation is excerpted from my model, so the variable name may be confused:-)
