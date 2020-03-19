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
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings
