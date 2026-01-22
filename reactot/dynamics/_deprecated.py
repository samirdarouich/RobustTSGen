        # else:
        #     # Crop the all connected edge_index to only those within cutoff distance
        #     # This avoids overflow in EGNN due to sum aggregation over too many neighbors.
        #     # This is important for larger non-periodic systems. Furthermore, add a max
        #     # number of neighbors per atom. This is also important for stability.
        #     if self.crop_edge_index_with_cutoff:
        #         cutoff = self.model.cutoff
        #         i, j = edge_index
        #         dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        #         inner_subgraph_mask = torch.zeros(edge_index.size(1), 1, device=dist.device)
        #         inner_subgraph_mask[torch.where(dist < cutoff)[0]] = 1
                
        #         i_, j_ = edge_index.T[torch.where(inner_subgraph_mask > 0)[0]].T
        #         atom_distance_sqr = torch.sum((pos[i_] - pos[j_]) ** 2, dim=1)
        #         atom_distance_sqr = atom_distance_sqr.view(-1)
                
        #         mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        #             natoms=combined_mask[0:frag_index[1]].bincount().repeat(3),
        #             index=i_,
        #             atom_distance=atom_distance_sqr,
        #             max_num_neighbors_threshold=50,
        #         )
                
        #         index1 = torch.masked_select(i_, mask_num_neighbors)
        #         index2 = torch.masked_select(j_, mask_num_neighbors)

        #         assert torch.isin(index2[index1 == 0], j_[i_ == 0]).all(), "New neighbors of atom 0 should be a subset of old neighbors. Something went wrong with max neighbor masking."
                
        #         edge_index = torch.stack((index1, index2))

        #         if subgraph_mask is not None:
        #             subgraph_mask = subgraph_mask[torch.where(inner_subgraph_mask > 0)[0]]
        #             subgraph_mask = torch.masked_select(subgraph_mask, mask_num_neighbors)