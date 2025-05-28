import torch
import torch.nn as nn

class Sinkhorn_sparse_rpcl(nn.Module):
    def __init__(self, max_iter=10, epsilon=50) -> None:
        super(Sinkhorn_sparse_rpcl, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        
    # @torch.no_grad()
    # def forward(self, sims):
    #     size = (sims.size(0), sims.size(1))
        
    #     num_row, num_col = sims.shape
    #     device = sims.device

    #     sims = torch.exp(sims * self.epsilon)

    #     if num_row >= num_col:
    #         sims = sims.T
        
    #     for k in range(self.max_iter):
    #         sims = sims / torch.sum(sims, dim=1, keepdim=True)
    #         sims = sims.transpose(0, 1)
        
    #     row = torch.tensor([i for i in range(sims.size(0))]).to(device)
        
    #     cols_10 = []
    #     if not self.training:
    #         s = sims.T if num_row >= num_col else sims
    #         for slice in range(s.size(0) // 512 + 1):
    #             s_slice = s[slice * 512: (slice+1) * 512].cpu()
    #             col_10 = torch.argsort(-s_slice, dim=-1)[:, :10]
    #             cols_10.append(col_10)
    #         cols_10 = torch.cat(cols_10, dim=0).to(device)
        
    #     # col = self.get_rid_of_duplicate_multi_cross(sims)
        
    #     col = torch.argmax(sims, dim=-1).to(device)   

    #     assert len(row) == len(col), 'the shape of the pseudo label which generated from sinkhorn is wrong.'
    #     if num_row >= num_col:
    #         indices = torch.stack((col, row), dim=0).to(device)   
    #     else:
    #         indices = torch.stack((row, col), dim=0).to(device)     
    #     values = torch.ones(len(row)).to(device)
    #     return None, torch.sparse_coo_tensor(indices=indices, values=values, size=size, device=device), cols_10 if not self.training else None

    @torch.no_grad()
    def forward(self, sims):
        size = (sims.size(0), sims.size(1))
        
        num_row, num_col = sims.shape
        device = sims.device

        sims = torch.exp(sims * self.epsilon)

        if num_row >= num_col:
            sims = sims.T
        
        for k in range(self.max_iter):
            sims = sims / torch.sum(sims, dim=1, keepdim=True)
            sims = sims.transpose(0, 1)
        
        row = torch.tensor([i for i in range(sims.size(0))]).to(device)
        
        cols_10 = []
        if not self.training:
            s = sims.T if num_row >= num_col else sims
            for slice in range(s.size(0) // 512 + 1):
                s_slice = s[slice * 512: (slice+1) * 512].cpu()
                col_10 = torch.argsort(-s_slice, dim=-1)[:, :10]
                cols_10.append(col_10)
            cols_10 = torch.cat(cols_10, dim=0).to(device)
        
        # col = self.get_rid_of_duplicate_cross(s)
        col = self.get_rid_of_duplicate_multi_cross(sims)
        # col = self.binary(s)
        # rival = self.get_rival_max_except_label(sims, col, num_row, num_col)
        # #sinkhorn 原本
        # col = torch.argmax(sims, dim=-1).to(device)   

        assert len(row) == len(col), 'the shape of the pseudo label which generated from sinkhorn is wrong.'
        if num_row >= num_col:
            indices = torch.stack((col, row), dim=0).to(device)   
        else:
            indices = torch.stack((row, col), dim=0).to(device)     
        values = torch.ones(len(row)).to(device)
        return None, torch.sparse_coo_tensor(indices=indices, values=values, size=size, device=device), cols_10 if not self.training else None


    def binary(sel, sims):
        col = torch.full((sims.shape[0],), -1).to(sims.device)
        for _ in range(sims.shape[0]):
            index = sims.argmax()
            rowindex = torch.div(index, sims.shape[1], rounding_mode="floor")
            colindex = index % sims.shape[1]
            col[rowindex] = colindex
            sims[rowindex, :] = 0
            sims[:, colindex] = 0
        return col

    def get_rid_of_duplicate_cross(self, sims):
        r"""sims.shape[0] < sims.shape[1]
        """
        col = torch.full((sims.shape[0], ), -1).to(sims.device)
        values, index = torch.topk(sims, k=2, dim=-1)
        conf_col = (values[:, 0] - values[:, 1]).to(sims.device) # confidence = top1 - top2
        row = torch.tensor([i for i in range(sims.shape[0])]).to(sims.device)
        indices = torch.stack((row, index[:, 0]), dim=0).to(sims.device)
        
        tmp1 = torch.sparse_coo_tensor(indices=indices, values=torch.ones(sims.shape[0]).to(sims.device), \
                                       size=[sims.size(0), sims.size(1)], device=sims.device)
        conf_tmp1 = torch.sparse_coo_tensor(indices=indices, values=conf_col, \
                                       size=[sims.size(0), sims.size(1)], device=sims.device)
        
        
        values, index = torch.topk(sims, k=2, dim=0)
        conf_row = (values[0] - values[1]).to(sims.device)
        row = torch.tensor([i for i in range(sims.shape[1])]).to(sims.device)
        
        indices = torch.stack((index[0], row), dim=0).to(sims.device)

        tmp2 = torch.sparse_coo_tensor(indices=indices, values=torch.ones(sims.shape[1]).to(sims.device), \
                                       size=[sims.size(0),sims.size(1)], device=sims.device)
        conf_tmp2 = torch.sparse_coo_tensor(indices=indices, values=conf_row, \
                                        size=[sims.size(0), sims.size(1)], device=sims.device)
        

        # select the maximize index that in src <--> tgt.
        indices = (tmp1+tmp2)._indices().T # nx2
        values = (tmp1+tmp2)._values()
        confi = (conf_tmp1+conf_tmp2)._values()
        index = torch.where(values == 2)[0]

        
        col[indices[index][:, 0]] = indices[index][:, 1]
        mask_row = torch.isin(indices[:, 0].clone(), indices[index][:, 0].clone())
        mask_col = torch.isin(indices[:, 1].clone(), indices[index][:, 1].clone())
        mask = ~(mask_row + mask_col)
        indices = indices[mask] # 第一列src, 第二列tgt. nx2
        values = values[mask]
        confi = confi[mask]

        # select other "argmaxs" according to the confi
        while len(confi) !=0: 
            index = torch.argmax(confi)
            col[indices[index][0]] = indices[index][1]
            mask_row = torch.isin(indices.T[0].clone(), indices[index][0].clone())
            mask_col = torch.isin(indices.T[1].clone(), indices[index][1].clone())
            indices = indices[~(mask_row+mask_col)]
            values = values[~(mask_row + mask_col)]
            confi = confi[~(mask_row + mask_col)]

        # select the not-assigned src and tgt through col.
        mask_row = torch.where(col== -1)[0]
        mask_col = torch.isin(torch.arange(sims.shape[1]).to(sims.device).clone(), col.clone())
        
        idx = (~mask_col).cumsum(dim=0) -1
        
        col_tmp = self.binary(sims[mask_row].T[~mask_col].T)
        
        for i in range(len(col_tmp)):
            col[mask_row[i]] = torch.where(idx == col_tmp[i])[0][0]
        return col

    def get_rid_of_duplicate_multi_cross(self, sims):
        s = sims
        col = torch.full((sims.shape[0], ), -1).to(sims.device)
        rest_col = torch.arange(0, sims.shape[1]).to(sims.device)
        rest_row = torch.arange(0, sims.shape[0]).to(sims.device)
        iter_num = 0
        while s.shape[0] > 1:
            iter_num += 1
            col_tmp = torch.full((s.shape[0], ), -1).to(sims.device)

            values, index = torch.topk(s, k=2, dim=-1)
            conf_col = (values[:, 0] - values[:, 1]).to(sims.device)
            row = torch.tensor([i for i in range(s.shape[0])]).to(sims.device)
            indices = torch.stack((row, index[:, 0]), dim=0).to(sims.device)

            tmp1 = torch.sparse_coo_tensor(indices=indices, values=torch.ones(s.shape[0]).to(sims.device), \
                                        size=[s.size(0), s.size(1)], device=sims.device)
            conf_tmp1 = torch.sparse_coo_tensor(indices=indices, values=conf_col, \
                                        size=[s.size(0), s.size(1)], device=sims.device)

            values, index = torch.topk(s, k=2, dim=0)
            conf_row = (values[0] - values[1]).to(sims.device)
            row = torch.tensor([i for i in range(s.shape[1])]).to(sims.device)
            indices = torch.stack((index[0], row), dim=0).to(sims.device)

            tmp2 = torch.sparse_coo_tensor(indices=indices, values=torch.ones(s.shape[1]).to(sims.device), \
                                        size=[s.size(0),s.size(1)], device=sims.device)
            conf_tmp2 = torch.sparse_coo_tensor(indices=indices, values=conf_row, \
                                        size=[s.size(0), s.size(1)], device=sims.device)
        
            # select the maximize index that in src <--> tgt.
            indices = (tmp1 + tmp2)._indices().T # nx2
            values = (tmp1 + tmp2)._values()
            confi = (conf_tmp1 + conf_tmp2)._values()
            index = torch.where(values == 2)[0]

            col_tmp[indices[index][:, 0]] = indices[index][:, 1]
            
            # mask -> smaller
            mask_row = torch.isin(indices[:, 0].clone(), indices[index][:, 0].clone())
            mask_col = torch.isin(indices[:, 1].clone(), indices[index][:, 1].clone())
            mask = ~(mask_row + mask_col)
            indices = indices[mask] # 第一列src, 第二列tgt. nx2
            values = values[mask]
            confi = confi[mask]

            # select other "argmaxs" according to the confi
            while len(confi) !=0: 
                index = torch.argmax(confi)
                col_tmp[indices[index][0]] = indices[index][1]
                mask_row = torch.isin(indices.T[0].clone(), indices[index][0].clone())
                mask_col = torch.isin(indices.T[1].clone(), indices[index][1].clone())
                indices = indices[~(mask_row+mask_col)]
                values = values[~(mask_row + mask_col)]
                confi = confi[~(mask_row + mask_col)]
            
            # map from col_tmp to col, row_tmp tp row
            # tmp = torch.gather(input=rest_col, dim=0, index=col_tmp)
            # col = torch.scatter(input=col, dim=0, index=rest_row, src=tmp)

            # for i in torch.where(col_tmp > -1)[0]:
            #     col[rest_row[i]] = rest_col[col_tmp[i]]
            mask = col_tmp > -1
            indices = rest_row[mask]
            col = col.scatter_(dim=0, index=indices, src=rest_col[col_tmp[mask]])
            

            # mask-->smaller
            # select the not-assigned src and tgt through col.
            rest_row = torch.where(col == -1)[0]
            rest_col = ~(torch.isin(torch.arange(sims.shape[1]).to(sims.device).clone(), col.clone()))
            rest_col = torch.nonzero(rest_col).squeeze(1)
        
            s = sims[rest_row].T[rest_col].T
            
        if s.shape[0] == 1:
            col[rest_row] = rest_col[torch.argmax(s, dim=-1)]
        return col

    def get_rival_max_except_label(self, sims, col, num_row, num_col):
        r"""
        select the rival as the maximize but not the label.
        """
        col_rival = torch.argmax(sims, dim=-1)
        index = torch.where(~torch.eq(col_rival, col))[0]
        if num_row > num_col:
            rival = torch.stack((col_rival[index], index), dim=0).to(sims.device)
        else:
            rival = torch.stack((index, col_rival[index]), dim=0).to(sims.device)
        return rival    
        
       
class Sinkhorn_sparse(nn.Module):
    def __init__(self, max_iter=10) -> None:
        super(Sinkhorn_sparse, self).__init__()
        self.max_iter = max_iter
    
    def forward(self, sims, batch_size=256):
        size = (sims.size(0), sims.size(1))
        s = torch.exp(sims*50)
        num_row, num_col = sims.shape
        device = sims.device
        
        if num_row >= num_col:
            s = s.T  
        
        for k in range(self.max_iter):
            row_slice = []
            for slice in range(s.size(0) // batch_size + 1):
                s_slice = s[slice * batch_size: (slice+1) * batch_size]
                s_slice = s_slice / torch.sum(s_slice, dim=1, keepdim=True)
                row_slice.append(s_slice)
            s = torch.cat(row_slice, dim=0).T.to(device)

            
        # assert s.shape == sims.shape, 'the num of iteration must be even.'
        row = torch.tensor([i for i in range(s.size(0))]).to(device)
        col = torch.argmax(s, dim=-1).to(device)

        assert len(row) == len(col), 'the shape of the pseudo label which generated from sinkhorn is wrong.'
        if num_row >= num_col:
            indices = torch.stack((col, row), dim=0).to(device)   
        else:
            indices = torch.stack((row, col), dim=0).to(device)     
        values = torch.ones(len(row)).to(device)
        cols_10 = []
        if not self.training:
            if num_row >= num_col:
                s = s.T
            for slice in range(s.size(0) // batch_size + 1):
                s_slice = s[slice * batch_size: (slice+1) * batch_size].cpu()
                col_10 = torch.argsort(-s_slice, dim=-1)[:, :10]
                cols_10.append(col_10)
            cols_10 = torch.cat(cols_10, dim=0).to(device)
        return s, torch.sparse_coo_tensor(indices=indices, values=values, size=size, device=device), cols_10 if not self.training else None
    

