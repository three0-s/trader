
import torch
import torch.nn as nn
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Dueling_DQN(nn.Module):
    def __init__(self, in_features: int, 
                num_actions: int, 
                emb_dim: int, 
                n_stocks: int, 
                num_head: int, 
                num_layers: int,
                time_quantum: int):
        super(Dueling_DQN, self).__init__()
        assert emb_dim%num_head==0, "Model dimension should be multiple of 'num_head'"
        self.num_actions = num_actions

        self.L = time_quantum
        self.N = n_stocks
        self.D = emb_dim

        self.emb = StateEmbedding(in_features=in_features, out_features=emb_dim, time_quantum=time_quantum)
        self.attns = nn.Sequential(*[ResidualAttentionBlock(emb_dim, num_head, n_stocks) for _ in range(num_layers)])
        # L, NxB, D -> L, N, B, D -> B, N, LxD
        self.fc1_adv = nn.Linear(in_features=time_quantum*emb_dim, out_features=emb_dim)
        self.fc1_val = nn.Linear(in_features=time_quantum*emb_dim, out_features=emb_dim)

        self.fc2_adv = nn.Linear(in_features=emb_dim, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=emb_dim, out_features=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x:torch.Tensor):
        x = self.emb(x)
        x = self.attns(x)
        x = x.reshape(self.L, self.N, -1, self.D) # x: (L, NxB, D) -> (L, N, B, D)
        x = x.permute(2, 1, 0, 3)   # x: (L, N, B, D) -> (B, N, L, D)
        x = x.reshape(x.shape[0], x.shape[1], -1) # x: (B, N, L, D) -> (B, N, LxD)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val)
        
        x = val + adv - adv.mean(-1).unsqueeze(-1)
        return  x
    

class StateEmbedding(nn.Module):
    def __init__(self, in_features, out_features, time_quantum):
        super(StateEmbedding, self).__init__()

        self.L = time_quantum
        self.D = out_features
        self.norm = nn.LayerNorm([time_quantum, in_features])
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x:torch.Tensor):
        '''Params]
                x: torch.Tensor (B, N, L, F)
                                 B [batch size]
                                 N [# of the stocks]
                                 L [time quantum (max sequence)]
                                 F [# of the features]
            Return]
                out: torch.Tensor (L, NxB, D)
        '''
        out = self.norm(x)
        out = self.relu(self.layer(out))
        out = out.permute(2, 1, 0, 3).reshape((self.L, -1, self.D))
        return out


class ResidualAttentionBlock(nn.Module):
    '''Folked from https://github.com/openai/CLIP/blob/main/clip/model.py
    '''
    def __init__(self, d_model: int, n_head: int, n_stock: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.n_stock = n_stock
        # self.attn_list = nn.ModuleList([nn.MultiheadAttention(d_model, n_head) for _ in range(n_stock)])
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        # self.mlp_list = nn.ModuleList([nn.Sequential(OrderedDict([
        #                     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #                     ("gelu", QuickGELU()),
        #                     ("c_proj", nn.Linear(d_model * 4, d_model))
        #                 ])) for _ in range(n_stock)])
        self.mlp = nn.Sequential(OrderedDict([
                            ("c_fc", nn.Linear(d_model, d_model * 4)),
                            ("gelu", QuickGELU()),
                            ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        # return self.attn_list[i](x[:, i, ...], x[:, i, ...], x[:, i, ...], need_weights=False, attn_mask=self.attn_mask)[0]
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        '''Params]
                x: torch.Tensor (L, NxB, D)
           Return]
                out: torch.Tensor (L, NxB, D)
        '''
        # out = []
        # x = self.ln_1(x)
        # for i in range(self.n_stock):
        #     tmp = x[:, i, ...] + self.attention(x, i)
        #     tmp = tmp + self.mlp_list[i](self.ln_2(tmp))
        #     out.append(tmp.unsqueeze(1))
        # out = torch.concat(out, dim=1)
        
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

if __name__ == "__main__":
    from torchinfo import summary
    B, N, L, F, D = 32, 1, 64, 7, 256
    A = 9
    a = torch.zeros(B, N, L, F)
    model = Dueling_DQN(F, A, D, N, 8, 3, L)
    summary(model, (B, N, L, F))
    # print(model(a))