import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_Debias(nn.Module):
    def __init__(self, item_num, device, **kwargs):
        super().__init__()
        self.item_num = item_num + 1
        self.device = device
    
    def forward(self, items):
        """
            calculate the sampled weights
            Base_Debias utilizes the uniform sampling
        """
        # pos_items : B x 1  (N_p is the length of padded [sequence] positive items)
        # neg_items : B x B
        # select other training examples as negatives
        return -torch.log(self.item_num * torch.ones_like(items))


class Pop_Debias(Base_Debias):
    """
        debias the weights according to the popularity
    """
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count.shape[0], device)
        pop_count = torch.from_numpy(pop_count).to(self.device)  # check whether use the device parameter
        if mode == 1:
            pop_count = pop_count
        elif mode == 2:
            pop_count = torch.log( 1 + pop_count ) + 1e-8
        elif mode == 3:
            pop_count = pop_count ** 0.75
        else:
            raise ValueError

        pop_count = torch.cat([torch.zeros(1, device=self.device), pop_count])
        self.pop_prob = pop_count / pop_count.sum()  # other normalization strategy can be satisfied
        self.pop_prob[0] = torch.ones(1, device=self.device)  # padding values
        
    def forward(self, items):
        return torch.log(self.pop_prob[items])


class EstPop_Debias(Base_Debias):
    """
        debias the weights according to the popularity
    """
    def __init__(self, item_num, device, alpha=1e-4, **kwargs):
        super().__init__(item_num, device)
        
        self.primes = [4993, 4999, 5003, 5009, 5011]
        # self.A = torch.zeros(self.item_num, device=self.device)
        # self.B = torch.ones(self.item_num, device=self.device)
        self.A = [torch.zeros(self.primes[i], device=self.device) for i in range(len(self.primes))]
        self.B = [torch.ones(self.primes[i], device=self.device) for i in range(len(self.primes))]
        self.t = torch.zeros(1, device=self.device)
        self.alpha = alpha
    
    def forward(self, items):
        self.t += 1
        pi = []
        for i in range(len(self.A)):
            keys = items % self.primes[i]
            delta = (1 - self.alpha) * self.B[i][keys] + self.alpha * (self.t - self.A[i][keys])
            self.B[i] = self.B[i].index_put((keys,), values=delta)
            self.A[i] = self.A[i].index_put((keys,), values=self.t)
            pi.append(delta)
        # return a Bx1 tensor
        return torch.log(1 / torch.max(torch.stack(pi, dim=0), dim=0)[0])
    
    def get_pop_bias(self, items):
        return self.forward(items)

    def resample(self, score, log_prob, sample_size):
        # score : B x B
        # log_prob: B 
        sample_weight = F.softmax(score - log_prob, dim=-1)
        indices = torch.multinomial(sample_weight, sample_size, replacement=True)
        return -torch.log(self.item_num * torch.ones_like(log_prob)), indices, -torch.log(self.item_num * torch.ones_like(log_prob[indices]))


class MixNeg_Debias(Pop_Debias):
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)

    def get_pop_bias(self, items):
        return torch.log(self.pop_prob[items])

    def forward(self, items, ratio=0.5):
        return torch.log(ratio * self.pop_prob[items] + (1 - ratio) * (1.0 / self.item_num))


class ReSample_Debias(Pop_Debias):
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)

    def get_pop_bias(self, items):
        return torch.log(self.pop_prob[items])

    def resample(self, score, log_prob, sample_size):
        # score : B x B
        # log_prob: B
        sample_weight = F.softmax(score - log_prob, dim=-1)
        indices = torch.multinomial(sample_weight, sample_size, replacement=True)
        return -torch.log(self.item_num * torch.ones_like(log_prob)), indices, -torch.log(self.item_num * torch.ones_like(log_prob[indices]))

    
class BatchMixup_Debias(Pop_Debias):
    def __int__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)

    def get_pop_bias(self, items):
        return torch.log(self.pop_prob[items])

    def forward(self, items):
        return self.get_pop_bias(items)
    
    
if __name__ == '__main__':
    # item_list = [
    #     [1,2,3],
    #     [4,5,6],
    #     [2,7,8],
    #     [1,3,4],
    #     [5,6,10],
    # ]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # debias = EstPop_Debias(item_num=3, device=device, alpha=0.5)
    # for items in item_list:
    #     items = torch.tensor(items, dtype=torch.long, device=device)
    #     log_p = debias(items)
    #     print(log_p.view(1, -1).repeat(2, 1))
    # IndM = torch.randint(3, size=(3, 2), device=device)
    # print(IndM)
    # print(log_p[IndM])
    a = torch.zeros(3,3)
    b = torch.ones(3,3)
    c = torch.stack([a,b], dim=0)
    print(c)