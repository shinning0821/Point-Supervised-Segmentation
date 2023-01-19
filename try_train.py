import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_arch_list())
a = torch.zeros(1).cuda()