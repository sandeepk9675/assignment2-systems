import torch

# 1. Initialization and accumulation in float32
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

# 2. Initialization and accumulation in float16
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

# 3. Mixed precision accumulation: initializing in float32, accumulating in float16
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

# 4. Mixed precision accumulation: initializing in float32, accumulating in float16
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

# Example 3 and 4 are same, because in case before addition, the float16 tensor is cast to float32