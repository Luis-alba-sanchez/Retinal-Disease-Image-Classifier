import torch

print("torch.version.cuda: ", torch.version.cuda)
print("torch.cuda.is_available(): ", torch.cuda.is_available())
print("torch.cuda.device_count(): ", torch.cuda.device_count())
print("torch.cuda.current_device(): ", torch.cuda.current_device())
print("torch.cuda.get_arch_list(): ", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"- torch.cuda.get_device_name({i}): ", torch.cuda.get_device_name(i))
        print(f"- torch.cuda.get_device_properties({i}): ", torch.cuda.get_device_properties(i))
