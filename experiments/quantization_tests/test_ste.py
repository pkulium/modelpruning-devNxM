import torch

from admm_ds.layers import FakeQuantizedFullyConnectedLayer


class SimpleModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()

        l0_parameter = torch.nn.parameter.Parameter(data=torch.rand(hidden_size, input_size))
        self.l0 = FakeQuantizedFullyConnectedLayer(l0_parameter)

        l1_parameter = torch.nn.parameter.Parameter(data=torch.rand(output_size, hidden_size))
        self.l1 = FakeQuantizedFullyConnectedLayer(l1_parameter)

    def forward(self, input):
        intermediate = torch.nn.functional.relu(self.l0(input))
        return self.l1(intermediate)


batch_size = 4
in_size = 16
hidden_size = 32
out_size = 8

model = SimpleModel(in_size, hidden_size, out_size)
optimizer = torch.optim.Adam(model.parameters())

model.train()
optimizer.zero_grad()

# Should be None
for name, param in model.named_parameters():
    print(name)
    print(param.grad)

in_tensor = torch.rand(batch_size, in_size)
target = torch.rand(batch_size, out_size)
output = model(in_tensor)
loss = torch.linalg.norm(output - target)
loss.backward()

# Should have values
for name, param in model.named_parameters():
    print(name)
    print(param.grad)

optimizer.step()
