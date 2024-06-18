import torch
import struct


def save_tensor_list(file_name, *tensor_list):
    with open(file_name, 'wb') as file:
        for tensor in tensor_list:
            for f in tensor.reshape(-1).detach().numpy():
                file.write(struct.pack('f', float(f)))


def transpose():
    a = torch.randn([3, 32, 8], requires_grad=True)
    t = torch.randn([8, 32, 3])
    b = torch.transpose(a, 0, 2)
    loss = torch.nn.MSELoss()(b, t)
    loss.backward()
    save_tensor_list("Transpose.bytes", a, t, b, a.grad)


def tanh():
    a = torch.randn([3, 32, 8], requires_grad=True)
    t = torch.randn([8, 32, 3])
    b = torch.transpose(a, 0, 2)
    c = torch.tanh(b)
    loss = torch.nn.MSELoss()(c, t)
    loss.backward()
    save_tensor_list("Tanh.bytes", a, t, c, a.grad)


def relu():
    a = torch.randn([3, 32, 8], requires_grad=True)
    t = torch.randn([8, 32, 3])
    b = torch.transpose(a, 0, 2)
    c = torch.relu(b)
    loss = torch.nn.MSELoss()(c, t)
    loss.backward()
    save_tensor_list("ReLU.bytes", a, t, c, a.grad)


def sigmoid():
    a = torch.randn([3, 32, 8], requires_grad=True)
    t = torch.randn([8, 32, 3])
    b = torch.transpose(a, 0, 2)
    c = torch.sigmoid(b)
    loss = torch.nn.MSELoss()(c, t)
    loss.backward()
    save_tensor_list("Sigmoid.bytes", a, t, c, a.grad)


def add():
    a = torch.randn([3, 32, 8], requires_grad=True)
    b = torch.randn([8, 32, 3], requires_grad=True)
    c = torch.transpose(b, 0, 2)
    t = torch.randn([3, 32, 8])
    d = a + c
    loss = torch.nn.MSELoss()(d, t)
    loss.backward()
    save_tensor_list("Add.bytes", a, b, t, d, a.grad, b.grad)


def mul():
    a = torch.randn([3, 32, 8], requires_grad=True)
    b = torch.randn([8, 32, 3], requires_grad=True)
    c = torch.transpose(b, 0, 2)
    t = torch.randn([3, 32, 8])
    d = a * c
    loss = torch.nn.MSELoss()(d, t)
    loss.backward()
    save_tensor_list("Mul.bytes", a, b, t, d, a.grad, b.grad)


def res():
    a = torch.randn([3, 32, 8], requires_grad=True)
    b = torch.randn([3, 32, 8], requires_grad=True)
    c = torch.sigmoid(a * b) + torch.relu(a + b)
    t = torch.randn([3, 32, 8])
    loss = torch.nn.MSELoss()(c, t)
    loss.backward()
    save_tensor_list("Res.bytes", a, b, t, c, a.grad, b.grad)


def linear():
    a = torch.randn([3, 32, 8], requires_grad=True)
    b = torch.transpose(a, 1, 2)
    linear = torch.nn.Linear(32, 16)
    c = linear(b)
    t = torch.randn([3, 8, 16])
    loss = torch.nn.MSELoss()(c, t)
    loss.backward()
    save_tensor_list("Linear.bytes", a, t, c, a.grad)


def mat_mul1():
    a = torch.randn([32, 8], requires_grad=True)
    b = torch.randn([32, 8], requires_grad=True)
    c = torch.transpose(a, 0, 1)
    t = torch.randn([8, 8])
    d = torch.matmul(c, b)
    loss = torch.nn.MSELoss()(d, t)
    loss.backward()
    save_tensor_list("MatMul1.bytes", a, b, t, d, a.grad, b.grad)


def mat_mul2():
    a = torch.randn([32, 8], requires_grad=True)
    b = torch.randn([32, 8], requires_grad=True)
    c = torch.transpose(b, 0, 1)
    t = torch.randn([32, 32])
    d = torch.matmul(a, c)
    loss = torch.nn.MSELoss()(d, t)
    loss.backward()
    save_tensor_list("MatMul2.bytes", a, b, t, d, a.grad, b.grad)



if __name__ == "__main__":
    mat_mul1()
    mat_mul2()
