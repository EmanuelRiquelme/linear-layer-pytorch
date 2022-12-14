import torch
import torch.nn as nn
class linear(nn.Module):
    def __init__(self,input_size = 28**2,num_clases = 3):
        super(linear, self).__init__()
        self.hidden_layer = nn.Linear(input_size,num_clases)
        self.size = input_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,img):
        img = self.hidden_layer(img)
        img = self.relu(img)
        return self.softmax(img)

if __name__ == '__main__':
    model = linear()
    img = torch.rand([5,28**2])
    labels = ['positive','neutral','negative']
    output = torch.argmax(model(img),dim=1)
    print([labels[index] for index in output])
