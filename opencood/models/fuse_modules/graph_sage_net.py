import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class FlattenCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(FlattenCNN, self).__init__()
        # Example CNN architecture, modify as needed
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GraphSageNet(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(GraphSageNet, self).__init__()
        self.flatten_cnn = FlattenCNN(input_channels, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, mask):
        # Flatten BEV features
        batch_size, num_cars, c, h, w = x.size()
        x = self.flatten_cnn(x.view(-1, c, h, w))  # Output shape: [batch_size * num_cars, hidden_dim]
        x = x.reshape(batch_size, num_cars, -1)  # Output shape: [batch_size, num_cars, hidden_dim]

        # Apply mask
        x = x * mask.unsqueeze(-1)  # Mask out the features of absent cars

        # Construct edge index.
        edge_index = self.create_bidirectional_fully_conncted_edges(num_cars).to(x.device)
        edge_index = edge_index.repeat(batch_size, 1, 1)  # Output shape: [batch_size, 2, num_cars * num_cars]

        # Apply GraphSAGE layers for each graph in the batch
        out = []
        for i in range(batch_size):
            data_i = Data(x=x[i], edge_index=edge_index[i])
            xi = F.relu(self.sage1(data_i.x, data_i.edge_index))
            xi = self.sage2(xi, data_i.edge_index)
            out.append(xi)

        res = torch.stack(out)

        return res

    @staticmethod
    def create_bidirectional_fully_conncted_edges(num_cars):
        edges = []
        for i in range(num_cars):
            for j in range(num_cars):
                if i != j:  # Example: connect every node with every other node
                    edges.append((i, j))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


if __name__ == "__main__":
    # Example Usage
    num_cars = 5
    batch_size = 4
    input_channels = 3
    hidden_dim = 64
    output_dim = 10

    bev_input = torch.randn(batch_size, num_cars, input_channels, 64, 64)
    mask = torch.randint(0, 2, (batch_size, num_cars))

    model = GraphSageNet(input_channels, hidden_dim, output_dim)

    # Process each graph in the batch
    output = model(bev_input, mask)  # batch_size, num_cars, output_dim

    print(output)
