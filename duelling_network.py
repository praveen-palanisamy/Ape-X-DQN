import torch

class DuellingDQN(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DuellingDQN, self).__init__()
        self.input_shape = state_shape
        self.action_dim = action_dim
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(state_shape[0], 64, 8, stride=4),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 4, stride=2),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1),
                                          torch.nn.ReLU())
        self.value_stream_layer = torch.nn.Sequential(torch.nn.Linear( 64 * 7 * 7, 512),
                                                      torch.nn.ReLU())
        self.advantage_stream_layer = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                                          torch.nn.ReLU())
        self.value = torch.nn.Linear(512, 1)
        self.advantage = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        #assert x.shape == self.input_shape, "Input shape should be:" + str(self.input_shape) + "Got:" + str(x.shape)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = x.view(-1, 64 * 7 * 7)
        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1/self.action_dim) * advantage.sum() )
        return value, advantage, action_value
