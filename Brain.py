class Network(nn.Module):
    def __init__(self, ):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x);
        x = self.conv3(x); x = x.view(x.size(0), -1)
        x = self.fc1(x);   return self.out(x)

class DQNBrain(object):
    def save(self):
        print("Save DQN net parameters.")
        torch.save(self.net.state_dict(), "DQN.pth")
        
    def load(self):
        if os.path.exists("DQN.pth"):
            print("Load DQN net parameters.")
            self.net.load_state_dict(torch.load("DQN.pth") )
            self.tnet.load_state_dict(torch.load("DQN.pth") )
            
    def __init__(self, state_size, action_dim):
        #初始化值函数网络和目标网络
        self.net, self.tnet = Network(state_size, action_dim), Network(state_size, action_dim)
        self.load()
        self.func = nn.MSELoss()
        lrc = 1e-3
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lrc)
    
    def target_value(self,state,action):
        return self.tnet(s,a)            #目标网络提供Q'(s',a')

    def predict(self,state,action):      #预测值函数Q(s,a)
        return self.net(s,a)

    def target_policy(self, currentstate):
        QValue = self.(currentstate)  # 算出所有Q(s,a)(a属于A)的值
        action = np.zeros(self.num_actions)
        action_index = np.argmax(QValue.detach().numpy())
        action[action_index] = 1

        return action

    def explore_policy(self, currentstate):
        '''ε贪心'''
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.num_actions)
            action[action_index] = 1
            return action
        else:
            return target_policy(currentstate)

    def train(self, pre, lab):
        self.optimizer.zero_grad()
        loss = self.func(pre, lab)
        loss.backward()
        self.optimizer.step()
        return loss