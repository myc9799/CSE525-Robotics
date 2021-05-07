import torch.nn as nn
from torch.optim import Adam
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from OU_Noise import *
from Memory import *
from Utils import *

hidden1 = 400
hidden2 = 300
rate = 0.001
prate = 0.0001
discount = 0.99
bsize = 64
rmsize = 6000000
tau = 0.001
ou_theta = 0.15
ou_sigma = 0.2
ou_mu = 0.0
init_w = 0.003
epsilon = 50000
def generate_gaussian(mean, variance):
    dist = []
    for i in range(mean.shape[0]):
        m_temp = mean[i][0].detach().numpy()
        v_temp = variance[i][0].detach().numpy()
        dist_temp = list(np.array((np.random.normal(m_temp, v_temp, 10000))))
        dist.append(dist_temp)
    dist = np.array(dist)
    return to_tensor(dist)

def caculate_loss(dist1, dist2):
    diff = []
    for i in range(dist1.shape[0]):
        dist_1 = dist1[i].detach().numpy()
        dist_2 = dist2[i].detach().numpy()
        diff_temp = stats.wasserstein_distance(dist_1, dist_2)
        diff.append(diff_temp)
    diff = np.array(diff)
    return to_tensor(diff).unsqueeze(-1)

class DDPG(object):
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': hidden1,
            'hidden2': hidden2,
            'init_w': init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_target = Actor(self.nb_states, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=prate)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_target = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=rate)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=rmsize)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=ou_theta, mu=ou_mu,
                                                       sigma=ou_sigma)

        # Hyper-parameters
        self.batch_size = bsize
        self.tau = tau
        self.discount = discount
        self.depsilon = 1.0 / epsilon

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        self.alpha = np.random.uniform(0.0, 1.0, size=1)[0]
        self.loss = nn.BCEWithLogitsLoss()
        self.value_loss = 0.0
        self.policy_loss = 0.0

    def update_alpha(self):
        self.alpha = np.random.uniform(0.0, 1.0, size=1)[0]

    def update_policy(self):
        # Sample batch

        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        temp = np.zeros((64, 1)) + self.alpha
        next_state_batch_a = np.hstack((next_state_batch, temp))
        next_q_values, next_variance_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch_a, volatile=True)),
        ])

        next_q_values.volatile = False
        next_variance_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values


        target_variance_batch = to_tensor(reward_batch) * to_tensor(reward_batch) + \
                        2 * self.discount * to_tensor(reward_batch)* to_tensor(terminal_batch.astype(np.float)) *\
                        next_q_values + self.discount * self.discount * next_variance_values * to_tensor(terminal_batch.astype(np.float))+ \
                        self.discount * self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values * next_q_values - \
                        target_q_batch * target_q_batch


        # Critic update
        self.critic.zero_grad()

        q_batch , variance_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])


        dist_current = generate_gaussian(q_batch, variance_batch)
        dist_future = generate_gaussian(target_q_batch, target_variance_batch)

        value_loss = caculate_loss(dist_current, dist_future)
        value_loss = Variable(value_loss, requires_grad=True)

        value_loss = value_loss.mean()
        self.value_loss = value_loss
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        std_term = norm.pdf(self.alpha) / norm.cdf(self.alpha)

        temp = np.zeros((64, 1)) + self.alpha
        state_batch_a = np.hstack((state_batch, temp))
        q_est, v_est = self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch_a))
        ])

        policy_loss = -(q_est - to_tensor(np.array([std_term])) * torch.sqrt(v_est))

        policy_loss = policy_loss.mean()
        self.policy_loss = policy_loss

        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        s_t_a = np.append(s_t, np.array([self.alpha]))
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t_a])))
        ).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self,num):
        torch.save(
            self.actor.state_dict(),
            'models/actor_'+str(num)+'.pkl'
        )
        torch.save(
            self.critic.state_dict(),
            'models/critic_'+str(num)+'.pkl'
        )

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states+1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x.float())
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x.float())
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = torch.sigmoid(self.fc3(out))

        q_values = out[:,0].unsqueeze(-1)
        variance_values = out[:,1].unsqueeze(-1)

        return q_values, variance_values