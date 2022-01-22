#!/usr/bin/env python
#================================================================
#
#   File name   : BipedalWalker-v3_PPO
#   Author      : PyLessons
#   Created date: 2020-10-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/Reinforcement_Learning
#   Description : BipedalWalker-v3 PPO continuous agent
#   TensorFlow  : 2.3.1
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
import random
import gym
import pylab
import rospy
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, Add
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy

#from openai_ros.task_envs.turtlebot2 import turtlebot2_maze
from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time

import wandb
from wandb.keras import WandbCallback
# Initialize your W&B project allowing it to sync with TensorBoard
wandb.init(name='PPO1', project="PPO Algorithm Testing", sync_tensorboard=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print('GPUs {}'.format(gpus))
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, visualize=False):
        super(Environment, self).__init__()
        self.env = gym.make(env_name)
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()
            #if self.is_render and self.env_idx == 0:
                #self.env.render()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, reward, done, info])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)

        # X = Conv1D(32, 9, strides=4, padding="same", activation="relu", data_format="channels_first", input_shape=input_shape)(X_input)
        # X = Conv1D(64, 5, strides=2, padding="same", activation="relu", data_format="channels_first")(X)
        # X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        # output_v = Dense(1, activation="sigmoid")(X)
        # output_w = Dense(1, activation="tanh")(X)
        # output = tf.concat([output_v, output_w], axis=1)
        #print('output: ' + str(output))

        self.Actor = Model(inputs = X_input, outputs = output)
        #self.Actor.load_weights(self.Actor_name)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        V = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)

        # V = Conv1D(32, 9, strides=4, padding="same", activation="relu", data_format="channels_first", input_shape=input_shape)(X_input)
        # V = Conv1D(64, 5, strides=2, padding="same", activation="relu", data_format="channels_first")(V)
        # V = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        #self.Critic.load_weights(self.Critic_name)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        print('ou_state: ' + str(ou_state))
        action = action + ou_state
        v_action = np.clip(action[0,0], 0, 1)
        #print('v_action: ' + str(v_action))
        w_action = np.clip(action[0,1], -1, 1)
        #print('w_action: ' + str(w_action))
        action = np.array([[v_action, w_action]])
        return action
        # return np.clip(action + ou_state, self.low, self.high)
    

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name, model_name=""):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.shape[0]
        #print('action_size: ' + str(self.action_size))
        self.state_size = self.env.observation_space.shape
        #print('state_size: ' + str(self.state_size))
        self.EPISODES = 200000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = True
        self.Training_batch = 512
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = "{}_PPO_Actor.h5".format(self.env_name)
        self.Critic_name = "{}_PPO_Critic.h5".format(self.env_name)
        #self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)
        #print('pred: ' + str(pred))
        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh

        # v_random = np.random.uniform(0, 1)
        # w_random = np.random.uniform(-1, 1)
        # action_random = np.array([[v_random, w_random]])

        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std

        # action = pred +  action_random * self.std

        # v_action = np.clip(action[0,0], 0, 1)
        # w_action = np.clip(action[0,1], -1, 1)
        # action = np.array([[v_action, w_action]])
        
        action = np.clip(action, low, high)

        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=1, shuffle=self.shuffle, callbacks=[WandbCallback()])
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=1, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        # self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        # self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        wandb.log({"Scores": score})
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if (self.average_[-1] >= self.max_average and save) or str(episode)[-3:] == "000":
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            self.lr *= 0.99
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run_batch(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        #state = np.expand_dims(state, axis=-1)
        #print('state: ' + str(state.shape))
        # noise = OUNoise(self.env.action_space)
        # noise.reset()
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            for t in range(self.Training_batch):
                #self.env.render()
                #print('state: ' + str(state))
                #print('state: ' + str(state.shape))
                value = self.Critic.predict(state)
                # Actor picks an action
                action, logp_t = self.act(state)
                #print('action1: ' + str(action))
                # action = noise.get_action(action, t)
                #print('action_noise: ' + str(action))
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action[0])
                print('episode: ' + str(self.episode) + ', action=' + str(action) + ', reward=' + str(reward) + ', q_value=' + str(value))
                wandb.log({"Q Value": value})
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = np.reshape(next_state, [1, self.state_size[0]])
                #state = np.expand_dims(state, axis=-1)
                score += reward
                if done:
                    
                    average, SAVING = self.PlotModel(score, self.episode)
                    wandb.log({"Average": average})
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar('Workers:{}/score_per_episode'.format(1), score, self.episode)
                    self.writer.add_scalar('Workers:{}/learning_rate'.format(1), self.lr, self.episode)
                    self.writer.add_scalar('Workers:{}/average_score'.format(1),  average, self.episode)
                    
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
                    #state = np.expand_dims(state, axis=-1)
                    self.episode += 1
            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break

        self.env.close()


    def run_multiprocesses(self, num_worker = 4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states =        [[] for _ in range(num_worker)]
        next_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        logp_ts =       [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    # self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    # self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    # self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # terminating processes after a while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):#evaluate
        self.load()
        for e in range(101):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.Actor.predict(state)[0]
                # Predict with 2 output values
                # (pred_v, pred_w) = self.Actor.predict(state)
                # action = np.array([[pred_v[0,0], pred_w[0,0]]])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()
            

if __name__ == "__main__":
    rospy.init_node('turtlebot2_wall_qlearn', anonymous=True, log_level=rospy.FATAL)
    env_name = 'MyTurtleBot2Wall-v0'
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    #env_name = 'BipedalWalker-v3'
    agent = PPOAgent(env_name)
    agent.run_batch() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    #agent.test()