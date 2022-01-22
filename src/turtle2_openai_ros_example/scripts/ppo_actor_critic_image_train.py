#!/usr/bin/env python
#import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2



import gym
import numpy as np
import rospy

# import our training environment
from openai_ros.task_envs.turtlebot2 import turtlebot2_maze
from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

tf.keras.backend.set_floatx('float64')
#wandb.init(name='PPO', project="deep-rl-tf2")


#The parameter of Network

gamma = rospy.get_param('/turtlebot2/gamma')
update_interval = rospy.get_param('/turtlebot2/update_interval')
actor_lr = rospy.get_param('/turtlebot2/actor_lr')
critic_lr = rospy.get_param('/turtlebot2/critic_lr')
clip_ratio = rospy.get_param('/turtlebot2/clip_ratio')
lmbda = rospy.get_param('/turtlebot2/lmbda')

epochs = rospy.get_param('/turtlebot2/epochs')




class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        feature_extractor = MobileNetV2(include_top=False, weights='imagenet')
        for layer in feature_extractor.layers:
            layer.trainable = False
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(1024, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(
            tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(
            new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float64)

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(old_policy, logits, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        # feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
        # for layer in feature_extractor.layers:
        #     layer.trainable = False
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(1024, activation='relu'),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + gamma * forward_val - v_values[k]
            gae_cumulative = gamma * lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=5000):
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.model.predict(
                    np.reshape(state, [1, self.state_dim]))
                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward * 0.01)
                old_policy_batch.append(probs)

                if len(state_batch) >= update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            if episode_reward > 400:
                print('best reward=' + str(episode_reward))
                self.actor.model.save('model_actor_{}_{}.hdf5'.format(ep, episode_reward))
                self.actor.model.save('model_critic_{}_{}.hdf5'.format(ep, episode_reward))
            print(self.state_dim)
            print(self.action_dim)
            print(np.shape(state))
            print(state)
            print(np.reshape(state, [1, self.state_dim]))
            #wandb.log({'Reward': episode_reward})



def main():
    rospy.init_node('turtlebot2_maze_qlearn', anonymous=True, log_level=rospy.FATAL)
    env_name = 'MyTurtleBot2Wall-v0'
    env = gym.make(env_name)
    agent = Agent(env)
    agent.train()


if __name__ == "__main__":
    main()
