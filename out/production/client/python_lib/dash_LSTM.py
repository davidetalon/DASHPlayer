 # -*- coding: utf-8 -*-

import tensorflow as tf
from pandas import DataFrame
import numpy as np
import utils
import random

tf.reset_default_graph()

#import warnings
#warnings.filterwarnings('error')

class Agent:
        
    lstm_size = 64
    n_input = 4
    n_steps = 10
    n_classes = 8
    batch_size = 10
    
    gamma=0.99
    replay_memory_size = 2000
    target_network_update_step = 3

    learning_rate = 0.01
    

    def __init__(self):
        """     
        
            batch_size         -> Number of segments to be extracted from replay memory during training
            replay_memory_size -> Size of replay memory (number of events)
        
            gamma              -> Discount factor in the Bellman equation
        
            target_network_update_step -> Update target network after C steps
            
            # Network Parameters
            n_hidden_1  -> 1st layer number of features
            n_input     -> State length
            n_classes   -> Number of actions
            
            learning_rate -> Learning rate for the gradent descent
        
        """
        
        # Initialize replay meoory and video memory (a video is added to replay memory only at the end)
        self.replay_mem = []        
        self.video_mem = DataFrame(columns=['state','action','reward','future_state'])
        
        # Initialize network
        print("Initializing network...")
        self.initialize_network()
        
        # Initialize target network
        print("Initializing target network...")
        self.initialize_target_network()
        
        # Initializing the variables
        print("Initializing variables...")
        self.init = tf.initialize_all_variables()
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        # Start session
        self.session = tf.Session()
        self.session.run(self.init)
        
        # Counter for target network update
        self.target_network_update_counter = 0
        
        
    def initialize_network(self):
        
        
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=self.lstm_size,
                                                  input_size=None,
                                                  use_peepholes=False,
                                                  cell_clip=None,
                                                  initializer=None,
                                                  num_proj=None,
                                                  num_unit_shards=1,
                                                  num_proj_shards=1,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True,
                                                  activation=tf.tanh)
                                  
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_steps, self.n_input])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        
        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.lstm_size, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.zeros([self.n_classes])+(1.0/(1-self.gamma))+1)
        }
        
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        
        # Permuting batch_size and n_steps. Now the shape is (n_steps, batch_size, n_input)
        x_t = tf.transpose(self.x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x_t = tf.reshape(x_t, [-1, self.n_input])
        ## Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x_t = tf.split(0, self.n_steps, x_t)
        
        # Evaluate the outputs and states of the LSTM cell
        # self.lstm_outputs, lstm_states = tf.nn.rnn(self.lstm_cell, x_t, dtype=tf.float32)
        self.lstm_outputs, lstm_states = tf.nn.rnn(self.lstm_cell, x_t, dtype=tf.float32)
        
        # Final output (Q value for each action) is obtained using hidden-to-output linear activation function
        self.final_outputs = tf.matmul(self.lstm_outputs[-1], self.weights['out']) + self.biases['out']
        
        # Add temperature to softmax
        self.softmax_temperature = tf.placeholder("float")
        
        softmax_temperature_no_zero = tf.maximum(1e-8,self.softmax_temperature)
        softmax_input = tf.div(self.out_layer, softmax_temperature_no_zero)
        
        # Softmax output
        self.softmax_output = tf.nn.softmax(softmax_input)
        
        # Model evaluation
        self.loss = tf.reduce_mean( (self.y - self.final_outputs)**2 )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        
        
    def initialize_target_network(self):
          
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.n_steps+1, self.n_input])

        x_t = tf.transpose(self.x_target, [1, 0, 2])
        x_t = tf.reshape(x_t, [-1, self.n_input])
        x_t = tf.split(0, self.n_steps+1, x_t)      
      
        # Create target network
        target_lstm_outputs, target_lstm_states = tf.nn.rnn(self.lstm_cell, x_t, dtype=tf.float32, scope='target')      
        
        # Define weights
        self.target_weights = {
            'out': tf.Variable( self.weights['out'].initialized_value() )
        }
        self.target_biases = {
            'out': tf.Variable( self.biases['out'].initialized_value() )
        }
        
        #  Target final output
        self.target_final_outputs = tf.matmul(target_lstm_outputs[-1], self.target_weights['out']) + self.target_biases['out']
        
        
        lstm_weights = tf.get_collection(tf.GraphKeys.VARIABLES, 'RNN/LSTMCell/W_0:0')[0]
        lstm_biases  = tf.get_collection(tf.GraphKeys.VARIABLES, 'RNN/LSTMCell/B:0')[0]
        target_lstm_weights = tf.get_collection(tf.GraphKeys.VARIABLES, 'target/LSTMCell/W_0:0')[0]
        target_lstm_biases  = tf.get_collection(tf.GraphKeys.VARIABLES, 'target/LSTMCell/B:0')[0]
                
        self.set_target_lstm_weights = target_lstm_weights.assign(lstm_weights)
        self.set_target_lstm_biases  = target_lstm_biases.assign(lstm_biases)
        self.set_target_out_weights = self.target_weights['out'].assign(self.weights['out'])
        self.set_target_out_biases  = self.target_biases['out'].assign(self.biases['out'])
    
    
    def choose_action_softmax(self, state, temperature=1):
        ### Forward step on network and apply softmax sample on output
                     
        if temperature == 0:
            return self.choose_action_epsilon_greedy(state, temperature)
            
        # Get latest n_steps-1 states from video memory
        state_video_mem = self.video_mem['state'][-agent.n_steps+1:].as_matrix().tolist()
        
        while len(state_video_mem) < (self.n_steps-1):
            state_video_mem.insert(0,np.zeros(len(state)))
            
        # Add current state
        state_video_mem.append(state)
        
        # Change shape to match network input
        state_video_mem = np.array(state_video_mem).reshape(1, self.n_steps, self.n_input)
         
        # Evaluate softmax output
        softmax_output = self.session.run(self.softmax_output, feed_dict={self.x: state_video_mem, self.softmax_temperature: temperature})[0]
        
        # Sample the action using softmax output as mass pdf
        action = np.random.choice(np.arange(0,np.size(softmax_output)), p=softmax_output)
            
        return action
        
        
    def choose_action_epsilon_greedy(self, state, epsilon):
        ### With probability epsilon choose a random action, otherwise choose
        ### the action with highest Q value
        
        if epsilon > 1 or epsilon < 0:
            raise Exception('epsilon value between 0 and 1')
        
        if np.random.rand() < epsilon:
            # Generate random action
            action = np.random.randint(0,self.n_classes)
        else:
            
            # Get latest n_steps-1 states from video memory
            state_video_mem = self.video_mem['state'][-agent.n_steps+1:].as_matrix().tolist()
            
            while len(state_video_mem) < (self.n_steps-1):
                state_video_mem.insert(0,np.zeros(len(state)))
                
            # Add current state
            state_video_mem.append(state)
            
            # Change shape to match network input
            state_video_mem = np.array(state_video_mem).reshape(1, self.n_steps, self.n_input)
            
            # Evaluate output
            out_layer = self.session.run(self.final_outputs, feed_dict={self.x: state_video_mem})[0]
            # Choose the index of the max
            action = np.argmax(out_layer)
            
        return action

        
    def update(self, state, action, reward, future_state):
                
        global y_vec, state_vec, futurestate_vec, reward_vec, action_vec, target_final_outputs, y, train_set, rand_idx
                
        # Add (state, action, reward, future_state) to video memory (current video)
        self.add_to_video_mem(state, action, reward, future_state)
        
        # Check if there are videos into replay memory (list of previous videos)
        if len(self.replay_mem) == 0:
            return
            
        # Randomly sample from replay memory
        if len(self.replay_mem) < self.batch_size:
            train_set = self.replay_mem
        else:
            random.shuffle(self.replay_mem)
            train_set = self.replay_mem[:self.batch_size]

        
        y_vec = []
        state_vec = []
        futurestate_vec = []
        reward_vec = []
        action_vec = []
        for vid in train_set:
            
            # Generate random index
            vid_len = len(vid)
            rand_idx = np.random.randint(self.n_steps+1, vid_len)
            # Extract n_steps previous state
            states = np.array( vid['state'][rand_idx-self.n_steps:rand_idx].as_matrix().tolist() )
            state_vec.append(states)
            
            # Extract n+1_steps previous future_state
            futurestate = np.array( vid['future_state'][rand_idx-(self.n_steps+1):rand_idx].as_matrix().tolist() )
            futurestate_vec.append(futurestate)
            
            # Evaluate reward and action for random index (-1) sample
            reward = vid['reward'][rand_idx-1]
            action = int(vid['action'][rand_idx-1])
            reward_vec.append(reward)
            action_vec.append(action)
        action_vec = np.array(action_vec)
        
        # Evaluate target network output using future states as input
        target_final_outputs = self.session.run(self.target_final_outputs, feed_dict={self.x_target: futurestate_vec})

        # Take the max for wach future_state
        max_future_q = np.max(target_final_outputs, axis=1)
        
        # Evaluate the wanted output (in array form)
        y_vec = reward_vec + self.gamma * max_future_q
        
        # Evaluate the network output for gradient descend (Convert in matrix form, with correct action)
        y = np.zeros([len(y_vec), self.n_classes])
        for i in range(self.n_classes):
            mask = action_vec==(i+1)
            y[mask,i] = y_vec[mask]
        
        # Gradient descend with L = (y_j - Q(state_j, action_j) )^2
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict={self.x: state_vec, self.y: y})
        
        # Every C step update target network Q_target weigths (equal to Q)
        self.target_network_update_counter += 1
        if self.target_network_update_counter >= self.target_network_update_step:
            self.target_network_update_counter = 0
            self.update_target_network()
    
        return loss


    def add_to_video_mem(self, state, action, reward, future_state):
        ### Add (state, action, reward, future_state) to video memory
        
        # Create a dataframe with new event
        event = DataFrame([(state, action, reward, future_state)], columns=['state','action','reward','future_state'])

        # Add event to video memory
        self.video_mem = self.video_mem.append(event, ignore_index=True)
        
        # Reindex dataframe
        self.video_mem.index = range(0,len(self.video_mem))


    def update_replay_mem(self):
        ### Save the current video the the replay memory and clear video memory
        self.replay_mem.append(self.video_mem)
        self.video_mem = DataFrame(columns=['state','action','reward','future_state'])
        
        # Verify if replay mem is too big
        while len(self.replay_mem) > self.replay_memory_size:
            # Remove the first (oldest) element
            self.replay_mem = self.replay_mem[1:]


    def update_target_network(self):
        ### Copy weights and biases from Q network to target network
    
        self.session.run(self.set_target_lstm_weights)
        self.session.run(self.set_target_lstm_biases)
        self.session.run(self.set_target_out_weights)
        self.session.run(self.set_target_out_biases)

    def close_session(self):
        ### Close tensorflow session opened during initialization
        self.session.close()


    def save_model(self, path):
        ### Save current model
        self.saver.save(self.session, path)


    def load_model(self, path):
        ### Load model
        self.saver.restore(self.session, path)


import time

if __name__ == "__main__":

    agent = Agent()
     
#    x_ = np.random.rand(agent.batch_size,agent.n_steps,agent.n_input)
#    lstm_outputs_, final_outputs, target_final_outputs = agent.session.run([agent.lstm_outputs, agent.final_outputs,agent.target_final_outputs], feed_dict={agent.x: x_})
    agent.update_target_network()

    for i in range(20):
        print(i+1)
        t = time.time()
        for i in range(20):
            state, action, reward, future_state = utils.generate_fake_event(memory_len = 0)
            
            agent.choose_action_softmax(state, 1)
            agent.choose_action_epsilon_greedy(state, 0.3)
            
            agent.update( state, action, reward, future_state )
            
            agent.update_target_network()
            
        print(time.time()-t)
        agent.update_replay_mem()
        
    
    agent.close_session()
    
    
#%%
    
import tensorflow as tf    
    
lstm_size = 64
n_steps = 10
n_input = 4
n_classes = 8
learning_rate = 0.1

lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_size,
                                          input_size=None,
                                          use_peepholes=False,
                                          cell_clip=None,
                                          initializer=None,
                                          num_proj=None,
                                          num_unit_shards=1,
                                          num_proj_shards=1,
                                          forget_bias=1.0,
                                          state_is_tuple=True,
                                          activation=tf.tanh)

x = tf.placeholder(tf.float32, shape=[None, n_steps, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([lstm_size, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, n_steps, n_input)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

# Permuting batch_size and n_steps. Now the shape is (n_steps, batch_size, n_input)
x_t = tf.transpose(x, [1, 0, 2])
# Reshaping to (n_steps*batch_size, n_input)
x_t = tf.reshape(x_t, [-1, n_input])
## Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x_t = tf.split(0, n_steps, x_t)

# Evaluate the outputs and states of the LSTM cell
# lstm_outputs, lstm_states = tf.nn.rnn(lstm_cell, x_t, dtype=tf.float32)
lstm_outputs, lstm_states = tf.nn.rnn(lstm_cell, x_t, dtype=tf.float32)

# Final output (Q value for each action) is obtained using hidden-to-output linear activation function
final_outputs = tf.matmul(lstm_outputs[-1], weights['out']) + biases['out']

# Add temperature to softmax
softmax_temperature = tf.placeholder("float")
softmax_input = final_outputs / softmax_temperature

# Softmax output
softmax_output = tf.nn.softmax(softmax_input)

# Model evaluation
loss = tf.reduce_mean( (y - final_outputs)**2 )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



sess = tf.Session()

x_ = np.random.rand(50, n_steps, n_input)
sess.run(final_outputs, feed_dict={x: x_})

sess.close()