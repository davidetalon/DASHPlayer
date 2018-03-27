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
        
#    lstm_size = 64
#    n_input = 4
#    n_steps = 10
#    n_classes = 8
#    batch_size = 100
#    
#    gamma=0.9
#    replay_memory_size = 2000
#    target_network_update_step = 3
#
#    learning_rate = 0.01


    def __init__(self, batch_size=10, replay_memory_size=100, gamma=0.9, target_network_update_step=10, lstm_size=64, n_steps=64, n_input=5, n_classes=8, learning_rate=0.001, en_peepholes=False):
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
        
        tf.reset_default_graph()

        # PARAMETERS
        self.lstm_size = lstm_size
        self.n_input = n_input
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.batch_size = batch_size
        
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.target_network_update_step = target_network_update_step
    
        self.learning_rate = learning_rate
        self.en_peepholes = en_peepholes
        
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
                                                  use_peepholes=self.en_peepholes,
                                                  cell_clip=None,
                                                  initializer=None,
                                                  num_proj=None,
                                                  num_unit_shards=1,
                                                  num_proj_shards=1,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True,
                                                  activation=tf.tanh)
                                  
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.n_input])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
                
        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.lstm_size, self.n_classes])*0.05)
        }
        self.biases = {
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }
        
        # Evaluate the outputs and states of the LSTM cell
        # self.lstm_outputs, lstm_states = tf.nn.rnn(self.lstm_cell, x_t, dtype=tf.float32)
        self.lstm_outputs, self.lstm_states = tf.nn.dynamic_rnn(self.lstm_cell, self.x, dtype=tf.float32, time_major=False)
        
        # Final output (Q value for each action) is obtained using hidden-to-output linear activation function
        self.final_outputs = tf.matmul(self.lstm_outputs[:,-1,:], self.weights['out']) + self.biases['out']
        
        # Add temperature to softmax
        self.softmax_temperature = tf.placeholder("float")
        
        softmax_temperature_no_zero = tf.maximum(1e-8,self.softmax_temperature)
        softmax_input = tf.div(self.final_outputs, softmax_temperature_no_zero)
        
        # Softmax output
        self.softmax_output = tf.nn.softmax(softmax_input)
        
#         Define loss and optimizer
#        self.actions = tf.placeholder("int32", None)
#        one_hot_mask = tf.one_hot(indices=self.actions, depth=self.n_classes, on_value=1.0)
#        self.sel_out_layer=one_hot_mask*self.final_outputs
        
        self.sel_out_layer=self.final_outputs # Remove if one hot output is enabled
        
        # Model evaluation
        self.loss = tf.reduce_mean( tf.square(self.y - self.sel_out_layer) )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        
        
    def initialize_target_network(self):
             
        # Create target network
        self.target_lstm_outputs, self.target_lstm_states = tf.nn.dynamic_rnn(self.lstm_cell, self.x, dtype=tf.float32, time_major=False, scope='target')      
        
        # Define weights
        self.target_weights = {
            'out': tf.Variable( self.weights['out'].initialized_value() )
        }
        self.target_biases = {
            'out': tf.Variable( self.biases['out'].initialized_value() )
        }
        
        #  Target final output
        self.target_final_outputs = tf.matmul(self.target_lstm_outputs[:,-1,:], self.target_weights['out']) + self.target_biases['out']
        
        
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
        if self.n_steps == 0:
            state_video_mem = self.video_mem['state'].as_matrix().tolist()
        else:
            state_video_mem = self.video_mem['state'][-self.n_steps+1:].as_matrix().tolist()
                    
        # Add current state
        state_video_mem.append(state)
        
        # Change shape to match network input [batch_size, max_time, ...]
        state_video_mem = np.array(state_video_mem)
        state_video_mem = np.expand_dims(state_video_mem, 0)
         
        # Evaluate softmax output (last output)
        final_outputs, softmax_output = self.session.run([self.final_outputs, self.softmax_output], feed_dict={self.x: state_video_mem, self.softmax_temperature: temperature})
        
        softmax_output = softmax_output[-1]
        final_outputs = final_outputs[-1]
        
        # Sample the action using softmax output as mass pdf
        action = np.random.choice(np.arange(0,np.size(softmax_output)), p=softmax_output)
            
        return action+1, final_outputs
        
        
    def choose_action_epsilon_greedy(self, state, epsilon):
        ### With probability epsilon choose a random action, otherwise choose
        ### the action with highest Q value
        
        if epsilon > 1 or epsilon < 0:
            raise Exception('epsilon value between 0 and 1')
                
        # Get latest n_steps-1 states from video memory
        if self.n_steps == 0:
            state_video_mem = self.video_mem['state'].as_matrix().tolist()
        else:
            state_video_mem = self.video_mem['state'][-self.n_steps+1:].as_matrix().tolist()
        
        # Add current state
        state_video_mem.append(state)
        
        # Change shape to match network input [batch_size, max_time, ...]
        state_video_mem = np.array(state_video_mem)
        state_video_mem = np.expand_dims(state_video_mem, 0)
        
        # Evaluate output (last output)
        final_outputs = self.session.run(self.final_outputs, feed_dict={self.x: state_video_mem})[-1]
        
        if np.random.rand() < (epsilon / (self.n_classes - 1) * self.n_classes):
            # Generate random action
            action = np.random.randint(0,self.n_classes)
        else:
            # Choose the index of the max
            action = np.argmax(final_outputs)
            
        return action+1, final_outputs

        
    def update(self, state, action, reward, future_state):
#        global y_vec, state_mat, futurestate_mat, reward_mat, action_mat, target_final_outputs, y, train_set, rand_idx, action_vec
                
        # Add (state, action, reward, future_state) to video memory (current video)
#        self.add_to_video_mem(state, action, reward, future_state)
        
        
        # Check if there are videos into replay memory (list of previous videos)
        if len(self.replay_mem) == 0:
            return 0
            
        # Randomly sample from replay memory
        train_set = []
        while len(train_set) < self.batch_size:
            train_set += self.replay_mem
#        random.shuffle(train_set)
        train_set = train_set[:self.batch_size]
            
        # Generate random index
        episode_len = len(self.replay_mem[0])
        
        state_mat = []
        futurestate_mat = []
        reward_mat = []
        action_mat = []
        for episode in train_set:    
            rand_idx = np.random.randint(self.n_steps+1,episode_len)
            if self.n_steps == 0:
                state_mat.append(np.array(episode['state'][:rand_idx].as_matrix().tolist()))
                futurestate_mat.append(episode['future_state'][:rand_idx].as_matrix().tolist())
                reward_mat.append(episode['reward'][:rand_idx].as_matrix().tolist())
                action_mat.append(episode['action'][:rand_idx].as_matrix().tolist())
            else:
                state_mat.append(np.array(episode['state'][rand_idx-self.n_steps:rand_idx].as_matrix().tolist()))
                futurestate_mat.append(episode['future_state'][rand_idx-self.n_steps-1:rand_idx].as_matrix().tolist())
                reward_mat.append(episode['reward'][rand_idx-self.n_steps:rand_idx].as_matrix().tolist())
                action_mat.append(episode['action'][rand_idx-self.n_steps:rand_idx].as_matrix().tolist())
        state_mat       = np.array(state_mat)
        futurestate_mat = np.array(futurestate_mat)
        reward_mat      = np.array(reward_mat)
        action_mat      = np.array(action_mat) - 1 #!!!!!!!!!!!!!
              
        action_vec = action_mat[:, -1]
        reward_vec = reward_mat[:,-1]
        
        # Evaluate current output
        current_outputs = self.session.run(self.final_outputs, feed_dict={self.x: state_mat})

        # Evaluate target network output using future states as input
        target_future_outputs = self.session.run(self.target_final_outputs,  feed_dict={self.x: futurestate_mat})

        future_max_vector = np.max(target_future_outputs, axis = 1)
        wanted_output = current_outputs.copy()
        for i in range(len(wanted_output)):
            max_value = future_max_vector[i]
            max_index = int(action_vec[i])
            wanted_output[i, max_index] = reward_vec[i] + self.gamma * max_value
                           
        # Gradient descend with L = (y_j - Q(state_j, action_j) )^2
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict={self.x: state_mat, self.y: wanted_output})
        
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


    def update_replay_mem(self, test=False):
        ### Save the current video the the replay memory and clear video memory
        if not test:
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
        
    def delete_replay_memory(self):        
        self.replay_mem = []


import time

if __name__ == "__main__":
    
    # PARAMETERS
    lstm_size = 128
    n_input = 4
    n_steps = 5
    n_classes = 8
    batch_size = 50
    
    gamma=0.1
    replay_memory_size = 100
    target_network_update_step = 10

    learning_rate = 0.1
    
    # INITIALIZATION
    agent = Agent(lstm_size=lstm_size, n_input=n_input, n_steps=n_steps, n_classes=n_classes, batch_size=batch_size, gamma=gamma, replay_memory_size=replay_memory_size, target_network_update_step=target_network_update_step, learning_rate=learning_rate)
     
#    # %%Test initialization
#    x_ = np.random.rand(agent.batch_size, agent.n_steps, agent.n_input)
#    lstm_outputs_, lstm_states_, lstm_final_outputs_ = agent.session.run([agent.lstm_outputs, agent.lstm_states, agent.final_outputs], feed_dict={agent.x: x_})
#    target_lstm_outputs_, target_lstm_states_, target_lstm_final_outputs_ = agent.session.run([agent.target_lstm_outputs, agent.target_lstm_states, agent.target_final_outputs], feed_dict={agent.x: x_})
#    agent.update_target_network()
#    target_lstm_outputs_2, target_lstm_states_2, target_lstm_final_outputs_2 = agent.session.run([agent.target_lstm_outputs, agent.target_lstm_states, agent.target_final_outputs], feed_dict={agent.x: x_})
#    if ( np.sum(target_lstm_outputs_2 != lstm_outputs_) + np.sum(target_lstm_final_outputs_2 != lstm_final_outputs_) ) > 0:
#        print ("ERROR: Target network update problem!!")
#    
#    
#    #%% Choose action
#    state, action, reward, future_state = utils.generate_fake_event(memory_len = n_input-4)
#    action, final_outputs = agent.choose_action_softmax(state, 1)
#    print(action)
#    print(final_outputs)
#    action, final_outputs = agent.choose_action_epsilon_greedy(state, 0.1)
#    print(action)
#    print(final_outputs)
    
    
#    #%% Populate video memory and replay memory
#    for j in range(10):
#        for i in range(50):
#            state, action, reward, future_state = utils.generate_fake_event(memory_len = n_input-4) 
#            agent.add_to_video_mem(state, action, reward, future_state)
#        agent.update_replay_mem()
#    for i in range(50):
#        state, action, reward, future_state = utils.generate_fake_event(memory_len = n_input-4) 
#        agent.add_to_video_mem(state, action, reward, future_state)
#    
#    state, action, reward, future_state = utils.generate_fake_event(memory_len = n_input-4)  
#    action, final_outputs = agent.choose_action_softmax(state, 1)
#    print(action)
#    
    
    #%% Update test
    
#    # Clear replay and video memory
#    agent.replay_mem = []        
#    agent.video_mem = DataFrame(columns=['state','action','reward','future_state'])
#    
#    # Always same state and random action
#    for j in range(10):
#        print(j)
#        time.sleep(2)
#        for i in range(400):
#            state, action, reward, future_state = utils.generate_fake_event(memory_len = n_input-4)
#            state = future_state = [0.5,0.8,0.7,0.6]
#            reward = 0.1
#            agent.update(state, action, reward, future_state)
#            final_outputs, target_final_outputs = agent.session.run([agent.final_outputs, agent.target_final_outputs], feed_dict={agent.x: np.array(state).reshape(1,1,4)})
#            print (final_outputs)
#        agent.update_replay_mem()
    
    
    #%% TEST ONE BEST ACTION
    
    video_len = 400
    video_num = 3
            
    state = np.random.rand(4)
    for i in range(video_num):
        print(i)
        for j in range(video_len):
            future_state = np.random.rand(4)
            action, out_layer = agent.choose_action_softmax(state,1)
            if action==3:
                reward=0.99
            else:
                reward=0.01
                
            agent.add_to_video_mem(state, action, reward, future_state)
            agent.update(state, action, reward, future_state)
            state = future_state
            print(out_layer)
        agent.update_replay_mem()
            
    for i in range(1000):
        state = np.random.rand(4)
        action, out_layer = agent.choose_action_softmax(state,0)
        print('%d - %s' % (action, state))
    
#    # Get latest n_steps-1 states from video memory
#    state_video_mem = agent.video_mem['state'].as_matrix().tolist()
#    
#    # Add current state
#    state_video_mem.append(state)
#    
#    # Change shape to match network input [batch_size, max_time, ...]
#    state_video_mem = np.array(state_video_mem)
#    state_video_mem = np.expand_dims(state_video_mem, 0)    
#    
#    # Check networks output
#    agent.update_target_network()
#    final_outputs, target_final_outputs = agent.session.run([agent.final_outputs, agent.target_final_outputs], feed_dict={agent.x: state_video_mem})
#    if np.sum(final_outputs != target_final_outputs) > 0:
#        print('ERROR: target network output problem!')
#    
#    agent.update(state, action, reward, future_state)
#    final_outputs2, target_final_outputs2 = agent.session.run([agent.final_outputs, agent.target_final_outputs], feed_dict={agent.x: state_video_mem})
#    if np.sum(final_outputs != final_outputs2) == 0:
#        print('ERROR: No update!!')
        
#%%    x_ = np.random.rand(agent.batch_size,agent.n_steps,agent.n_input)
#    lstm_outputs_, final_outputs, target_final_outputs = agent.session.run([agent.lstm_outputs, agent.final_outputs,agent.target_final_outputs], feed_dict={agent.x: x_})
#    agent.update_target_network()
#
#    for i in range(20):
#        print(i+1)
#        t = time.time()
#        for i in range(20):
#            state, action, reward, future_state = utils.generate_fake_event(memory_len = 0)
#            
#            agent.choose_action_softmax(state, 1)
#            agent.choose_action_epsilon_greedy(state, 0.3)
#            
#            agent.update( state, action, reward, future_state )
#            
#            agent.update_target_network()
#            
#        print(time.time()-t)
#        agent.update_replay_mem()
        
    
#    agent.close_session()
    
    
#%%
    
#import tensorflow as tf    
#import utils
#    
#tf.reset_default_graph()
#
#lstm_size = 64
#n_steps = 10
#n_input = 4
#n_classes = 8
#learning_rate = 0.1
#
#lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_size,
#                                          input_size=None,
#                                          use_peepholes=False,
#                                          cell_clip=None,
#                                          initializer=None,
#                                          num_proj=None,
#                                          num_unit_shards=1,
#                                          num_proj_shards=1,
#                                          forget_bias=1.0,
#                                          state_is_tuple=True,
#                                          activation=tf.tanh)
#
#x = tf.placeholder(tf.float32, shape=[None, None, n_input]) # If time_major == False -> [batch_size, max_time, ...]
#y = tf.placeholder(tf.float32, shape=[None, n_classes])
#
## Define weights
#weights = {
#    'out': tf.Variable(tf.random_normal([lstm_size, n_classes]))
#}
#biases = {
#    'out': tf.Variable(tf.random_normal([n_classes]))
#}
#
#
## Evaluate the outputs and states of the LSTM cell
## lstm_outputs, lstm_states = tf.nn.rnn(lstm_cell, x_t, dtype=tf.float32)
#lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, time_major=False)
#
## Final output (Q value for each action) is obtained using hidden-to-output linear activation function
#final_outputs = tf.matmul(lstm_outputs[-1], weights['out']) + biases['out']
#
## Add temperature to softmax
#softmax_temperature = tf.placeholder("float")
#softmax_input = final_outputs / softmax_temperature
#
## Softmax output
#softmax_output = tf.nn.softmax(softmax_input)
#
## Model evaluation
#loss = tf.reduce_mean( (y - final_outputs)**2 )
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
#
#init = tf.initialize_all_variables()
#
#sess = tf.Session()
#sess.run(init)
#
#
#
#state, action, reward, future_state = utils.generate_fake_event(memory_len = 1)  
#
#
#x_ = np.random.rand(50, 10, n_input)
#lstm_outputs_, lstm_states_ = sess.run([lstm_outputs, lstm_states], feed_dict={x: x_})
#
#sess.close()