 # -*- coding: utf-8 -*-

import tensorflow as tf
from pandas import DataFrame
import numpy as np
import utils

#import warnings
#warnings.filterwarnings('error')

class Agent:
        
    def __init__(self, batch_size=100, replay_memory_size=1000, gamma=0.9, target_network_update_step=10, n_hidden_1=64, n_hidden_2=64, n_input=5, n_classes=8, learning_rate=0.001, dropout_keep_prob=1):
        """     
        
            batch_size         -> Number of segments to be extracted from replay memory during training
            replay_memory_size -> Size of replay memory (number of events)
        
            gamma              -> Discount factor in the Bellman equation
        
            target_network_update_step -> Update target network after C steps
            
            # Network Parameters
            n_hidden_1  -> 1st layer number of features
            n_hidden_2  -> 2nd layer number of features
            n_input     -> State length
            n_classes   -> Number of actions
            
            learning_rate -> Learning rate for the gradent descent
        
        """
        
        ### PARAMETERS
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size # In this case is the video memory
        
        self.gamma = gamma # Discount factor
        
        self.target_network_update_step = target_network_update_step # Update target network after C steps
    
        # Network Parameters
        self.n_hidden_1 = n_hidden_1 # 1st layer number of features
        self.n_hidden_2 = n_hidden_2 # 2nd layer number of features
        self.n_input = n_input     # State length
        self.n_classes = n_classes   # Number of actions
        
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
    
        # Initialize replay memory and video memory (a video is added to replay memory only at the end)
        self.replay_mem = []        
        self.video_mem = DataFrame(columns=['state','action','reward','future_state'])
        
        # Initialize network
        self.initialize_network()
        
        # Initializing the variables
        self.init = tf.initialize_all_variables()
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        # Start session
        self.session = tf.Session()
        self.session.run(self.init)
        
        # Counter for target network update
        self.target_network_update_counter = 0
        
        
    def initialize_network(self):
        
        n_hidden_1 = self.n_hidden_1
        n_hidden_2 = self.n_hidden_2 
        n_input = self.n_input 
        n_classes = self.n_classes
        
        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
    
        self.keep_prob = tf.placeholder(tf.float32)
    
        # Store layers weight & bias
        #creo i pesi degli archi che collegano i neuroni tf.variable
        #tf.random_normal(shape)
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*0.01),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*0.01),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*0.01)
        }

        # Creo i biases, imposto i tensori tutti a zero
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
#            'out': tf.Variable(tf.zeros([n_classes])+(1.0/(1-self.gamma))+1)
            'out': tf.Variable(tf.zeros([n_classes]))
        }
        
        # Hidden layer with RELU activation
        #(weights * ingressi) + biases
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])

        #funzione attivazione tangente iperbolica
        layer_1 = tf.nn.tanh(layer_1)

        # Hidden layer with RELU activation
        #(weights * valoriLayerPrec) + biases
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])

        #funzione attivazione tangente iperbolica
        layer_2 = tf.nn.tanh(layer_2)
        
        # Dropout
        # con prob keep_prob dà 1/keep_prob altrimenti da 0
        layer_2_DO = tf.nn.dropout(layer_2, self.keep_prob)
            
        
        # Output layer with linear activation
        #output = (weights * layer2_DO)  + biases
        self.out_layer = tf.add(tf.matmul(layer_2_DO, self.weights['out']), self.biases['out'])
        
        # Add temperature to softmax
        self.softmax_temperature = tf.placeholder(tf.float32)
                
        softmax_temperature_no_zero = tf.maximum(1e-8,self.softmax_temperature)
        softmax_input = tf.div(self.out_layer, softmax_temperature_no_zero)
        
        # Softmax output
        self.softmax_output = tf.nn.softmax(softmax_input)
    
        # Initialize target network
        self.initialize_target_network()
        
        # Define loss and optimizer
        self.action_vector = tf.placeholder(tf.int32, [None])
        self.reward_vector = tf.placeholder(tf.float32, [None])
          
        # Take the max for each future_state
        max_future_q = tf.reduce_max(self.target_out_layer, reduction_indices=1)
        
        # Evaluate the wanted output (in array form)
        target_vector = tf.add(self.reward_vector, tf.multiply(self.gamma, max_future_q))
        
        one_hot_mask_positive = tf.one_hot(indices=self.action_vector, depth=self.n_classes, on_value=0.0, off_value=1.0)
        one_hot_mask_negative = tf.one_hot(indices=self.action_vector, depth=self.n_classes, on_value=1.0, off_value=0.0)
        
        masked_out = tf.multiply(one_hot_mask_positive, self.out_layer)
        masket_target = tf.multiply(tf.transpose([target_vector]*8), one_hot_mask_negative)
        
        wanted_out = tf.add(masked_out, masket_target)       
        
        self.loss = tf.reduce_mean( (self.out_layer - wanted_out)**2 )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        
        
    def initialize_target_network(self):
                
        # tf Graph input
        self.target_x = tf.placeholder(tf.float32, [None, self.n_input])
    
        # Store layers weight & bias (initialized as the original network)
        self.target_weights = {
            'h1': tf.Variable( self.weights['h1'].initialized_value() ),
            'h2': tf.Variable( self.weights['h2'].initialized_value() ),
            'out': tf.Variable(self.weights['out'].initialized_value() )
        }
        self.target_biases = {
            'b1': tf.Variable( self.biases['b1'].initialized_value() ),
            'b2': tf.Variable( self.biases['b2'].initialized_value() ),
            'out': tf.Variable( self.biases['out'].initialized_value() )
        }
                
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.target_x, self.target_weights['h1']), self.target_biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.target_weights['h2']), self.target_biases['b2'])
        layer_2 = tf.nn.tanh(layer_2)
        

        # Output layer with linear activation
        self.target_out_layer = tf.matmul(layer_2, self.target_weights['out']) + self.target_biases['out']

        # Define update operators
        # copia i valori dei dei pesi e dei biases nella target network      
        self.set_target_weights_h1  = self.target_weights['h1'].assign(self.weights['h1'])
        self.set_target_weights_h2  = self.target_weights['h2'].assign(self.weights['h2'])
        self.set_target_weights_out = self.target_weights['out'].assign(self.weights['out'])
        
        self.set_target_biases_b1  = self.target_biases['b1'].assign(self.biases['b1'])
        self.set_target_biases_b2  = self.target_biases['b2'].assign(self.biases['b2'])
        self.set_target_biases_out = self.target_biases['out'].assign(self.biases['out'])
    
    
    def choose_action_softmax(self, state, temperature):
        ### Forward step on network and apply softmax sample on output
        
        if temperature == 0:
            return self.choose_action_epsilon_greedy(state, temperature)
        
        # Change state shape and datatype to match network input
        if len(np.shape(state)) == 1:
            state = np.array(state).reshape(-1,len(state))
        
        # Evaluate softmax output
        out_layer, softmax_output = self.session.run([self.out_layer, self.softmax_output], feed_dict={self.x: state, self.softmax_temperature: temperature, self.keep_prob: 1})
        
        softmax_output = softmax_output[0]
        out_layer = out_layer[0]
        
        # Sample the action using softmax output as mass pdf
        action = np.random.choice(np.arange(0,np.size(softmax_output)), p=softmax_output)
            
        return action+1, out_layer
        
        
    def choose_action_epsilon_greedy(self, state, epsilon):
        ### With probability epsilon choose a random action, otherwise choose
        ### the action with highest Q value
        
        if epsilon > 1 or epsilon < 0:
            raise Exception('epsilon value between 0 and 1')
        
        # Change state shape and datatype to match network input
        if len(np.shape(state)) == 1:
            state = np.array(state).reshape(-1,len(state))
        # Evaluate output
        out_layer = self.session.run(self.out_layer, feed_dict={self.x: state, self.keep_prob: 1})[0]
            
        if np.random.rand() < (epsilon / (self.n_classes - 1) * self.n_classes):
            # Generate random action
            action = np.random.randint(0,self.n_classes)
        else:
            # Choose the index of the max
            action = np.argmax(out_layer)
            
        return action+1, out_layer

        
    def update(self, state, action, reward, future_state):
                                
        # Add (state, action, reward, future_state) to video memory
#        self.add_to_video_mem(state, action, reward, future_state)
        
        # Check if there are videos into video memory
        if len(self.video_mem) == 0:
            return
            
        # Randomly sample from replay memory
        if len(self.video_mem) < self.batch_size:
            train_set = self.video_mem
        else:
            train_set = self.video_mem.sample(self.batch_size)

        # Separate the single vector from training set
        state_matrix        = np.array( train_set['state'].as_matrix().tolist() )
        future_state_matrix = np.array( train_set['future_state'].as_matrix().tolist() )
        reward_vec          = np.array( train_set['reward'].as_matrix().tolist() )
        action_vec          = np.array( train_set['action'].as_matrix().tolist() , dtype=int) - 1
        
        # Gradient descend with L = (y_j - Q(state_j, action_j) )^2
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict={self.target_x: future_state_matrix, self.x: state_matrix, self.reward_vector: reward_vec, self.action_vector: action_vec, self.keep_prob: self.dropout_keep_prob})
        
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
        
        # Verify if video mem is too big
        while len(self.video_mem) > self.replay_memory_size:
            # Remove the first (oldest) element
            self.video_mem = self.video_mem[1:]
        
        # Reindex dataframe
        self.video_mem.index = range(0,len(self.video_mem))


    def update_replay_mem(self, test=False):
        ### In this case the network does not have memory, so I use the video memory as replay memory,
        ### and never delete it
        return        


    def update_target_network(self):
        ### Copy weights and biases from Q network to target network
    
        self.session.run(self.set_target_weights_h1)
        self.session.run(self.set_target_weights_h2)
        self.session.run(self.set_target_weights_out)
        self.session.run(self.set_target_biases_b1)
        self.session.run(self.set_target_biases_b2)
        self.session.run(self.set_target_biases_out)
    

    def close_session(self):
        ### Close tensorflow session opened during initialization
        self.session.close()


    def save_model(self, path):
        ### Save current model
        self.saver.save(self.session, path)


    def load_model(self, path):
        ### Load model
        self.saver.restore(self.session, path)


#%%
if __name__ == "__main__":

    
    agent = Agent(batch_size=1,
                  replay_memory_size=100000,
                  gamma=0.1,
                  target_network_update_step=10, 
                  n_hidden_1=64, 
                  n_hidden_2=128, 
                  n_input=5, 
                  n_classes=8, 
                  learning_rate=0.0001,
                  dropout_keep_prob=0.75)
    
    video_len = 400
    video_num = 10
    

# BISOGNA STABILIRE 
#   1. REWARD
#   2. FUTURE STATE    

    state = np.random.rand(5)
    for i in range(video_num):
        print(i)
        for j in range(video_len):
            future_state = np.random.rand(5)
#            action, out_layer = agent.choose_action_softmax(state,1)
            action, out_layer = agent.choose_action_epsilon_greedy(state,0.1)
            if action==3:
                reward=0.99
            else:
                reward=0.01
                
            agent.add_to_video_mem(state, action, reward, future_state)
            agent.update(state, action, reward, future_state)
            state = future_state
            print(out_layer)
            
    for i in range(1000):
        state = np.random.rand(5)
        action, out_layer = agent.choose_action_softmax(state,0.01)
        print('%f - %d' % (state[0], action))

    agent.close_session()