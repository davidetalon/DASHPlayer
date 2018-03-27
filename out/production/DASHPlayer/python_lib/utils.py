# -*- coding: utf-8 -*-

import numpy as np

def generate_fake_state(memory_len = 0):
    # Generate a fake state
    # Previous capacities can be included increasing "memory_len"
    # return random (C, B, D, q)

    state = ()
    for i in range(memory_len+1):
        C = np.random.rand()*20
        state += (C,)
        
    B = np.random.rand()*20
    D = np.random.randint(1,6)
    q = np.random.rand()
 
    state += (B, D, q)
    
    return np.array(state)
    

def generate_fake_event(memory_len = 0):
    # Generate a fake event (state, action, reward, future_state)
    # Previous capacities can be included in state vector increasing "memory_len"
    # return a random (state, a, r, future_state)

    state = generate_fake_state(memory_len=memory_len)
    action = np.random.randint(1,9)
    reward = np.random.rand()
    future_state = generate_fake_state(memory_len=memory_len)
    
    return (state, action, reward, future_state)


def generate_fake_episode(memory_len = 0, length = 200):
    # Generate a whole random video of specified length
    
    episode = []
    for i in range(length):
        event = generate_fake_event(memory_len=memory_len)
        episode.append(event)
        
    return episode


def generate_fake_replay_memory(memory_len=0, video_length=200, capacity=4000):
    # Generate a fake replay memory
    #   memory_len      -> previous capacities to include in the state vector
    #   video_length    -> number of segments in a single video
    #   capacity        -> replay memory capacity

    replay_mem = []
    for i in range(capacity):
        print ("Generating video %d/%d" % (i+1, capacity) )
        episode = generate_fake_episode(memory_len=memory_len, length=video_length)
        replay_mem.append(episode)
        
    return replay_mem