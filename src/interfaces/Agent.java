/**
 * @file Agent.java
 * @brief Agent interface
 *
 * @author Davide Talon (<talon.davide@gmail.com>)
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */



package interfaces;


public interface Agent {


    /**
     * @brief implements Epsilon-Greedy policy
     *
     * @param array_state       array of double which indicates the current state
     * @param epsilon           a double with E-Greedy parameter
     * @param netSnap           a NetworkSnap object
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void choose_action_epsilon_greedy(double[] array_state, double epsilon, NetworkSnap netSnap);


    /**
     * @brief implements Softmax policy
     *
     * @param array_state       array of double which indicates the current state
     * @param epsilon           a double with E-Greedy parameter
     * @param netSnap           a NetworkSnap object
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void choose_action_softmax(double[] array_state, double epsilon, NetworkSnap netSnap);


    /**
     * @brief update network weights
     *
     * @param array_state           array of double which indicates the current state
     * @param action                a int with got action
     * @param reward                a double with network's reward
     * @param array_future_state    array of double with future state
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public double update(double[] array_state, int action, double reward, double[] array_future_state);


    /**
     * @brief add what learned from the current video to video memory
     *
     * @param array_state           array of double which indicates the current state
     * @param action                a int with got action
     * @param reward                a double with network's reward
     * @param array_future_state    array of double with future state
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void add_to_video_mem(double[] array_state, int action, double reward,double[] array_future_state);


    /**
     * @brief save current network into a file
     *
     * @param path      String which indicates file's path
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void save_model(String path);


    /**
     * @brief load a network from file
     *
     * @param path      String which indicates file's path
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void load_model(String path);


    /**
     * @brief close tensorflow session
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void close_session();


    /**
     * @brief kill Agent's python process
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void kill_python_process();



}
