/**
 * @file NetworkSnap.java
 * @brief provide a network snap containing action and network's output layer
 *
 * @author Davide Talon (<talon.davide@gmail.com>)
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */

package interfaces;



public class NetworkSnap {

    //public to allow python to access
    public int action;
    public double[] output;
    static int OUTPUT_LENGTH = 8;

    /**
     * @brief public contructor for NetworkSnap
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public NetworkSnap() {
        this.output = new double[OUTPUT_LENGTH];
    }


    /**
     * @brief public contructor for NetworkSnap
     *
     * @param action       int with next action
     * @param output       array of double which indicates network's output
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public NetworkSnap(int action, double[] output) {
        this.action = action;
        this.output = new double[output.length];

        System.arraycopy(output, 0, this.output, 0, output.length);

    }


    /**
     * @brief get network action
     *
     * @return next action
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public int getAction() {
        return action;
    }


    /**
     * @brief get network output
     *
     * @return computed network's output
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public double[] getOutput() {
        return output;
    }


    /**
     * @brief set network action
     *
     * @param newAction     int which indicates a new action
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
//    public void setAction(int newAction) {
//        action = newAction;
//    }

    /**
     * @brief set network output
     *
     * @param newOutput     array of double for the new network
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
//    public void setOutput(double[] newOutput) {
//        output = newOutput;
//    }

    /**
     * @brief transform NetworkSnap into a string
     *
     * @return      a string with action and network's output
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public String toString() {
        String s = "[";
        for (int i = 0; i < OUTPUT_LENGTH - 1; i++) {
            s += output[i] + " ,";
        }
        s += output[OUTPUT_LENGTH -1] + "], " + action;

        return s;
    }


}
