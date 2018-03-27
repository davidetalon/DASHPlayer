import interfaces.Agent;
import interfaces.NetworkSnap;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import py4j.GatewayServer;

import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by davidetalon on 13/05/17.
 */
public class MarkovDecisionProcess {

    private static int CHANNEL_SAMPLES = 5;
//    communication variable
    private GatewayServer server;

//    Variables declaration
    private Agent agent;

//    states
    private double[] state;
    private double[] prevState;
//    private double[] futureState;



    private double[][] complexityMatrix;
    private double[][] qualities;

//   algorithm loss function;
    private NetworkSnap networkSnap;
    private double loss;
    private double reward;
    private double prevQuality;
    private double quality;

    private double dashSegDuration;

    private static double MLP2_REBUFFERING_PENALTY_COEFF = 50.0;    //gamma
    private static double MLP2_FLUCTUATION_PENALTY_COEFF = 2.0;   //beta
    private static double MLP2_LOWBUFFER_PENALTY_COEFF = 0.001;
    private static double MLP2_SAFE_BUFFER = 10.0;
    private static String DEFAULT_MODEL_PATH = "../../Models/trained.ckpt";
    private int maxBitrate;

    private static final Logger LOGGER = Logger.getLogger( "MDPLog" );


    private double buffer;
    private double prevCapacity;
    private double capacity;
    private CircularFifoQueue<Double> capacitiesArray;


    public MarkovDecisionProcess(double dashSegDuration, int maxBitrate){

        this();

        this.dashSegDuration = dashSegDuration;
        this.maxBitrate = maxBitrate;
//        this.capacitiesArray = new CircularFifoQueue<Double>(CHANNEL_SAMPLES);



    }

    public MarkovDecisionProcess() {

        //initializing states
        state = new double[3 + CHANNEL_SAMPLES];
        prevState = new double[3 + CHANNEL_SAMPLES];

        this.loss = 0;
        networkSnap = new NetworkSnap();

        double[][] matrix = {{-0.0101529728434649, -0.0288832138324094, -0.0242726545067643, 0.00415396333169108, 0.999470864310074},
                {-0.0106444184411198, -0.0229079907856676, -0.0253096474216347, 0.000741787815715741, 0.999695979414017},
                {-0.0105082902276503, -0.0538481485732781, -0.0821086136160996, 0.0136133264030814, 1.00032754665181},
                {-0.00505349968198585, 0.00553965817491702, -0.0172601861523108, 0.000220312168207077, 0.999767453624563},
                {0.00997854854814642, 0.0759046797737938, -0.0113807478426485, 0.000398673897694183, 0.999835529217596}};

        complexityMatrix = matrix;
        this.capacitiesArray = new CircularFifoQueue<Double>(CHANNEL_SAMPLES);

    }



    public double addToNetworkMemory(boolean update){

        try {

            //add video to memory
            agent.add_to_video_mem(prevState, networkSnap.getAction(), reward, state);

            //updating network weights
            if(update){
                loss = agent.update(prevState, networkSnap.getAction(), reward, state);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return loss;

    }

    public int getNextAction(double epsilon, boolean eGreedy){

        if (eGreedy) {

            try {
                //choose next action with E-Greedy
                agent.choose_action_epsilon_greedy(state, epsilon, networkSnap);
            } catch (Exception e) {
                e.printStackTrace();
            }

        } else {

            try {

                //choose next action with Softmax policy
                agent.choose_action_softmax(state, epsilon, networkSnap);

            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        return networkSnap.getAction();

    }

    public void computeNextState(double lastChBitrate, int bitrateToDownload, int action, double downloadTime, int currentSegment, double wait) {

//        double complexity = state[3] * 5;
        double complexity = (int) (state[CHANNEL_SAMPLES + 1]);

        quality = qualityFunction(currentSegment, action);

        double[] rewardArray = new double[5];

        if (currentSegment > 0){
            rewardArray = reward(prevQuality, quality, downloadTime, buffer);
            reward = rewardArray[0];
        } else {
            reward = quality;
        }

//        prevCapacity = capacity;
//        capacity = lastChBitrate;
        capacitiesArray.add(lastChBitrate);

        double rebufferingTime = Math.max(0, downloadTime - buffer);

        buffer = Math.max(0, buffer - downloadTime) + dashSegDuration - wait;

        if (buffer > 20) {
            buffer = buffer - dashSegDuration + downloadTime;
        }

        double fluctuation = quality - prevQuality;




        //collecting qValues
        double[] qValuesArray = networkSnap.getOutput();
        String qValues = "";
        for(int i = 0; i < qValuesArray.length - 1; i++) {
            qValues +=  qValuesArray[i] + ", ";
        }
        qValues += qValuesArray[qValuesArray.length - 1];

        // array of stat    t | C_t, | a_t | B_t | D_t | r_t | q_t | r_t^q | r_t^b(1) - lowBuffer | r_t^b(2) - rebuffering pen| loss | Q-values

        String stateCapacities = "";
        for(int i = 0; i < CHANNEL_SAMPLES - 1; i++) {
            stateCapacities += capacitiesArray.get(i)+", ";
        }
        stateCapacities += capacitiesArray.get(CHANNEL_SAMPLES-1);



        String s = "{\"capacity\": "+capacitiesArray.get(CHANNEL_SAMPLES - 1)+", \"action\": "+action+", \"quality\": "+quality+", \"buffer\": "+buffer+", \"rebuffering_time\": "+rebufferingTime+", \"reward\": "+reward+", \"state\": ["+quality+", " + stateCapacities + ", " + complexity + ", " + buffer + "]}";
//        System.out.println(s);

//        LOGGER.log(Level.INFO, "{0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10}", stats);
        LOGGER.log(Level.INFO, s);
//            String[] statistiche = {String.valueOf(capacitiesArray.get(CHANNEL_SAMPLES-1)), String.valueOf(quality), String.valueOf(action)};
//        LOGGER.log(Level.INFO , "{Capacity: {0}, quality {1}, action {2}}", statistiche);


//        String[] qParams = {String.valueOf(state[3]), String.valueOf(quality), String.valueOf(fluctuation)};
//        String[] rParams = {String.valueOf(downloadTime), String.valueOf(state[4]), String.valueOf(reward)};
//        String[] sParams = {String.valueOf(state[0]), String.valueOf(state[1]), String.valueOf(state[2]),
//                String.valueOf(state[3]), String.valueOf(state[4]), String.valueOf(state[5]), String.valueOf(state[6]), String.valueOf(state[7])};
//        String[] nSParams = {String.valueOf(quality), String.valueOf(capacitiesArray.get(0)), String.valueOf(capacitiesArray.get(1)), String.valueOf(capacitiesArray.get(2)),String.valueOf(capacitiesArray.get(3)), String.valueOf(capacitiesArray.get(4)), String.valueOf(prevCapacity), String.valueOf(capacity),
//                String.valueOf(0), String.valueOf(buffer)};
//        String[] params = {String.valueOf(networkSnap.getAction()), String.valueOf(bitrateToDownload) ,String.valueOf(loss)};
//
//        LOGGER.log( Level.INFO, "State - PrevQuality:{0}, 5prevCh:{1}, 4prevCh:{2}, 3prevCh:{3}, 2prevCh:{4}, PrevCh:{5}, " +
//                "QualityIndex:{6}, Buffer: {7}", sParams);
////        LOGGER.log( Level.INFO, "\t\tQ-VALUES:{0}", aParams);
//        LOGGER.log( Level.INFO, "\t\tAction:{0},BitrateToDownload:{1}, Loss:{2}", params);
//
//        LOGGER.log( Level.INFO, "\t\tQualityIndex: {0}, SSIM:{1}, SSIMdelta:{2}", qParams);
//        LOGGER.log( Level.INFO, "\t\tTimeForSegDownload:{0}, BufferTime:{1}, Reward:{2}", rParams );
//        LOGGER.log( Level.INFO, "\t\tNEXT State - PrevQuality:{0}, 5prevCh:{1}, 4prevCh:{2}, 3prevCh:{3}, 2prevCh:{4}, PrevCh:{5}, " +
//                "QualityIndex:{6}, Buffer: {7}\n", nSParams);


    }

    public void moveNextState(int complexity, int currentSegment) {

        System.arraycopy(state, 0, prevState, 0, state.length);

        if( currentSegment > 0 && Math.max(currentSegment - 1,0) < 1) {
            for (int j = 0; j < CHANNEL_SAMPLES - 1; j++) {
                capacitiesArray.add(capacitiesArray.get(CHANNEL_SAMPLES - 1));
            }
        }

        prevQuality = quality;
        state[0] = quality;

        for (int i = 0; i < capacitiesArray.size(); i++) {
            state[i + 1] = capacitiesArray.get(i);
        }

        state[1 + CHANNEL_SAMPLES] = complexity;
        state[1 + CHANNEL_SAMPLES + 1] = buffer;

    }

    public void init() {
        Arrays.fill(prevState, 0);
        Arrays.fill(state, 0);
        capacitiesArray.clear();
        for(int i = 0; i < CHANNEL_SAMPLES; i++) {
            capacitiesArray.add(0.0);
        }


//        Arrays.fill(futureState, 0);

        networkSnap = new NetworkSnap();
        loss = 0;
        reward = 0;
        prevQuality = 0;
        quality = 0;

        buffer = 0;

    }

    /**
     * @brief get a quality paramater from action a_t and quality index D_t
     *
     * @param segment      an integer representing the current segment index
     * @param action an integer with selected action index
     *
     * @return a double with the segment's quality
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public double qualityFunction(int segment, int action) {
//        double normalized = (double) bitrate/maxBitrate;
//        double rsf = Math.log10(normalized);
//        double[] videoQuality = new double[complexityMatrix[qualityIndex].length];
//        for(int i = 0; i < complexityMatrix[qualityIndex].length; i++) {
//            videoQuality[i] = complexityMatrix[qualityIndex][i];
//        }
//        double quality = polyval(videoQuality, rsf);
//        return quality;
        return qualities[segment][action];
    }

    /**
     * @brief compute a polynomial on the point x
     *
     * @param coeff     array of index for the polynomial
     * @param x         where calculate the polynomial
     *
     * @return a double with the value of the polynomial function
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    private double polyval(double[] coeff, double x){

        double value = 0;
        for (int i = 0; i < coeff.length; i++) {
            value += Math.pow(x, coeff.length - i -1 ) * coeff[i];
        }

        if (value > 1){
            value = 1.0;
        }

        return value;
    }


    /**
     * @brief compute network's reward
     *
     * @param quality                   double, current quality
     * @param prevQuality               double, previous quality
     * @param segmentDownloadTime   double, estimated time for the next segment to download
     * @param playOutTime               double, time necessary to play buffered segments
     *
     * @return a double with network's reward
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    //compute network's reward
    private double[] reward (double prevQuality, double quality, double segmentDownloadTime, double playOutTime){

        double rebufferingTime = Math.max(0, segmentDownloadTime - playOutTime);
        double rebufferingPenalty = Math.min(MLP2_REBUFFERING_PENALTY_COEFF * rebufferingTime, 1);
        double qualityOscillationPenalty = MLP2_FLUCTUATION_PENALTY_COEFF * Math.abs(quality - prevQuality);

        double finalBuffer = Math.max(0, playOutTime - segmentDownloadTime);
        double lowbufferPenalty = MLP2_LOWBUFFER_PENALTY_COEFF * (Math.pow(Math.max(0, MLP2_SAFE_BUFFER - finalBuffer), 2));

        double reward = quality - qualityOscillationPenalty - rebufferingPenalty - lowbufferPenalty;

//        String[] rParams = {String.valueOf(rebufferingPenalty), String.valueOf(qualityOscillationPenalty), String.valueOf(lowbufferPenalty)};
//        LOGGER.log( Level.INFO, "\t\tREWARD - rebuffering:{0}, fluctuation:{1}, low buffer: {2}",rParams);

        //get reward's component array
        double[] rewardArray = new double[5];
        rewardArray[0] = reward;
        rewardArray[1] = quality;
        rewardArray[2] = qualityOscillationPenalty;
        rewardArray[3] = rebufferingPenalty;
        rewardArray[4] = lowbufferPenalty;

        return  rewardArray;

    }


    /**
     * @brief starts the python process with agent
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    //need to be improved and robuster
    public void startSession() throws IOException{

//        starts python agent thread
//        ProcessBuilder pb = new ProcessBuilder(Arrays.asList("bash", "./src/DashAlgorithm/tester/start_agent.sh"));
//        try {
//            Process p = pb.start();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

//        starts python agent thread
//        ProcessBuilder pb = new ProcessBuilder("./src/DashAlgorithm/tester/start_agent.sh");
//        ProcessBuilder pAgentb = new ProcessBuilder(Arrays.asList("python", "./src/DashAlgorithm/tester/python_lib/Agent.py"));

//        IOException
//        File log = new File("logdiprova.log");
//        pb.redirectErrorStream(true);
//        pb.redirectOutput(ProcessBuilder.Redirect.appendTo(log));
//        Process p = pb.start();
//        assert pb.redirectInput() == ProcessBuilder.Redirect.PIPE;
//        assert pb.redirectOutput().file() == log;
//        assert p.getInputStream().read() == -1;
//
        //initializing java-python gateway server
        server = new GatewayServer();
        server.start();

        //instance agent
        agent = (Agent) server.getPythonServerEntryPoint(new Class[]{Agent.class});

    }


    public void prebuffering(){

    }


    public void closeSession() throws IOException {
        server.shutdown();
//        agent.close_session();
//        agent.kill_python_process();
    }


    /**
     * @brief method that save the current trained model into a file.
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void saveTrainedModel() {
        agent.save_model(DEFAULT_MODEL_PATH);
    }


    /**
     * @brief method that load a trained model from a file.
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    public void loadTrainedModel(){
        agent.load_model(DEFAULT_MODEL_PATH);

    }

    public double getBuffer() {
        return buffer;
    }

    public double getReward(){
        return reward;
    }

    public double getQuality() {
        return quality;
    }

    public void setMaxBitrate(int maxBitrate) {
        this.maxBitrate = maxBitrate;
    }

    public void setDashSegDuration(double dashSegDuration) {
        this.dashSegDuration = dashSegDuration;
    }

    public void setQualities(double[][] qualities){
        this.qualities = qualities;
    }



}
