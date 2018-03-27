

/**
 * Created by davidetalon on 12/04/17.
 */
public class MarkovChannel {


    private static int[] capacityLevels = {300, 500, 1000, 2000, 3000, 4000, 5000, 6500, 8000, 10000, 15000};
    private static int[] initialLevels = {400, 750, 1500, 2500, 3500, 4500, 5750, 7250, 9000, 12500};
    private double channelCapacity;
    private int state;
    private double[][] transitionMatrix;


    public MarkovChannel() {


        int initial = initialLevels[(int)Math.floor(Math.random()*10)];
//        System.out.println(initial);
//        find initial state
        this.state = quantize(initial, capacityLevels);
        this.channelCapacity = capacityLevels[state];

        transitionMatrix = new double[initialLevels.length][initialLevels.length];
        double p = 1.0;
        double rho = 1.0;
        double[] state_p = new double[2];
        state_p[0] = (double)2/3;
        state_p[1] = (double)1/3;
        for (int i = 0; i < initialLevels.length; i++){
            double sum = 0;
            for(int j = 0; j < initialLevels.length; j++) {
                double pup = (double) p / (1 + rho);
                double pdown = (double) p * rho / (1 + rho);
                if (i == j + 2) {
                    transitionMatrix[i][j] = pup * state_p[1];
                } else if (i == j + 1) {
                    transitionMatrix[i][j] = pup * state_p[0];
                } else if (i == j - 2) {
                    transitionMatrix[i][j] = pdown * state_p[1];
                } else if (i == j - 1){
                    transitionMatrix[i][j] = pdown * state_p[0];
                }
                sum += transitionMatrix[i][j] ;
            }
            transitionMatrix[i][i] = 1 - sum;
//            if (transitionMatrix[i][i] < 0.000001) {
//                transitionMatrix[i][i] = 0;
//            }
        }

//        for(int riga = 0; riga < initialLevels.length; riga++) {
//            String rowval="";
//            for(int col = 0; col < initialLevels.length; col++) {
//                rowval += "," + transitionMatrix[riga][col];
//            }
//            System.out.println(rowval);
//        }

    }


    public void changeChannelCapacity(){

        double event= Math.random();
        boolean transition = false;
        for (int i = 0 ; !transition && i < 8; i++) {
            event = event - transitionMatrix[state][i];
            if(event <= 0) {
                state = i;
                channelCapacity = (capacityLevels[i] + capacityLevels[i + 1]) / 2;
                transition = true;
            }
        }

    }


    public double getChannelCapacity(){
        return channelCapacity;
    }

    private int quantize(int initial, int[] capacityLevels) {

        int level = 1;
        while(level < capacityLevels.length - 2 && initial > capacityLevels[level]){
            level++;
        }

        return level;
    }
}
