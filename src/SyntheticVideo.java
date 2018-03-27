import interfaces.Video;

/**
 * Created by davidetalon on 14/04/17.
 */

public class SyntheticVideo implements Video{


    public static final int[] bitrates = {10000, 6000, 4000, 3000, 2000, 1000, 500, 300};
    private int nSegments;
    private int[] segmentQualityIndexes;
    private static double AVG_SCENE_DURATION = 5.0;
    private  double[][] complexityMatrix;


    public SyntheticVideo(int nSegments) {

        double[][] matrix = {{-0.0101529728434649, -0.0288832138324094, -0.0242726545067643, 0.00415396333169108, 0.999470864310074},
                {-0.0106444184411198, -0.0229079907856676, -0.0253096474216347, 0.000741787815715741, 0.999695979414017},
                {-0.0105082902276503, -0.0538481485732781, -0.0821086136160996, 0.0136133264030814, 1.00032754665181},
                {-0.00505349968198585, 0.00553965817491702, -0.0172601861523108, 0.000220312168207077, 0.999767453624563},
                {0.00997854854814642, 0.0759046797737938, -0.0113807478426485, 0.000398673897694183, 0.999835529217596}};

        complexityMatrix = matrix;

        this.nSegments = nSegments;

        int segment = 0;
        int old = 0;
        int scene = 0;

        segmentQualityIndexes = new int[nSegments];

        while (segment < nSegments) {

            while (scene == old){
                scene = (int) Math.floor(Math.random()*5);
            }
            old = scene;

            int sceneDuration = (int) Math.floor(exponential(AVG_SCENE_DURATION));

            if(sceneDuration > nSegments){
                sceneDuration = nSegments;
            }

            if(segment + sceneDuration > nSegments){
                sceneDuration = nSegments - segment;
            }


            for(int i = segment; i <= segment + sceneDuration -1; i++){
                segmentQualityIndexes[i] = scene;
            }

            segment = segment + sceneDuration;
        }



    }

    private double exponential(double mean) {

        double rand = Math.random();
        double x =  - (Math.log(rand)*mean);

        return x;
    }

    public int getNFrames() {
        return nSegments;
    }

    public int[] getBitrates(){
        return bitrates;
    }

    public int[] getSegmentComplexityIndexes(){
        return segmentQualityIndexes;
    }

    public double[][] getQualities() {
        double[][] qualities = new double[nSegments][bitrates.length];
        for (int i = 0; i < nSegments; i++) {
            for (int j = 0; j < bitrates.length; j++) {
                qualities[i][j] = qualityFunction(i, j);
            }
        }
        return qualities;
    }

    private double qualityFunction(int complexity, int bitrate) {
        double normalized = (double) bitrate/bitrates[0];
        double rsf = Math.log10(normalized);
        double[] videoQuality = new double[complexityMatrix[complexity].length];
        for(int i = 0; i < complexityMatrix[complexity].length; i++) {
            videoQuality[i] = complexityMatrix[complexity][i];
        }
        double quality = polyval(videoQuality, rsf);
        return quality;

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




}
