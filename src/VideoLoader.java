import exceptions.InsufficientVideoRealTracesException;
import interfaces.Video;

import java.io.FileReader;
import java.util.Scanner;

/**
 * Created by davidetalon on 26/07/17.
 */
public class VideoLoader implements Video{

    public static final int[] bitrates = {10000, 6000, 4000, 3000, 2000, 1000, 500, 300};
    private int[][] complexities;
    private static String DEFAULT_REAL_VIDEO_COMPLEXITIES = "Models/real_traces/Videos_complexities.txt";
    private static String PRETRAIN_VIDEO_COMPLEXITIES = "Models/real_traces/pretrain_video_complexities.txt";
    private static int MAX_N_SEGMENT = 400;
    private static int MAX_N_VIDEO = 100;
    private int segments;
    private int nVideo;
    private int currentVideo;
    private double[][] complexityMatrix;





    public VideoLoader(int segments, int nVideo, boolean pretrain) throws InsufficientVideoRealTracesException{

        double[][] matrix = {{-0.0101529728434649, -0.0288832138324094, -0.0242726545067643, 0.00415396333169108, 0.999470864310074},
                {-0.0106444184411198, -0.0229079907856676, -0.0253096474216347, 0.000741787815715741, 0.999695979414017},
                {-0.0105082902276503, -0.0538481485732781, -0.0821086136160996, 0.0136133264030814, 1.00032754665181},
                {-0.00505349968198585, 0.00553965817491702, -0.0172601861523108, 0.000220312168207077, 0.999767453624563},
                {0.00997854854814642, 0.0759046797737938, -0.0113807478426485, 0.000398673897694183, 0.999835529217596}};

        complexityMatrix = matrix;

        if (segments > MAX_N_SEGMENT || MAX_N_VIDEO > 100){
             throw new InsufficientVideoRealTracesException();
        } else {

            this.segments = segments;
            this.nVideo = nVideo;
            this.complexities = new int[nVideo][segments];
            this.currentVideo = 0;

            try {

                FileReader fileReader = null;
                if (pretrain) {
                   fileReader = new FileReader(PRETRAIN_VIDEO_COMPLEXITIES);
                } else {
                    fileReader = new FileReader(DEFAULT_REAL_VIDEO_COMPLEXITIES);
                }

                Scanner in = new Scanner(fileReader);

                int i = 0;

                while (in.hasNextLine() && i < nVideo) {
                    String video = in.nextLine();
                    Scanner line = new Scanner(video);

                    int j = 0;
                    while (line.hasNext() && j < segments) {
                        complexities[i][j] = Integer.parseInt(line.next()) - 1 ;
                        j++;
                    }
                    i++;
                }

                in.close();

            } catch (Exception e) {
                e.printStackTrace();
            }


//            for (int j = 0; j < nVideo; j++) {
//                String s = "";
//                for (int k = 0; k < segments; k++) {
//                    s += complexities[j][k] + ",";
//                }
//                System.out.println(s);
//            }
        }
    }

    public int[] getBitrates(){
        return bitrates;
    }


    public int[] getSegmentComplexityIndexes() {

        int[] videoComplexities = new int[segments];

        for (int i = 0; i < segments; i++) {
            videoComplexities[i] = complexities[currentVideo][i];
        }

        currentVideo++;

        return videoComplexities;
    }

    public int getNFrames() {
        return segments;
    }

    public double[][] getQualities() {
        double[][] qualities = new double[segments][bitrates.length];
        for (int i = 0; i < segments; i++) {
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
