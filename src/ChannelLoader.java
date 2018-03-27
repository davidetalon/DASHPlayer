import exceptions.InsufficientVideoRealTracesException;
import interfaces.Channel;

import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

/**
 * Created by davidetalon on 26/07/17.
 */


public class ChannelLoader implements Channel{


    private static String DEFAULT_REAL_CHANNEL = "Models/real_traces/belgium.txt";
    private static String PRETRAIN_CHANNEL = "Models/real_traces/pretrain_channel.txt";
    private int MAX_N_SEGMENT = 400;
    private int MAX_N_VIDEO = 1000;
    private double[] capacities;
    private int currentSample;
    private int bitrate;



    ChannelLoader(int nSample, boolean pretrain) throws InsufficientVideoRealTracesException{

        if (nSample > MAX_N_SEGMENT * MAX_N_VIDEO){
            throw new InsufficientVideoRealTracesException();
        } else {

            this.capacities = new double[nSample];
            this.currentSample = 0;
            int i = 0;


            try {

                FileReader fileReader = null;

                if (pretrain) {
                    fileReader = new FileReader(PRETRAIN_CHANNEL);
                } else {
                    fileReader = new FileReader(DEFAULT_REAL_CHANNEL);
                }


                Scanner in = new Scanner(fileReader);

                while (in.hasNext() && i < nSample) {
                    capacities[i] = Double.parseDouble(in.next());
                    i++;
                }


                in.close();

            } catch (Exception e) {
                e.printStackTrace();
            }

            String s = "";
        }

    }


    public double download(int bitrate){

        this.bitrate = bitrate;

        return capacities[currentSample];
    }

    public void changeChannelCapacity(){
        currentSample++;
    }

    public double getLastSegmentDownloadTime(){

        return bitrate * 2 / capacities[currentSample];

    }

    public double downloadFile(String url, String destFilePath, String header) throws IOException {
        return 0;
    }

    public double downloadFile(String url, String destFilePath) throws IOException {
        return 0;
    }
}
