

/**
 * @file MLP2.java
 * @brief MLP2 algorithm for DDASH
 *
 * @author Davide Talon (<talon.davide@gmail.com>)
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */



import py4j.GatewayServer;
import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;


public class MLP2 extends DashAlgorithm {

    private static final Logger LOGGER = Logger.getLogger( MLP2.class.getName() );

//    communication variable
    private GatewayServer server;


    /**
     * state[0]     q_(t-1)     previous quality
     * state[1]     C_(t-2)     2 moments ago channel sample
     * state[2]     C_(t-1)     previous channel sample
     * state[3]     D_t         quality index
     * state[4]     B_t         playout time for buffered segment
     */


    private int segDuration;
    private int preBufferingBitRateIndex;

    private int[] dimensions;

    private int bitrateToDownload;
    private double reward;
    private double lastChBitrate;


    private static int DASH_SEG_DURATION = 2;

    private static int MLP2_PREBUFFERING_BITRATE_INDEX = 3;

    private static double MLP2_EGREEDY_EPSILON = 0.0;



    /**
     * @brief Default constructor for the class MLP2.
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    public MLP2(MediaListPlayer player, String tempFolderPath, String mpdUrl){

        super(player, tempFolderPath, mpdUrl);

        //start MDP session
        try {

            markovDP.startSession();

        } catch (Exception e) {
            e.printStackTrace();
        }

        markovDP.loadTrainedModel();

        segDuration = DASH_SEG_DURATION;
        preBufferingBitRateIndex = MLP2_PREBUFFERING_BITRATE_INDEX;

//        logging utils

        FileHandler fh = null;

        try {

            fh = new FileHandler("MLP2.log");
            SimpleFormatter formatter = new SimpleFormatter();

            fh.setFormatter(formatter);
            LOGGER.addHandler(fh);

        } catch (Exception e) {
            e.printStackTrace();
        }


//        LOGGER.setUseParentHandlers(false);

    }




    /**
     * @brief method that implements the Multilayer perceptron network with 2 hidden layers using Deep-Q Neural Network
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    @Override
    void getNextSeg() throws IOException, InterruptedException {

        if(current <= 0) {
            markovDP.init();
        }

        markovDP.moveNextState(complexities[current], current);

        if (current > 0) {
            double loss = 0;
            loss = markovDP.addToNetworkMemory(false);
        }
//
//        double reward = markovDP.getReward();
//        totalReward += reward;

        int action = markovDP.getNextAction(0, false);

        // setting segment url for download
        int bitrateToDownload = bitrates[action];

        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Integer.toString(action + 1)).replace("$Number$", Integer.toString(current + 1)));

        System.out.println("Downloading: " + segmentUrl);

        //setting previous channel sample and downloading
        lastChBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
                tempFolderPath + "init" + File.separator + (action +1 ) + "_init");

        double segDownloadTime = downloader.getLastSegmentDownloadTime();

        //adding downloaded file to buffer
        buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");

        System.out.println("Download: bitrate: "+(lastChBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());


        //setting next state s_(t+1)
        markovDP.computeNextState(lastChBitrate, bitrateToDownload, action, segDownloadTime, current, 0);

        if (current <= 0) {
//            player.play();
        }

//         If buffer is empty do a pre-buffering
        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
//            preBuffering();
            player.playItem(PlayerEventListener.segIndex - 1);
        }

        current++;

    }


    /**
     * @brief kill the python agent process
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    void closeMDPSession(){
        isInterrupted = true;

        try {
            markovDP.closeSession();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    /**
     * @brief Method that do the pre-buffering and fill the buffer with a defined number of segment and a defined quality.
     *
     * @author Davide Talon (<talon.davide@gmail.com>)
     * @version 1.0
     * @since 1.0
     */
    @Override
    void preBuffering() throws IOException {
//
//        bitrateToDownload = bitrates[preBufferingBitRateIndex];
//
//        for (int i = 0; i < nSegPrebuffer; i++) {
//            current++;
//            String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$Bandwidth$", Long.toString(bitrateToDownload)).replace("$Number$", Integer.toString(current)));
//
//            //preparing state for MLP2 and downloading pre-buffered segments
//            state[0] = qualityFunction(qualityIndexes[current], bitrateToDownload);
//            state[1] =state[2];
//            state[2] = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + current + ".mp4",
//                    tempFolderPath + "init" + File.separator + bitrateToDownload + "_init");
//            state[3] = qualityIndexes[current];
//            double segDownloadTime = downloader.getLastSegmentDownloadTime();
//            state[4] = Math.max(0, state[4] + (double) DASH_SEG_DURATION - segDownloadTime);
//
//            //adding segment to buffer
//            buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + Integer.toString(current) + ".mp4");
//
//
//            String[] params = {String.valueOf(state[0])};
//            LOGGER.log( Level.INFO, "[REBUFFERING - SSIM:{0}]", params );
////            System.out.println("[ Actual SSIM: "+ state[0] +", SSIM fluctuation: "+
////                    ", Obteined reward: " + 0 + ", Rebuffering: " + "REBUFFERING" +"]");
//
//        }

    }

    @Override
    void setDashSegDuration (double dashSegDuration) {
        markovDP.setDashSegDuration(dashSegDuration);
    }

    @Override
    void setMaxBitrate(int maxBitrate) {
        markovDP.setMaxBitrate(maxBitrate);
    }

    @Override
    void setQualities(double[][] qualities) {
        markovDP.setQualities(qualities);
    }

}
