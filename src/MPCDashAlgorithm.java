/**
 * @file MPCDashAlgorithm.java
 * @brief MPC adaptive algorithm for DASH standard
 *
 * @author Davide Talon (<talon.davide@gmail.com>)
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */



import org.apache.commons.collections4.queue.CircularFifoQueue;
import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;


public class MPCDashAlgorithm extends DashAlgorithm {


    // Variables declaration
    private int bitrateToDownload;
    private int bitRateIndex;
    private int horizonLength;
    private CircularFifoQueue<Double> previousCapacities;
    private CircularFifoQueue<Double> errors;
    private int action;
    private int pastAction;

    private static int HORIZON_LENGTH = 5;
    private static int HISTORY_LENGTH = 5;


    /**
     * Default construct for class BitRateBasedDashAlgorithm
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    public MPCDashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super(player, tempFolderPath, mpdUrl);
        this.horizonLength = HORIZON_LENGTH;
        this.previousCapacities = new CircularFifoQueue<Double>(HISTORY_LENGTH);
        this.errors = new CircularFifoQueue<Double>(HISTORY_LENGTH);

    }

    /**
     * Method that fills the buffer following the BitRate strategy.
     */
    @Override
    void getNextSeg() throws IOException {

        if (current <= 0) {
            markovDP.init();
        }

        markovDP.moveNextState(complexities[current], current);


        bufferPlotter.addDataToChart(current, markovDP.getBuffer(), 1);
        rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
        qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);

        //collecting previous capacities


        //choose bitrate

        ArrayList<Double> futureCapacities = new ArrayList<Double>(HORIZON_LENGTH);

        //first segment uses worst avaiable bitrate
        if (current <= 0) {
            action = bitrates.length - 1;
            errors.add(0.0);
        } else {

            pastAction = action;
            int steps = Math.min(horizonLength, parser.getNFrames() - current);
            //predict future capacities
            futureCapacities = getFutureCapacities(steps);

            //choose action which maximize total reward
            action = MPCDecision(futureCapacities, steps);

        }

        bitrateToDownload = bitrates[action];

        // Download the chosen file
//        printMessage("BITRATE_BASED: Actual bitrate index: " + (getNearestBitrate(lastBitrate) + 1) + " on " + parser.getBitrates().length);

        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Long.toString(action + 1)).replace("$Number$", Integer.toString(current + 1)));

        System.out.println("SEGMENT URL " + segmentUrl);

        lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
                tempFolderPath + "init" + File.separator + (action + 1) + "_init");

        //add measured capacity to old capacities
        previousCapacities.add(lastBitrate);

        //insert absolute error percentage
        if (current > 0) {
            errors.add(Math.max(0, 1 - lastBitrate/futureCapacities.get(0)));
        }



        double segDownloadTime = downloader.getLastSegmentDownloadTime();
        System.out.println("Download: bitrate: "+(lastBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());

//        System.out.println("DEST FILE PATH: " + tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");
//        System.out.println("HEADER: " + tempFolderPath + "init" + File.separator + (action + 1) + "_init");

        buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");

        markovDP.computeNextState(lastBitrate, bitrateToDownload, action, segDownloadTime, current, 0);

        current++;


        // If buffer is empty do a pre-buffering

        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
            bitRateIndex = bitrates.length - 1;
//            player.playItem(PlayerEventListener.segIndex - 1);
            System.out.println("REBUFFERING EVENT OCCOURS");

        }
    }

    private ArrayList<Double> getFutureCapacities(int steps) {

        //compute max absolute percentage error
        double err = Collections.max(errors);
//        double err = 0;

        CircularFifoQueue<Double> oldCapacities = new CircularFifoQueue<Double>(HORIZON_LENGTH);

        for(int i = 0; i < previousCapacities.size(); i++) {
            oldCapacities.add(previousCapacities.get(i));
        }

        ArrayList<Double> predictedCapacities = new ArrayList<Double>(HORIZON_LENGTH);
        //predict using harmonic mean
        for (int i = 0; i < steps; i++) {
            double predicted = harmonicMean(oldCapacities) / (1 + err);
            predictedCapacities.add(predicted);

//            String s  = "";
//            for (int j = 0; j < oldCapacities.size(); j++) {
//                s+=oldCapacities.get(j) + ", ";
//
//            }
//            System.out.println("Old ch: " + s);

            oldCapacities.add(predicted);

        }

        return predictedCapacities;
    }


    private int MPCDecision(ArrayList<Double> futureCapacities, int steps) {

        //dynamic programming
        double gamma = 50;
        int actions = bitrates.length;
        int nStrategies = (int) Math.pow(actions, steps);
        int bestAction = bitrates.length - 1;
        double maxReward = Double.NEGATIVE_INFINITY;
        double quality = 0;


        for (int i = 0; i < nStrategies; i++) {

            double reward = 0;
            double oldQuality = quality;
            double playOutTime = markovDP.getBuffer();
            int[] chosenActions = new int[steps];
            for (int j = 0; j < steps; j++) {

                chosenActions[j] = (int)Math.floor((i % (int) Math.pow(actions, j+1)) / (int)Math.pow(actions, j));
                quality = markovDP.qualityFunction(current + j, chosenActions[j]);

                double rebufferingTime = Math.max(0, 2 * bitrates[chosenActions[j]] / futureCapacities.get(j) - playOutTime);
                playOutTime = Math.max(0, playOutTime - 2 * bitrates[chosenActions[j]] / futureCapacities.get(j)) + 2;
                reward = reward + quality - Math.abs(quality - oldQuality) - Math.min(1, rebufferingTime * gamma);
            }

            if (reward >= maxReward) {
                maxReward = reward;
                bestAction = chosenActions[0];

            }

        }

        return bestAction;
    }


    /**
     * Method that do the pre-buffering and fill the buffer with a defined number of segment and a defined quality.
     */
    @Override
    void preBuffering() throws IOException {
    }

    /**
     * @brief Close the MDP Session
     */
    @Override
    void closeMDPSession() {
        isInterrupted = true;
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

