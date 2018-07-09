/**
 * @file PANDADashAlgorithm.java
 * @brief Class the implement the PANDA algorithm for DASH.
 *
 * @author Davide Talon
 *
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */


import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class PANDADashAlgorithm extends DashAlgorithm {

    // Variables declaration
    private int bitrateToDownload;
    private int action;
    private double segDownloadTime;
    private ArrayList<Double> normalizedBitrates;
    private double[] pandaState;
    private double[] prevPanda;
    private double wait;

    private static double K_PARAMETER = 0.42;
    private static double W_PARAMETER = 0.3;
    private static double ALPHA_PARAMETER = 0.3;
    private static double BETA_PARAMETER = 0.3;
    private static double EPSILON_PARAMETER = 0.2;
    private static double MIN_BUFFER = 18;


    /**
     * Default construct for class BitRateBasedDashAlgorithm
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    public PANDADashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super(player, tempFolderPath, mpdUrl);

//        normalizedBitrates = new ArrayList<Double>();
//        for (int i = 0; i < bitrates.length; i++) {
//            normalizedBitrates.set(i, bitrates[i] / 1000.0);
//        }

        pandaState = new double[4];
        prevPanda = new double[4];
        segDownloadTime = 0;
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


//        System.arraycopy(pandaState, 0, prevPanda, 0, 4);

        // Download the chosen file
        if (current <= 0) {

            //initialize panda state
            System.out.println("LAST BITRATE: " + lastBitrate);
            pandaState[0] = bitrates[bitrates.length / 2]/1000.0;
            pandaState[1] = bitrates[bitrates.length /2]/1000.0;
            pandaState[2] = 0.0;
            pandaState[3] = (double)(bitrates.length / 2);

            normalizedBitrates = new ArrayList<Double>(bitrates.length);
            for (int i = 0; i < bitrates.length; i++) {
                normalizedBitrates.add(bitrates[i]/1000.0);
            }

//            action = bitrates.length - 1;
            action = (int) pandaState[3];
            wait = 0;

//            System.arraycopy(pandaState, 0, prevPanda, 0, 4);

        } else {

            System.arraycopy(pandaState, 0, prevPanda, 0, 4);

            pandaState = pandaDecision();
            action = (int)pandaState[3];

            wait = pandaState[2];
        }

        String s = "";
        String ps = "";
        for( int i = 0; i < pandaState.length; i++) {
            s += pandaState[i] + ", ";
            ps += prevPanda[i] + ", ";
        }

        System.out.println(ps);
        System.out.println(s);


        //moving panda state to prevPanda
//        System.arraycopy(pandaState, 0, prevPanda, 0, 4);

        printMessage("BITRATE_BASED: Actual bitrate index: " + (action + 1) + " on " + bitrates.length);

        //specifying URL
        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Long.toString(action + 1)).replace("$Number$", Integer.toString(current + 1)));

        System.out.println("SEGMENT URL - " + segmentUrl);

        //downloading segment
        lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
                tempFolderPath + "init" + File.separator + (action + 1) + "_init");

        segDownloadTime = downloader.getLastSegmentDownloadTime();

        //adding downloaded segment to buffer
        buffer.addMedia( tempFolderPath + "seg" + File.separator + "seg" + (current + 1 )+ ".mp4");

        System.out.println("Download: bitrate: "+(lastBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());

        if (wait > 0) {
            wait = Math.min(wait, markovDP.getBuffer());
            try {
                Thread.currentThread().sleep((long) wait);
            } catch (InterruptedException e) {
                System.err.println(e.getMessage());
            }
        }


        markovDP.computeNextState(lastBitrate, bitrates[action], action,segDownloadTime, current, wait);



        current++;



        // If buffer is empty do a pre-buffering

        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
//            bitRateIndex = bitrates.length - 1;
//            player.playItem(PlayerEventListener.segIndex - 1);
            System.out.println("REBUFFERING");

        }
    }

    /**
     * Method that do the pre-buffering and fill the buffer with a defined number of segment and a defined quality.
     */
    @Override
    void preBuffering() throws IOException {

//        bitRateIndex = bitrates.length - 1;
//
//        bitrateToDownload = bitrates[bitRateIndex];
//
//
//
//        for (int i = 0; i < nSegPrebuffer; i++) {
//            System.out.println("PREBUFFERING");
//
//            markovDP.moveNextState(complexities[current], current);
//
//            bufferPlotter.addDataToChart(current, markovDP.getBuffer(), 1);
//            rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
//            qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);
//
//            String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
//                    Long.toString(bitRateIndex + 1 )).replace("$Number$", Integer.toString(current + 1)));
//
//            System.out.println(segmentUrl);
//
//            lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (i + 1) + ".mp4",
//                    tempFolderPath + "init" + File.separator + (bitRateIndex + 1) + "_init");
//
//            double segDownloadTime = downloader.getLastSegmentDownloadTime();
//            System.out.println(segDownloadTime);
//
//
//            System.out.println("DEST FILE PATH: " + tempFolderPath + "seg" + File.separator + "seg" + (i + 1) + ".mp4");
//            System.out.println("HEADER: " + tempFolderPath + "init" + File.separator + (bitRateIndex + 1) + "_init");
//
//            buffer.addMedia( tempFolderPath + "seg" + File.separator + "seg" + Integer.toString(current + 1) + ".mp4");
//            bitrateToDownload = bitrates[Math.max(getNearestBitrate(lastBitrate) - 3, 3)];
//
//            markovDP.computeNextState(lastBitrate, bitrateToDownload, segDownloadTime, current);
//
//            current++;
//        }
    }

    private double[] pandaDecision () {

        double[] newPanda = new double[4];

        //estimate bandwidth share
        double newX = prevPanda[0] + (segDownloadTime + wait) * K_PARAMETER * (W_PARAMETER -
                Math.max(0, prevPanda[0] - lastBitrate/1000 + W_PARAMETER));

        newPanda[0] = newX;

        System.out.println("Smoothed: "+ newX);

        //smooth out x
        double newY = prevPanda[1] - (segDownloadTime + wait) * ALPHA_PARAMETER * (prevPanda[1] - newX);
        newPanda[1] = newY;

        System.out.println("Smoothed: "+ newY);

        //quantize y
        double delta_up = EPSILON_PARAMETER * newPanda[1];
        double delta_down = 0;

        //dead-zone quantizer
        int upIndex = bitrates.length - 1;
        while (upIndex > 0 && normalizedBitrates.get(upIndex - 1) < (newY - delta_up)) {
            upIndex--;
        }

        int downIndex = bitrates.length - 1;
        while(downIndex > 0 && normalizedBitrates.get(downIndex - 1) < (newY - delta_down)) {
            downIndex--;
        }

        System.out.println("UP treeshold: " + (newY - delta_up) + ", " + normalizedBitrates.get(upIndex));
        System.out.println("DOWN treeshold: " + (newY - delta_down) + ", " + normalizedBitrates.get(downIndex));

        newPanda[3]  = (double)downIndex;

        if(normalizedBitrates.get((int)prevPanda[3]) < normalizedBitrates.get(upIndex)) {
            newPanda[3] = (double)upIndex;
        }

        if(normalizedBitrates.get(upIndex) <= normalizedBitrates.get((int)prevPanda[3]) && normalizedBitrates.get((int)prevPanda[3]) <= normalizedBitrates.get((downIndex))) {
            newPanda[3] = prevPanda[3];
        }

        double buffer = Math.max(0, markovDP.getBuffer() - segDownloadTime) + 2;

        double delay = BETA_PARAMETER * (buffer - MIN_BUFFER);

        if (delay > 0) {
            delay = Math.min(delay, buffer);
        }

        newPanda[2] = delay;

        return newPanda;
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
