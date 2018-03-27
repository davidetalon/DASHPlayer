/**
 * @file DashAlgorithm.java
 * @brief Class that defines an abstract behaviour of a DashAlgorithm
 *
 * @author Iacopo Mandatelli
 * @author Matteo Biasetton
 * @author Luca Piazzon
 *
 * @version 1.0
 * @since 1.0
 *
 * @copyright Copyright (c) 2017-2018
 * @copyright Apache License, Version 2.0
 */

import org.apache.commons.collections4.queue.CircularFifoQueue;
import uk.co.caprica.vlcj.medialist.MediaList;
import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public abstract class DashAlgorithm extends Thread {

    private static int number = 0;
    //Constants declaration
    private static String MPD_FILENAME = "mpdList.mpd";


    // Variables declaration
    public boolean isInterrupted;
    int current;
    int nSegPrebuffer;
    double lastBitrate;
    FileDownloader downloader;
    static MediaListPlayer player;
    MediaList buffer;
    MPDParser parser;
    String mpdUrl;
    String tempFolderPath;
    int[] bitrates;
    int[] complexities;
    double[][] qualities;
    private String baseUrl;
    Plotter bufferPlotter;
    Plotter rewardPlotter;
    Plotter qualityPlotter;

    MarkovDecisionProcess markovDP;


    /**
     * Public constructor for class DashAlgorithm.
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    DashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super();
        this.player = player;
        buffer = player.getMediaList();
        downloader = new FileDownloader();
        this.mpdUrl = mpdUrl;
        this.tempFolderPath = tempFolderPath;
        baseUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1);
        current = 0;
        nSegPrebuffer = 3;
        number++;
        isInterrupted = false;

        markovDP = new MarkovDecisionProcess();
    }

    /**
     * Abstract method to be implemented differently by many version of DASH algorithm.
     */
    abstract void getNextSeg() throws IOException, InterruptedException;

    abstract void setDashSegDuration(double dashSegDuration);

    abstract void setMaxBitrate(int maxBitrate);

    abstract void setQualities(double[][] qualities);


    /**
     * Abstract method to be implemented differently by many version of DASH algorithm.
     */
    abstract void preBuffering() throws IOException;

    /**
     * @brief Close the MDP Session
     */
    abstract void closeMDPSession();

    /**
     * Method called to run code in background
     */
    @Override
    public void run() {

        try {
            lastBitrate = downloader.downloadFile(mpdUrl, tempFolderPath + MPD_FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Download all the header files
        parser = new MPDParser(tempFolderPath + MPD_FILENAME);
        bitrates = parser.getBitrates();


        complexities = parser.getSegmentComplexityIndexes();
        qualities = parser.getQualities();

        String s = "";
        for(int i = 0; i < bitrates.length; i++) {
            s+=bitrates[i] + ", ";
        }
        System.out.println(s);

        setMaxBitrate(bitrates[0]);
        setDashSegDuration(parser.getSegmentDuration());
        setQualities(qualities);

        String initTemplate = parser.getInitialization();
        for (int i = 1; i <= bitrates.length; i++) {
            try {
                lastBitrate = downloader.downloadFile(baseUrl + initTemplate.replace("$RepresentationID$", i + ""), tempFolderPath + File.separator + "init" + File.separator + i + "_init");
            } catch (IOException e) {
                System.err.println(e.getMessage());
            }
        }

        // Run algorithm
        try {
            preBuffering();
//            player.play();
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        while (!isInterrupted && current < parser.getNFrames()) {
            try {
                getNextSeg();
                printMessage("PLAYER: buffer dimension = " + bufferDimension());
            } catch (IOException e) {
                System.err.println(e.getMessage());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        try {

            closeMDPSession();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Return the index of the bitrate's array whit the higher bit rate <= to the given bit rate. If no index is found returns 0.
     *
     * @param lastBitRate last calculated bitrate
     * @return nearest bitrate available to download
     */
    protected int getNearestBitrate(double lastBitRate) {
        if (lastBitRate > 0) {
            for (int i = bitrates.length - 1; i > 0; i--) {
                if (bitrates[i] > lastBitrate) {
                    return i;
                }
            }
        }
        return 0;
    }

    /**
     * Utility function that calculates the Harmonic Mean.
     *
     * @param measurements an arraylist of double values, in our case the measured bandwidth.
     * @return the harmonic mean of measurements.
     */
    double harmonicMean(ArrayList<Double> measurements) {
        double sum = 0;
        for (int i = 0; i < measurements.size(); i++) {
            sum += (1 / measurements.get(i));
        }
        return measurements.size() / sum;
    }

    double harmonicMean(CircularFifoQueue<Double> measurements) {
        double sum = 0;
        for (int i = 0; i < measurements.size(); i++) {
            sum += (1 / measurements.get(i));
        }
        return measurements.size() / sum;

    }


    /**
     * Method that calculates the current dimension of the buffer
     *
     * @return the dimension of the buffer
     */
    public static int bufferDimension() {
        if ((player.getMediaList().size() > 0) && (PlayerEventListener.segIndex > 0)) {
            return (player.getMediaList().size() - PlayerEventListener.segIndex);
        }
        return 0;
    }

    public void setPlotters(Plotter bufferPlotter, Plotter qualityPlotter, Plotter rewardPlotter) {
        this.bufferPlotter = bufferPlotter;
        this.qualityPlotter = qualityPlotter;
        this.rewardPlotter = rewardPlotter;
    }

    /**
     * Stops the DASH Tread
     */
//    public void forceInterrupt() {
//        isInterrupted = true;
//
//        try {
//            markovDP.closeSession();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//
//        stop();
//    }

    /**
     * Append a new message to the messages window and save the message to the log.
     *
     * @param message text to append
     */
    public void printMessage(String message) {
        Player.addMessage(message);
        Player.addLog(message);
    }
}