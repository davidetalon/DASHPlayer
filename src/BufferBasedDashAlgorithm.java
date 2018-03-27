/**
 * @file BufferBasedDashAlgorithm.java
 * @brief Class the implement the BufferBased algorithm for DASH.
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


import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;

/**
 * Class the implement the BufferBased algorithm for DASH.
 */


public class BufferBasedDashAlgorithm extends DashAlgorithm {

    // Variables declaration
    private int bitrateToDownload;
    private int bitRateIndex;
    private int maxBufferDimension;


    /**
     * @brief Default construct for class BufferBasedDashAlgorithm
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    public BufferBasedDashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super(player, tempFolderPath, mpdUrl);
        bitRateIndex = 6;
        maxBufferDimension = 10;
    }

    /**
     * @brief Method that fills the buffer following the Buffer strategy.
     */
    @Override
    void getNextSeg() throws IOException {

        if (current <= 0) {
            markovDP.init();
        }

        // Wait until the buffer is full
        double wait = Math.max(0, bufferDimension()-maxBufferDimension);
            try {
                Thread.currentThread().sleep((long)wait);
            } catch (InterruptedException e) {
                System.err.println(e.getMessage());
            }



        markovDP.moveNextState(complexities[current], current);
//        LOGGER.log(Level.INFO, "Segment{0}", String.valueOf(current));

        bufferPlotter.addDataToChart((current), markovDP.getBuffer(), 1);
        rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
        qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);

        // Download the chosen file
        bitrateToDownload = bitrates[bitRateIndex];
        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Integer.toString(bitRateIndex + 1)).replace("$Number$", Integer.toString(current + 1)));
        lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
                tempFolderPath + "init" + File.separator + (bitRateIndex + 1) + "_init");

        System.out.println("URL: " + segmentUrl);
        System.out.println(("HEADER:"+ tempFolderPath + "init" + File.separator + (bitRateIndex + 1) + "_init"));
        System.out.println("Dest: " + tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");

        buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");

        printMessage("BUFFER_BASED: Actual bitrate index: " + (bitRateIndex + 1) + " on 20");

        //Choose the next video quality
        bitRateIndex = bufferDimension() * bitrates.length / maxBufferDimension;
        System.out.println(bitRateIndex);

        //get download time
        double segDownloadTime = downloader.getLastSegmentDownloadTime();
        System.out.println(segDownloadTime);

        //setting next state s_(t+1)
        markovDP.computeNextState(lastBitrate, bitrateToDownload, bitRateIndex, segDownloadTime, current, wait);

        current++;

        // If buffer is empty do a pre-buffering
        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
            bitRateIndex = 6;
//            preBuffering();
            player.playItem(PlayerEventListener.segIndex - 1);
        }

    }

    /**
     * Method that do the pre-buffering and fill the buffer with a defined number of segment and a defined quality.
     */
    @Override
    void preBuffering() throws IOException {
//        for (int i = 0; i < nSegPrebuffer && current < parser.getNFrames(); i++) {
//
//            System.out.println("Prebuffering: ");
//
//            markovDP.moveNextState(complexities[current], current);
////            LOGGER.log(Level.INFO, "Segment{0}", String.valueOf(current));
//
//            bufferPlotter.addDataToChart((current), markovDP.getBuffer(), 1);
//            rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
//            qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);
//
//            bitrateToDownload = bitrates[bitRateIndex];
//
//            String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$", Long.toString(bitRateIndex + 1)).replace("$Number$", Integer.toString(current + 1)));
//            lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
//                    tempFolderPath + "init" + File.separator + (bitRateIndex+1) + "_init");
//
//            double segDownloadTime = downloader.getLastSegmentDownloadTime();
//
//            System.out.println("URL: " + segmentUrl);
//            System.out.println("Dest: " + tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");
//            buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + Integer.toString(current + 1) + ".mp4");
//
//            markovDP.computeNextState(lastBitrate, bitrateToDownload, segDownloadTime, current);
//
//            current++;
//        }
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
