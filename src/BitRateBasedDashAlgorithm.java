/**
 * @file BitRateBasedDashAlgorithm.java
 * @brief Class the implement the BitRateBased algorithm for DASH.
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


public class BitRateBasedDashAlgorithm extends DashAlgorithm {

    // Variables declaration
    private int bitrateToDownload;
    private int bitRateIndex;


    /**
     * Default construct for class BitRateBasedDashAlgorithm
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    public BitRateBasedDashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super(player, tempFolderPath, mpdUrl);
    }


    /**
     * @brief Method that fills the buffer following the BitRate strategy.
     */
    @Override
    void getNextSeg() throws IOException {

        if (current <= 0) {
            markovDP.init();
        }

        markovDP.moveNextState(complexities[current], current);

        //plotting points
        bufferPlotter.addDataToChart(current, markovDP.getBuffer(), 1);
        rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
        qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);


        //select next action using the rate below the last channel sample
        int action = bitrates.length - 1;
        if (current > 0) {
            while (action > 0 && bitrates[action - 1] <=  lastBitrate) {
                action = action - 1;
            }
        }

        printMessage("BITRATE_BASED: Actual bitrate index: " + (getNearestBitrate(lastBitrate) + 1) + " on 33");

        //getting segment url
        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Long.toString(action + 1)).replace("$Number$", Integer.toString(current + 1)));

        System.out.println("SEGMENT URL "+segmentUrl);

        //downloading file
        lastBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4",
                tempFolderPath + "init" + File.separator + (action + 1) + "_init");


        //get last segment download time
        double segDownloadTime = downloader.getLastSegmentDownloadTime();
        System.out.println("Download: bitrate: "+(lastBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());

        //add chunk to playout buffer
        buffer.addMedia( tempFolderPath + "seg" + File.separator + "seg" + (current + 1 )+ ".mp4");

        //compute next MDP state
        markovDP.computeNextState(lastBitrate, bitrateToDownload, action, segDownloadTime, current, 0);

        current++;



        // If buffer is empty do a pre-buffering

        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
            bitRateIndex = bitrates.length - 1;
//            preBuffering();
            player.playItem(PlayerEventListener.segIndex - 1);
            System.out.println("REBUFFERING");

        }
    }


    /**
     * @brief Method that do the pre-buffering and fill the buffer with a defined number of segment and a defined quality.
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
//            markovDP.computeNextState(lastBitrate, bitrateToDownload, segDownloadTime, current, wait);
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

    /**
     * @brief Set Markov Decision Process' segment duration
     *
     *  @param  dashSegDuration     double with video's segment duration
     */
    @Override
    void setDashSegDuration (double dashSegDuration) {
        markovDP.setDashSegDuration(dashSegDuration);
    }

    /**
     * @brief Set Markov Decision Process' max avaiable bitrate
     *
     * @param maxBitrate    int with max avaiable rapresentation
     */
    @Override
    void setMaxBitrate(int maxBitrate) {
        markovDP.setMaxBitrate(maxBitrate);
    }

    /**
     * @brief Set Markov Decision Process' segment qualities
     *
     * @param qualities     Matrix of double containing qualities to pass to Markov Decision Process
     */
    @Override
    void setQualities(double[][] qualities) {
        markovDP.setQualities(qualities);
    }

}
