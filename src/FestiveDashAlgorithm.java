/**
 * @file FestiveDashAlgorithm.java
 * @brief Class that implements the Festive Algorithm for DASH
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
import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import java.io.File;
import java.io.IOException;


public class FestiveDashAlgorithm extends DashAlgorithm {

    // Variables declaration
    private CircularFifoQueue<Double> measuredThroughput;
    private CircularFifoQueue<Integer> switches;
    private int action;
    private int bitrateToDownload;
    private int prevAction;
    private int nChunks;

    private double bufferSize;
    private double reward;
    private double quality;

    private static double ALPHA = 12;
    private static double TARGET_BUFFER = 15;
    private static int SWITCHES_MEMORY_SIZE = 10;
    private static int THROUGHTPUT_MEMORY_SIZE = 20;


    /**
     * Default constructor for the class FestiveDashAlgorithm.
     *
     * @param player         player which have to playback the video
     * @param tempFolderPath path of the temporary folder used to store the downloaded segment
     * @param mpdUrl         URL of the mpd file associated to the video
     */
    FestiveDashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl) {
        super(player, tempFolderPath, mpdUrl);
        this.bufferSize = 0;
        this.quality = 0;
        this.reward = 0;
        this.measuredThroughput = new CircularFifoQueue<Double>(THROUGHTPUT_MEMORY_SIZE);
        this.switches = new CircularFifoQueue<Integer>(SWITCHES_MEMORY_SIZE);
        this.nChunks = 0;
    }

    /**
     * Specialized constructor for the class FestiveDashAlgorithm.
     *
     * @param player            player which have to playback the video
     * @param tempFolderPath    path of the temporary folder used to store the downloaded segment
     * @param mpdUrl            URL of the mpd file associated to the video
     * @param targetBuf         the dimension of the targetbuf.
     * @param amp               the delta in the uniform distribution.
     * @param bandwidthmeasures the number of measurements used to calculate the harmonic mean.
     * @param a                 the param alpha in the score functions.
     * @param mls               the value of the multi-level switch.
     * @param mrs               the value of the max-rate switch.
     */
    FestiveDashAlgorithm(MediaListPlayer player, String tempFolderPath, String mpdUrl, int targetBuf, int amp, int bandwidthmeasures, int a, int mls, int mrs) {
        super(player, tempFolderPath, mpdUrl);

        this.measuredThroughput = new CircularFifoQueue<Double>(THROUGHTPUT_MEMORY_SIZE);
        this.switches = new CircularFifoQueue<Integer>(SWITCHES_MEMORY_SIZE);
        this.nChunks = 0;

    }

    /**
     * Method that fills the buffer following the FESTIVE strategy.
     */
    @Override
    void getNextSeg() throws IOException, InterruptedException {

        if (current <= 0) {
            markovDP.init();
        }

        markovDP.moveNextState(complexities[current], current);
//        LOGGER.log(Level.INFO, "Segment{0}", String.valueOf(current));

        bufferPlotter.addDataToChart((current), markovDP.getBuffer(), 1);
        rewardPlotter.addDataToChart(current, markovDP.getReward(), 1);
        qualityPlotter.addDataToChart(current, markovDP.getQuality(), 1);


        bufferSize = markovDP.getBuffer();
        reward = markovDP.getReward();
        quality = markovDP.getQuality();

        double wait = 0;
        if (current < THROUGHTPUT_MEMORY_SIZE) {
            action = bitrates.length - 1;
            bitrateToDownload = bitrates[action];
            switches.add(0);
        } else {

            prevAction = action;
            double bandwidthEstimation;

            //number of the frame to download
            bandwidthEstimation = this.harmonicMean(measuredThroughput);

//        printMessage("FESTIVE: Actual bitrate index: " + (prevAction + 1) + " on "+bitrates.length+" | EstimatedBitrate " + Math.round(bandwidthEstimation) + " b/s");

            this.statefulDelayedBitrateSelection(bandwidthEstimation);

            if (action == prevAction) {
                switches.add(0);
                nChunks = nChunks + 1;
            } else {
                switches.add(1);
                nChunks = 1;
            }
        }

        //setting bitrate to download for selected action
        bitrateToDownload = bitrates[action];

        // Download the chosen segment
        String segmentUrl = mpdUrl.substring(0, mpdUrl.lastIndexOf("/") + 1) + (parser.getTemplate().replace("$RepresentationID$",
                Integer.toString(action +1)).replace("$Number$", Integer.toString(current + 1)));

        System.out.println("SEGMENT URL: " + segmentUrl);

        Double lastChBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + (current + 1 ) + ".mp4",
                tempFolderPath + "init" + File.separator + (action + 1) + "_init");

//        System.out.println("URL: " + segmentUrl);
//        System.out.println(("HEADER:"+ tempFolderPath + "init" + File.separator + (action + 1) + "_init"));
//        System.out.println("Dest: " + tempFolderPath + "seg" + File.separator + "seg" + (current + 1) + ".mp4");

        //add measured throughtput to memory
        measuredThroughput.add(lastChBitrate);

        //add downloaded segment to playoutbuffer
        buffer.addMedia(tempFolderPath + "seg" + File.separator + "seg" + Integer.toString(current) + ".mp4");

        //get download time
        double segDownloadTime = downloader.getLastSegmentDownloadTime();

        System.out.println("Download: bitrate: "+(lastChBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());

        //setting next state s_(t+1)
        markovDP.computeNextState(lastChBitrate, bitrateToDownload, action, segDownloadTime, current, wait);

        //control when to start the download of the next chunk
        if (markovDP.getBuffer() > TARGET_BUFFER) {
            double randbuffer = Math.random() * parser.getSegmentDuration() * 2 + (TARGET_BUFFER - parser.getSegmentDuration());
            wait = markovDP.getBuffer() - randbuffer;
            try {
                Thread.currentThread().sleep(Math.max(0,(long)wait));
            } catch (InterruptedException e) {
                System.err.println(e.getMessage());
            }

        }

        //stop freezing event
        if (!player.isPlaying() && PlayerEventListener.segIndex == player.getMediaList().size()) {
//            preBuffering();
            player.playItem(PlayerEventListener.segIndex - 1);
        }

        current++;
    }

    /**
     * Method that do the pre buffering and fill the buffer with <code>nSegPrebuffer</code> at <code>currentIndex</code> quality.
     */
    @Override
    void preBuffering() throws IOException {
    }



    /**
     * Bitrate selection policy that allows multilevel bitrate switches and different bitrates switching rate.
     * See the reference paper for more details.
     *
     * @param estimatedBandwidth the estimated bandwidth of the link.
     */
    private void statefulBitrateSelection(double estimatedBandwidth) {

        if (bitrates[prevAction] > 0.85 * estimatedBandwidth) {
            action = Math.min(bitrates.length - 1, prevAction + 1);
        } else {
            int future =  Math.max(0, prevAction - 1);
            if ((nChunks > bitrates.length - future ) && (bitrates[future] <= 0.85*estimatedBandwidth)) {
                action = future;
                System.out.println("future: "+ future);

            }
        }

        System.out.println("nChunks: " + nChunks);
        System.out.println("Action:" + action);

    }

    /**
     * The bitrate selection policy used in the more sophisticated festive dash algorithm, comprehensive of the delayed
     * update strategy. See the reference paper for more details.
     *
     * @param estimatedBandwidth the estimated bandwidth of the link.
     */
    private void statefulDelayedBitrateSelection(double estimatedBandwidth) {
        this.statefulBitrateSelection(estimatedBandwidth);
        if (action != prevAction) {
            //computing the various scores
            double scoreEffCurrent = this.scoreEfficiency(bitrates[prevAction], estimatedBandwidth);
            double scoreEffReference = this.scoreEfficiency(bitrateToDownload, estimatedBandwidth);
            double scoreStabCurrent = this.scoreStability(bitrates[prevAction]);
            double scoreStabReference = this.scoreStability(bitrateToDownload);
            //chooses if it's better the new calculated reference bitrate or the previous one
            if ((scoreEffReference + (ALPHA * scoreStabReference)) > (scoreEffCurrent + (ALPHA * scoreStabCurrent))) {
                action = prevAction;
            }
        }

    }

    /**
     * Utility function that compute how close to stable allocation the current and reference bitrate previously
     * calculated are. Lower is better, best is when bitrate=referenceBitrate (<code>bitrateToDownload</code>).
     *
     * @param bitrate   the input parameter of the score function.
     * @param bandwidth the input parameter of the score function.
     * @return the value of the calculated function.
     */
    private double scoreEfficiency(int bitrate, double bandwidth) {
        return Math.abs((((double) bitrate) / (Math.min(bandwidth, (double) bitrateToDownload))) - 1);
    }

    /**
     * Utility function that calculates the stability cost for a given bitrate, considering how many bitrate switches the
     * player has undergone recently. Lower is better.
     *
     * @param bitrate the input parameter of the score function.
     * @return the value of the calculated function.
     */
    private double scoreStability(int bitrate) {
        if (bitrate == bitrates[prevAction])
            return Math.pow((double) 2, getNumberOfSwitches());
        else
            return (Math.pow((double) 2, getNumberOfSwitches()) + 1);
    }

    private int getNumberOfSwitches() {

        int sum = 0;
        for (int i = 0; i < switches.size(); i++) {
            sum += switches.get(i);
        }

        return sum;
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
