import exceptions.InsufficientVideoRealTracesException;
import exceptions.InvalidPlotterException;
import interfaces.Channel;
import interfaces.Video;

import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;


/**
 * Created by davidetalon on 14/05/17.
 */
public class Trainer extends Thread {

    private String tempFolderPath;
    private String sourceUrl;

    private boolean isInterrupted;
    private int pretrain;
    private int train;
    private int test;
    private int nSegments;
    private boolean loadRealTraces;
    private boolean online;
    private int currentVideo;
    private int currentSegment;
    private int[] bitrates;
    private int[] complexities;
    private double[][] qualities;
    private MarkovDecisionProcess markovDP;
    private Channel downloader;

    private double totalBuffer;
    private double totalBufferBitrate;
    private double totalReward;
    private double totalRewardBitrate;
    private double totalQuality;
    private double totalQualityBitrate;

    private double lastChBitrate;
    private MarkovDecisionProcess markovDPBitRate;

    private Plotter bufferPlotter;
    private Plotter rewardPlotter;
    private Plotter qualityPlotter;

    private String videoName;


    //logit function
    private static double k_train = 0.3;
    private static double k_pretrain = 0.0125;
    public final double DASH_SEGMENT_DURATION = 2;
    private static String MPD_FILENAME = "mpdList.mpd";

    private static final Logger LOGGER = Logger.getLogger( "MDPLog" );


    public Trainer (int pretrain,int train, int test, int nSegments, boolean loadRealTraces, boolean online) throws IOException {


//        this.player = player;
        this.isInterrupted = false;
        this.pretrain = pretrain;
        this.train = train;
        this.test = test;
        this.nSegments = nSegments;
        this.loadRealTraces = loadRealTraces;
        this.online = online;
        this.sourceUrl = null;
        this.currentVideo = 0;
        this.currentSegment = 0;

        this.totalBuffer = 0;
        this.totalBufferBitrate = 0;
        this.totalReward = 0;
        this.totalRewardBitrate = 0;
        this.totalQuality = 0;
        this.totalQualityBitrate = 0;

        this.markovDP = new MarkovDecisionProcess(DASH_SEGMENT_DURATION, SyntheticVideo.bitrates[0]);
        this.markovDPBitRate = new MarkovDecisionProcess(DASH_SEGMENT_DURATION, SyntheticVideo.bitrates[0]);

        this.bufferPlotter = null;
        this.rewardPlotter = null;
        this.qualityPlotter = null;

        try {

            markovDP.startSession();

        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Loading model");
        markovDP.loadTrainedModel();

        FileHandler fh = new FileHandler("MDP.log");
        SimpleFormatter formatter = new SimpleFormatter();

        fh.setFormatter(formatter);
        LOGGER.addHandler(fh);
        LOGGER.setUseParentHandlers(false);


    }

    public void run() {

//        markovDP.loadTrainedModel();

        double i0_pretrain = pretrain/2;
        double i0_train = pretrain;
        double L_pretrain = 0.8;
        double L_train = L_pretrain*2;


        System.out.println("Start pretraining...");

        Video video = null;

        String[] videos = {"elephantsdream", "sintel", "tearsofsteel"};
        double[] temperatures = {1.000,0.000};


        if (online) {

            //use file downloader from url
            this.downloader = new FileDownloader();

        } else  {

            //load channel capacities and video complexities from real traces(from file)
            if(loadRealTraces) {

                try {
                    video = new VideoLoader(400, 1000, true);
                    this.downloader = new ChannelLoader(400000, true);
                } catch (InsufficientVideoRealTracesException e) {
                    e.printStackTrace();
                }

            }
        }



//        while(!isInterrupted && currentVideo < pretrain){
        for(int j = 0; j < temperatures.length && !isInterrupted; j++) {

            for (int k = 0; k < videos.length && !isInterrupted; k++) {

                videoName = videos[k];

                //            set the epsilon-greediness
                //            double epsilon = Math.max(0.0025, 0.5 / (Math.floor(currentVideo/ 3) + 1));

                if (online) {

                    try {

                        String source = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + videos[k] + "/" + videos[k] + "-simple.mpd";
                        //                    String source = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + "Video" + currentVideo + ".mpd";
                        downloader.downloadFile(source, tempFolderPath + MPD_FILENAME);

                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Download all the header files
                    video = new MPDParser(tempFolderPath + MPD_FILENAME);

                } else {

                    if (!loadRealTraces) {
                        video = new SyntheticVideo(nSegments);
                        downloader = new MarkovDownloader(video);
                    }

                }

                //            int ii = 2 * currentVideo + 1;
                //
                //            double epsilon = L_pretrain - (L_pretrain / (1 + Math.exp(-k_pretrain*(ii - i0_pretrain))));
                //            epsilon = Math.max(0.002, epsilon);

                try {

                    //                if((currentVideo + 1) % 2 == 0) {
                    //                    playVideo(video, 0, false);
                    //                } else {
                    //                    playVideo(video, epsilon, true);
                    //                }

                    //                playVideo(video, epsilon, true);
                    playVideo(video, temperatures[j], true);

                } catch (InvalidPlotterException e) {
                    e.printStackTrace();
                }

                System.out.println("Video "+ j + ", " + videos[k] + ", temperature " + temperatures[j] + " finished!");
                //            System.out.println("Video "+ currentVideo + " finished!");
                //            currentVideo++;

            }

//            System.out.println("Start training...");

            Video trainingVideo = null;

            //loading video from real traces(from file)
            if (!online) {

                try {
                    trainingVideo = new VideoLoader(400, 80, false);
                } catch (InsufficientVideoRealTracesException e) {
                    e.printStackTrace();
                }

            }

            //
            //        this.downloader = new ChannelLoader(32000, false);

            //        while(!isInterrupted && currentVideo < train){
            //
            //
            //            if (online) {
            //
            //                try {
            //
            //                    String source = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + "Video" + currentVideo + ".mpd";
            //                    //downloading MPD file
            //                    downloader.downloadFile(source, tempFolderPath + MPD_FILENAME);
            //
            //                } catch (IOException e) {
            //                    e.printStackTrace();
            //                }
            //
            //                //initializing MPDParser
            //                trainingVideo = new MPDParser(tempFolderPath + MPD_FILENAME);
            //
            //            }
            //            int ii = 2 * currentVideo + 1;
            //
            //            double epsilon =  L_train - (L_train / (1 + Math.exp(-k_train*(ii +1 -i0_train))));
            //            epsilon = Math.max(0.002, epsilon);
            //
            //
            ////            double epsilon = Math.max(0.0025, 0.5 / Math.floor((currentVideo/ 3) + 1));
            //
            //            System.out.println("Epsilon: " + epsilon);
            //
            //            try {
            //
            //                if((currentVideo + 1) % 2 == 0) {
            //                    playVideo(trainingVideo, 0, false);
            //                } else {
            //                    playVideo(trainingVideo, epsilon, true);
            //                }
            //
            //            } catch (InvalidPlotterException e) {
            //                e.printStackTrace();
            //            }
            //
            //
            //            currentVideo++;
            //            System.out.println("Video "+ currentVideo + " finished!");
            //        }

//            System.out.println("Start test...");
            //
            //
            //        while(currentVideo < test) {
            //
            //            playVideo(0);
            //
            //            currentVideo++;
            //            System.out.println("Video "+ currentVideo + " finished!");
            //        }


        }

        System.out.println("Saving model");
        markovDP.saveTrainedModel();

        try {
            markovDP.closeSession();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void playVideo(Video video,double epsilon, boolean update) throws InvalidPlotterException {

        if(bufferPlotter == null || rewardPlotter == null || qualityPlotter == null) {
            throw new InvalidPlotterException();
        }

        currentSegment = 0;

        totalBuffer = 0;
//        totalBufferBitrate = 0;
        totalReward = 0;
//        totalRewardBitrate = 0;
        totalQuality = 0;
//        totalQualityBitrate = 0;


        String log = "{\"Video\": "+videoName+", \"tau\": " + epsilon + "}";
//        LOGGER.log( Level.INFO, log);


        //getting information about bitrates and segmentComplexities
        bitrates = video.getBitrates();

        complexities = video.getSegmentComplexityIndexes();
        qualities = video.getQualities();
        markovDP.setQualities(qualities);

        markovDP.init();
        markovDPBitRate.init();

        while (!isInterrupted && currentSegment < video.getNFrames()) {
            getNextSeg(epsilon,update);
//            getNextSeg();
            downloader.changeChannelCapacity();
            currentSegment++;
        }

        double meanBuffer = totalBuffer/nSegments;
//        double meanBufferBitrate = totalBufferBitrate/nSegments;

        double meanReward = totalReward /nSegments;
//        double meanRewardBitrate = totalRewardBitrate/nSegments;

        double meanQuality = totalQuality/nSegments;
//        double meanQualityBitrate = totalQualityBitrate/nSegments;


        //plotting buffer, quality, reward mean
        if(update == false) {

            bufferPlotter.addDataToChart((currentVideo/2), meanBuffer, 1);
//            bufferPlotter.addDataToChart((currentVideo/2), meanBufferBitrate, 2);

            rewardPlotter.addDataToChart((currentVideo/2), meanReward, 1);
//            rewardPlotter.addDataToChart((currentVideo/2), meanRewardBitrate, 2);

            qualityPlotter.addDataToChart((currentVideo/2), meanQuality, 1);
//            qualityPlotter.addDataToChart((currentVideo/2), meanQualityBitrate, 2);

        }

        System.out.println("Video" + currentVideo + ": ");
        System.out.println("Mean buffer: " + meanBuffer);
//        System.out.println("Mean buffer bitrate-based" + meanBufferBitrate);
        System.out.println("Mean reward: " + meanReward);
//        System.out.println("Mean reward bitrate-based" + meanRewardBitrate);
        System.out.println("Mean quality: " + meanQuality);
//        System.out.println("Mean quality bitrate-based" + meanQualityBitrate);

    }

    private void getNextSeg(double epsilon, boolean update) {

        markovDP.moveNextState(complexities[currentSegment], currentSegment);

        if (currentSegment > 0) {
            double loss = 0;
            loss = markovDP.addToNetworkMemory(update);
        }

        double buffer = markovDP.getBuffer();
        totalBuffer += buffer;
        double reward = markovDP.getReward();
        totalReward += reward;
        double quality = markovDP.getQuality();
        totalQuality += quality;

        int action = markovDP.getNextAction(epsilon, false);


        //TEST TEMPERATURE = 0
//        int action = markovDP.getNextAction(0, false);


        // setting segment url for download
        int bitrateToDownload = bitrates[action];

        double lastChBitrate = 0;


        if (online) {

            String segmentUrl = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + videoName + "/" +
                    videoName + (action + 1) + "/" + videoName + (action + 1) + "_" + (currentSegment + 1) + ".mp4";
//            String segmentUrl = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + "bitrate" + action + ".mp4";


            //setting previous channel sample and downloading

            try {
                lastChBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + currentSegment + ".mp4");
            } catch (Exception e) {
                e.printStackTrace();
            }




        } else {

            lastChBitrate = downloader.download(bitrateToDownload);

        }

        //get download time
        double segDownloadTime = downloader.getLastSegmentDownloadTime();
        System.out.println("Download: bitrate: "+(lastChBitrate/1000000) + ", tempo: "+ segDownloadTime+ ", buffer" + markovDP.getBuffer());



        //setting next state s_(t+1)
        markovDP.computeNextState(lastChBitrate, bitrateToDownload, action, segDownloadTime, currentSegment,0);

    }


//    private void getNextSeg() {
//
//        markovDPBitRate.moveNextState(complexities[currentSegment], currentSegment);
//
//        double buffer = markovDPBitRate.getBuffer();
//        totalBufferBitrate += buffer;
//        double reward = markovDPBitRate.getReward();
//        totalRewardBitrate += reward;
//        double quality = markovDPBitRate.getQuality();
//        totalQualityBitrate += quality;
//
//        //get action with Bit-rate based policy
//        int action = 7;
//        if (currentSegment > 0) {
//            while (action > 0 && SyntheticVideo.bitrates[action - 1] <=  lastChBitrate) {
//                action = action - 1;
//            }
//        }
//
//        // setting segment url for download
//        int bitrateToDownload = bitrates[action];
//
//        if (online) {
//
//            String segmentUrl = sourceUrl.substring(0, sourceUrl.lastIndexOf("/") + 1) + "bitrate" + action + ".mp4";
//
//            //setting previous channel sample and downloading
//
//            try {
//                lastChBitrate = downloader.downloadFile(segmentUrl, tempFolderPath + "seg" + File.separator + "seg" + currentSegment + ".mp4");
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//
//
//        } else {
//
//            lastChBitrate = downloader.download(bitrateToDownload);
//
//        }
//
//        //get download time
//        double segDownloadTime = downloader.getLastSegmentDownloadTime();
//
//        //setting next state s_(t+1)
//        markovDPBitRate.computeNextState(lastChBitrate, bitrateToDownload, action, segDownloadTime, currentSegment, 0);
//
//    }

    public void setTempFolderPath (String tempFolderPath) {
        this.tempFolderPath = tempFolderPath;
    }

    public void setSourceUrl (String sourceUrl) {
        this.sourceUrl = sourceUrl;
    }

    public void setPlotters(Plotter bufferPlotter, Plotter qualityPlotter, Plotter rewardPlotter) {
        this.bufferPlotter = bufferPlotter;
        this.qualityPlotter = qualityPlotter;
        this.rewardPlotter = rewardPlotter;
    }

    public void forceInterrupt(){
        isInterrupted = true;
    }


}
