import interfaces.Channel;
import interfaces.Video;

/**
 * Created by davidetalon on 14/04/17.
 */
public class MarkovDownloader implements Channel{

    private MarkovChannel mCh;
    private Video syntheticVideo;
    private double lastSegmentSize;
    private double lastSegmentDownloadTime;
    private int[] bitrates;

    private static int SEGMENT_DURATION = 2;


    public MarkovDownloader(Video syntheticVideo) {
        this.syntheticVideo = syntheticVideo;
        this.bitrates = syntheticVideo.getBitrates();
        this.mCh = new MarkovChannel();
    }

    public double download(int bitrate) {

        lastSegmentSize = (double) bitrate*SEGMENT_DURATION;

        double channelCapacity = mCh.getChannelCapacity();
        lastSegmentDownloadTime = lastSegmentSize / channelCapacity;

        return channelCapacity;

    }

    public void changeChannelCapacity(){
        mCh.changeChannelCapacity();
    }

    public double getLastSegmentSize(){
        return lastSegmentSize;
    }

    public double getLastSegmentDownloadTime() {
        return lastSegmentDownloadTime;
    }

    public double downloadFile(String string, String string2) {
        return 0;
    }

    public double downloadFile(String string, String string3, String string4) {
        return 0;
    }


}
