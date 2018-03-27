package interfaces;

import java.io.IOException;

/**
 * Created by davidetalon on 26/07/17.
 */
public interface Channel {

    double download(int bitrate);

    double downloadFile(String url, String destFilePath, String header) throws IOException;

    double downloadFile(String url, String destFilePath) throws IOException;



    void changeChannelCapacity();

    double getLastSegmentDownloadTime();

}
