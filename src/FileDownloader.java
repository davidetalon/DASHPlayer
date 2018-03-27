/**
 * @file FileDownloader.java
 * @brief Public class FileDownloader. Download a file from a specified URL.
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

import interfaces.Channel;

import java.io.*;
import java.net.URL;


public class FileDownloader implements Channel{

    private double lastSegmentDownloadTime;


    public FileDownloader() {
        lastSegmentDownloadTime = 0;
    }
    /**
     * Download a File from the given URL.
     *
     * @param url          URL of the file
     * @param destFilePath name of the file to be saved
     * @return Bitrate calculated by downloading this file
     */
    public double downloadFile(String url, String destFilePath) throws IOException {
        long startDownloadTime = System.currentTimeMillis();
        long stopDownloadTime;
        BufferedInputStream in = null;
        FileOutputStream fout = null;
        try {
            in = new BufferedInputStream(new URL(url).openStream());
            fout = new FileOutputStream(destFilePath);

            final byte data[] = new byte[1024];
            int count;
            // Download file segment and add it to the same file
            while ((count = in.read(data, 0, 1024)) != -1) {
                fout.write(data, 0, count);
            }
        } catch(IOException e) {

            e.getMessage();

        } finally {

            if (in != null) {
                in.close();
            }
            if (fout != null) {
                fout.close();
            }
            stopDownloadTime = System.currentTimeMillis();
        }

        long fileDimension = (new File(destFilePath)).length();

        lastSegmentDownloadTime = (double) (stopDownloadTime - startDownloadTime) / 1000;

        double bitrate = (fileDimension * 8) / lastSegmentDownloadTime;

        Player.addMessage("DOWNLOADER: file " + url + " downloaded successfully");
        Player.addLog("DOWNLOADER: file " + url + " downloaded successfully");



        if (fileDimension == 0) {
            throw new IOException();
        }
        return bitrate;
    }

    /**
     * Download a File from the given URL and concatenates header to the downloaded file
     *
     * @param url          URL of the file
     * @param destFilePath name of the file to be saved
     * @return Bitrate calculated by downloading this file
     */
    public double downloadFile(String url, String destFilePath, String header) throws IOException {
        long startDownloadTime = System.currentTimeMillis();
        long stopDownloadTime;
        BufferedInputStream inHeader = null;
        BufferedInputStream in = null;
        FileOutputStream fout = null;
        try {
            inHeader = new BufferedInputStream(new FileInputStream(header));
            in = new BufferedInputStream(new URL(url).openStream());
            fout = new FileOutputStream(destFilePath);

            final byte data[] = new byte[1024];
            int count;
            // Add header to buffer
            while ((count = inHeader.read(data, 0, 1024)) != -1) {
                fout.write(data, 0, count);
            }
            // Download file segment and add it to the same file
            while ((count = in.read(data, 0, 1024)) != -1) {
                fout.write(data, 0, count);
            }
        } finally {
            if (in != null) {
                in.close();
            }
            if (inHeader != null) {
                inHeader.close();
            }
            if (fout != null) {
                fout.close();
            }
            stopDownloadTime = System.currentTimeMillis();
        }

        long fileDimension = (new File(destFilePath)).length();

        lastSegmentDownloadTime = (double) (stopDownloadTime - startDownloadTime) / 1000;

        long bitrate = Math.round((fileDimension * 8) / ((double) (stopDownloadTime - startDownloadTime) / 1000));

//        Player.addMessage("DOWNLOADER: segment " + destFilePath.substring(destFilePath.lastIndexOf("g") + 1, destFilePath.lastIndexOf(".")) + " downloaded | DownloadedPacketBitrate: " + bitrate + " b/s");
//        Player.addLog("DOWNLOADER: segment " + destFilePath.substring(destFilePath.lastIndexOf("g") + 1, destFilePath.lastIndexOf(".")) + " downloaded | DownloadedPacketBitrate: " + bitrate + " b/s");

        Player.addMessage("DOWNLOADER: file " + url + " downloaded successfully");
        Player.addLog("DOWNLOADER: file " + url + " downloaded successfully");

        if (fileDimension == 0) {
            throw new IOException();
        }
        return bitrate;
    }

    public double getLastSegmentDownloadTime(){
        return lastSegmentDownloadTime;
    }

    public void changeChannelCapacity(){

    }

    public double download(int bitrate){
        return 0;
    }


}
