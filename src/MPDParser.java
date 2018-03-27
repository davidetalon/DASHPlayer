/**
 * @file MPDParser.java
 * @brief Class that parse MPD file for DASH protocol
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

import interfaces.Video;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.util.Arrays;

public class MPDParser implements Video{
    private String path;
    private Document xmldoc;
    private double segmentDuration;
    private double[][] qualities;

    public MPDParser(String path) {
        this.path = path;
        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            xmldoc = dBuilder.parse(path);
            xmldoc.getDocumentElement().normalize();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns an int that rappresents the number of segments in which the video is splitted
     *
     * @return
     */
    public int getNFrames() {
        int hours = 0;
        int mins = 0;
        int secs = 0;
        int cents = 0;
        Double segDuration = 0.0;
        int numSeg;
        int totalSecs;
        try {
            Element templateNode = (Element) xmldoc.getElementsByTagName("SegmentTemplate").item(0);
            segDuration = Double.parseDouble(templateNode.getAttribute("duration")) / Double.parseDouble(templateNode.getAttribute("timescale"));
            segmentDuration = (double) segDuration;
            Element timeNode = (Element) xmldoc.getElementsByTagName("Period").item(0);
            String timeString = timeNode.getAttribute("duration");
            timeString = timeString.substring(2);
            int hourIndex = timeString.lastIndexOf("H");
            int minIndex = timeString.lastIndexOf("M");
            int secIndex = timeString.lastIndexOf(".");
            int centIndex = timeString.lastIndexOf("S");
            hours = Integer.parseInt(timeString.substring(0, hourIndex));
            mins = Integer.parseInt(timeString.substring(hourIndex + 1, minIndex));
            secs = Integer.parseInt(timeString.substring(minIndex + 1, secIndex));
            cents = Integer.parseInt(timeString.substring(secIndex + 1, centIndex));
        } catch (Exception e) {
            e.printStackTrace();
        }
        int add = (cents != 0) ? 1 : 0;
        totalSecs = hours * 3600 + mins * 60 + secs + add;
        numSeg = (int) (Math.ceil(totalSecs / segDuration));
        return numSeg;
    }


    /**
     * Returns an array of int containing the different quality index specified in the mpd file for each segment
     * @return int[]
     */
    public int[] getSegmentComplexityIndexes() {
        int len;
        int[] complexities = new int[getNFrames()];
//        try {
            NodeList nList = xmldoc.getElementsByTagName("Segment");
            len = nList.getLength();
            qualities = new double[len][getBitrates().length];
            for (int i = 0; i < len; i++) {
                Node current = nList.item(i);
                complexities[i] = (int)Double.parseDouble(current.getAttributes().getNamedItem("complexity").getNodeValue());
                NodeList chunks = current.getChildNodes();
                int chunksLen = chunks.getLength();
                int representation = 0;
                for (int j = 0; j < chunksLen; j++) {
                    Node currentChunk = chunks.item(j);
                    if (currentChunk.getNodeType() == Node.ELEMENT_NODE) {
                        Element element = (Element) currentChunk;
                        qualities[i][representation] = Double.parseDouble(element.getAttribute("quality"));
                        representation++;

                    }
                }

            }

        return complexities;

    }




    /**
     * Returns an array of Strings containing the different bitrates specified in the mpd file
     * @return String[]
     */
    public int[] getBitrates() {
        int len;
        int[] bitrates = null;
        try {
            NodeList nList = xmldoc.getElementsByTagName("Representation");
            len = nList.getLength();
            bitrates = new int[len];
            for (int i = 0; i < len; i++) {
                Node current = nList.item(i);
                if (current.getNodeType() == Node.ELEMENT_NODE) {
                    Element currElement = (Element) current;
                    bitrates[i] = Integer.parseInt(currElement.getAttribute("bandwidth"));
                }
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        Arrays.sort(bitrates);
        reverse(bitrates);
        return bitrates;
    }

    private static void reverse(int[] input) {
        int last = input.length - 1;
        int middle = input.length / 2;
        for (int i = 0; i < middle; i++) {
            int temp = input[i];
            input[i] = input[last - i];
            input[last - i] = temp;
        }
    }



    /**
     * Returns a String containing the standard name of the packet file
     * @return String
     */
    public String getTemplate() {
        String template = "";
        try {
            Element templateNode = (Element) xmldoc.getElementsByTagName("SegmentTemplate").item(0);
            template = templateNode.getAttribute("media");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return template;
    }

    /**
     * Returns a String containing the standard name of the init file
     *
     * @return String
     */
    public String getInitialization() {
        String template = "";
        try {
            Element templateNode = (Element) xmldoc.getElementsByTagName("SegmentTemplate").item(0);
            template = templateNode.getAttribute("initialization");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return template;
    }

    public double getSegmentDuration(){
       return segmentDuration;
    }
    public double[][] getQualities() {return qualities;}

}
