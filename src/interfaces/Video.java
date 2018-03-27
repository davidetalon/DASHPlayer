package interfaces;

/**
 * Created by davidetalon on 26/07/17.
 */
public interface Video {

    int[] getBitrates();

    int[] getSegmentComplexityIndexes();

    int getNFrames();

    double[][] getQualities();

}
