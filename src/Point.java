/**
 * Created by davidetalon on 23/05/17.
 */
public class Point {

    private int xValue;
    private double yValue;

    public Point(int xValue, double yValue) {
        this.xValue = xValue;
        this.yValue = yValue;
    }


    public int getxValue() {
        return xValue;
    }

    public double getyValue() {
        return yValue;
    }

    public void setxValue(int xValue) {
        this.xValue = xValue;
    }

    public void setyValue(double yValue) {
        this.yValue = yValue;
    }
}
