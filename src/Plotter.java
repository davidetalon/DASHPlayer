/**
 * Created by davidetalon on 22/05/17.
 */
import com.sun.jna.platform.win32.NTSecApi;
import javafx.animation.AnimationTimer;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;



public class Plotter {

    private static final int MAX_DATA_POINTS = 600;
    private static final double BUFFER_MAX_VALUE = 20.00;
    private static final double REWARD_MAX_VALUE = 1.00;
    private static final double QUALITY_MAXVALUE = 1.00;
    private int xSeriesData = 0;
    private XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
    private XYChart.Series<Number, Number> series2 = new XYChart.Series<>();


//    private XYChart.Series<Number, Number> series2 = new XYChart.Series<>();
//    private XYChart.Series<Number, Number> series3 = new XYChart.Series<>();
    private ExecutorService executor;
    private ConcurrentLinkedQueue<Point> data = new ConcurrentLinkedQueue<>();
    private ConcurrentLinkedQueue<Point> data2 = new ConcurrentLinkedQueue<>();
//    private ConcurrentLinkedQueue<Number> dataQ3 = new ConcurrentLinkedQueue<>();

    private NumberAxis xAxis;
    private NumberAxis yAxis;
    private int numero;

    //TODO modificare il plotter in funzione di ciò che bisogna graficare (reward, buffer, quality)
    private void init(JFXPanel jfxpanel, int plotterTyper) {

        xAxis = new NumberAxis(0, MAX_DATA_POINTS, MAX_DATA_POINTS/12);
        xAxis.setMinorTickVisible(false);


        //selecting axis for specific plot
        if (plotterTyper == 0) {
            yAxis = new NumberAxis(0, BUFFER_MAX_VALUE,BUFFER_MAX_VALUE/4);
            yAxis.setMinorTickVisible(false);
        } else {
            yAxis = new NumberAxis(0.85,   REWARD_MAX_VALUE,REWARD_MAX_VALUE/20);
        }

        numero = 0;

            // Create a LineChart
        final LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis){
//             Override to remove symbols on each data point
            @Override protected void dataItemAdded(Series<Number, Number> series, int itemIndex, Data<Number, Number> item) {
            }
        };
        //TODO rendere la linea più sottile
        lineChart.setStyle(".chart-series-line {    \n"
                + "    -fx-stroke-width: 0.5;\n"
                + "}");
        lineChart.setAnimated(true);
        switch (plotterTyper) {

            case 0:
                lineChart.setTitle("Buffer per video");
                break;
            case 1:
                lineChart.setTitle("Reward per video");
                break;
            case 2:
                lineChart.setTitle("Quality per video");

        }

        lineChart.setHorizontalGridLinesVisible(true);
        lineChart.setVerticalGridLinesVisible(true);

        // Set Name for Series
        series1.setName("MLP2");
        series2.setName("Bitrate");

        // Add Chart Series
        lineChart.getData().addAll(series1);
        lineChart.getData().addAll(series2);

        jfxpanel.setScene(new Scene(lineChart));
    }



    public void addDataToChart(int index, double value, int serie) {

        Point point = new Point(index, value);
        if (serie == 1) {
            data.add(point);
        } else {
            data2.add(point);
        }


    }

    public void start(JFXPanel jfxpanel, int plotterType) {

        init(jfxpanel, plotterType);

        executor = Executors.newCachedThreadPool(new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread thread = new Thread(r);
                thread.setDaemon(true);
                return thread;
            }
        });

//        AddToQueue addToQueue = new AddToQueue();
//        executor.execute(addToQueue);
        //-- Prepare Timeline
        prepareTimeline();
    }

//    private class AddToQueue implements Runnable {
//        public void run() {
//            try {
//                // add a item of random data to queue
//
//                Point point = new Point(numero, Math.random());
//                data.add(point);
//
//                Thread.sleep(500);
//                executor.execute(this);
//                numero++;
//            } catch (InterruptedException ex) {
//                ex.printStackTrace();
//            }
//        }
//    }

    //-- Timeline gets called in the JavaFX Main thread
    private void prepareTimeline() {
        // Every frame to take any data from queue and add to chart
        new AnimationTimer() {
            @Override
            public void handle(long now) {
                addDataToSeries();
            }
        }.start();
    }

    private void addDataToSeries() {
        for (int i = 0; i < 20; i++) { //-- add 20 numbers to the plot+
            if (data.isEmpty()) break;
            xSeriesData++;
            Point point = data.remove();
            int index = point.getxValue();
            double value = point.getyValue();
            series1.getData().add(new XYChart.Data(index, value));
        }

        for (int i = 0; i < 20; i++) { //-- add 20 numbers to the plot+
            if (data2.isEmpty()) break;
            xSeriesData++;
            Point point = data2.remove();
            int index = point.getxValue();
            double value = point.getyValue();
            series2.getData().add(new XYChart.Data(index, value));
        }

        // remove points to keep us at no more than MAX_DATA_POINTS
//        if (series1.getData().size() > MAX_DATA_POINTS) {
//            series1.getData().remove(0, series1.getData().size() - MAX_DATA_POINTS);
//        }

        // update
//        xAxis.setLowerBound(xSeriesData - MAX_DATA_POINTS);
        xAxis.setUpperBound(xSeriesData - 1);

    }

}