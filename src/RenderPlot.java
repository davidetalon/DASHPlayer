import javafx.embed.swing.JFXPanel;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.text.Text;

import javax.swing.*;
import java.awt.*;
/**
 * Created by davidetalon on 02/08/17.
 */
public class RenderPlot extends JFrame{


        public static int BUFFER_PLOT = 0;
        public static int REWARD_PLOT = 1;
        public static int QUALITY_PLOT = 2;


        private final JFXPanel jfxPanel = new JFXPanel();

        private final JPanel panel = new JPanel(new BorderLayout());

        private Plotter plotter;


        public RenderPlot(Plotter plotter, int plotterType) {

            super();
            initComponents(plotter, plotterType);

        }

        private void initComponents(Plotter plotter, int plotterType) {
            Scene scene = createScene();
//        jfxPanel.setScene(scene);

            plotter.start(jfxPanel, plotterType);

            panel.add(jfxPanel, BorderLayout.CENTER);

            getContentPane().add(panel);

            setPreferredSize(new Dimension(400, 250));
            pack();
        }


        private static Scene createScene() {
            Group root  =  new  Group();
            Scene  scene  =  new  Scene(root, javafx.scene.paint.Color.ALICEBLUE);
            Text text  =  new  Text();

            text.setX(40);
            text.setY(100);
            text.setFont(new javafx.scene.text.Font(25));
            text.setText("Welcome JavaFX!");

            root.getChildren().add(text);

            return (scene);
        }

//        public static void main(String[] args) throws IOException {
//            SwingUtilities.invokeLater(new Runnable() {
//
//                @Override
//                public void run() {
//                    Plotter plotter = new Plotter();
//                    TesterTrainer testerTrainer = new TesterTrainer(plotter);
//                    testerTrainer.setVisible(true);
//
//                    try {
//
//                        Trainer trainer = new Trainer(1000, 540, 8, 400, plotter, false, false);
//                        trainer.start();
//                        trainer.interrupt();
//                        System.out.println("Finished");
//
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    }
//
//                }
//
//            });
//        }


    }
