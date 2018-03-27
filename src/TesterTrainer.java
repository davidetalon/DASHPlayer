/**
 * Created by davidetalon on 15/05/17.
 */

import javafx.embed.swing.JFXPanel;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class TesterTrainer extends JFrame {

    private final JFXPanel jfxPanel = new JFXPanel();

    private final JPanel panel = new JPanel(new BorderLayout());

    private Plotter plotter;


    public TesterTrainer(Plotter plotter, int plotterType) {

        super();
        initComponents(plotter, plotterType);

    }

    private void initComponents(Plotter plotter, int plotterType) {
        Scene scene = createScene();
//        jfxPanel.setScene(scene);

        plotter.start(jfxPanel, plotterType);

        panel.add(jfxPanel, BorderLayout.CENTER);

        getContentPane().add(panel);

        setPreferredSize(new Dimension(1024, 600));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
    }


    private static Scene createScene() {
        Group  root  =  new  Group();
        Scene  scene  =  new  Scene(root, Color.ALICEBLUE);
        Text  text  =  new  Text();

        text.setX(40);
        text.setY(100);
        text.setFont(new Font(25));
        text.setText("Welcome JavaFX!");

        root.getChildren().add(text);

        return (scene);
    }

    public static void main(String[] args) throws IOException{
        SwingUtilities.invokeLater(new Runnable() {

            @Override
            public void run() {
                Plotter plotter = new Plotter();
                TesterTrainer testerTrainer = new TesterTrainer(plotter, 0);
                testerTrainer.setVisible(true);

                try {

                    Trainer trainer = new Trainer(1000, 540, 8, 400, false, false);
                    trainer.start();
                    trainer.interrupt();
                    System.out.println("Finished");

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }

        });
    }

}