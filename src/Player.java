/**
 * @file Player.java
 * @brief Player for DASH protocol
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


import uk.co.caprica.vlcj.discovery.NativeDiscovery;
import uk.co.caprica.vlcj.medialist.MediaList;
import uk.co.caprica.vlcj.player.MediaPlayerFactory;
import uk.co.caprica.vlcj.player.embedded.EmbeddedMediaPlayer;
import uk.co.caprica.vlcj.player.embedded.videosurface.CanvasVideoSurface;
import uk.co.caprica.vlcj.player.list.MediaListPlayer;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.logging.*;
import java.util.List;
import java.util.ArrayList;



public class Player extends javax.swing.JFrame {

    // Variables declaration
    private boolean videoAlreadyStarted;
    private boolean trainingAlreadyStarted;
    private JComboBox<String> jComboBox1;
    private JComboBox<String> jComboBoxTrainingTypes;
    private JLabel jLabel1;
    private JButton messagesButton;
    private JButton playButton;
    private JButton stopButton;
    private JButton startTrainingButton;
    private JButton statsButton;
    private JTextField urlTextfield;
    private Canvas videoCanvas;
    private JPanel videoContainer;
    private EmbeddedMediaPlayer mediaPlayer;
    private MediaListPlayer mediaListPlayer;
    private DashAlgorithm dash;
    private String tempFolderPath;
    private MediaList mediaList;
    private FileHandler fh;
    private static Messages messagesFrame;
    private static Logger logger;
    private static Logger MPDLog;

    private Trainer trainer;

    private Plotter bufferPlotter;
    private Plotter rewardPlotter;
    private Plotter qualityPlotter;


    private RenderPlot bufferRender;
    private RenderPlot rewardRender;
    private RenderPlot qualityRender;



    /**
     * Creates new form NewJFrame
     */
    public Player(String name) {
        super(name);

        // Init graphical components
        initComponents();
        videoAlreadyStarted = false;
        trainingAlreadyStarted = false;

        // Create required folder if not present
        tempFolderPath = System.getProperty("user.dir") + File.separator + "temp" + File.separator;
        File downloadFolder = new File(tempFolderPath);
        if (!downloadFolder.exists()) {
            downloadFolder.mkdir();
        }
        downloadFolder = new File(tempFolderPath + "init" + File.separator);
        if (!downloadFolder.exists()) {
            downloadFolder.mkdir();
        }
        downloadFolder = new File(tempFolderPath + "seg" + File.separator);
        if (!downloadFolder.exists()) {
            downloadFolder.mkdir();
        }

        // Create logger
        logger = Logger.getLogger("MyLog");
        MPDLog = Logger.getLogger("MDPLog");

        String logFolderPath = System.getProperty("user.dir") + File.separator + "logs" + File.separator;
        File logFolder = new File(logFolderPath);
        if (!logFolder.exists()) {
            logFolder.mkdir();
        }

        boolean exists = false;
        String logFilePath = "";
        for(int i = 0; !exists; i++){
            logFilePath = "logs/" + "MPD" + i + "log";
            File log = new File(logFilePath);
            if (!log.exists()){
                exists = true;
            }
        }

        System.out.println(logFilePath);
        FileHandler fh = null;
        try {
            fh = new FileHandler(logFilePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        SimpleFormatter formatter = new SimpleFormatter();


        fh.setFormatter(formatter);
        MPDLog.addHandler(fh);
        MPDLog.setUseParentHandlers(false);


        // Create hidden Frame for messages
        messagesFrame = new Messages("Messages");
        Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
        messagesFrame.setLocation((int) getLocation().getX() + this.getWidth(), this.getY());

    }


    /**
     * This method is called from within the constructor to initialize the form.
     */
    @SuppressWarnings("unchecked")
    private void initComponents() {
        //**************************************************//
        // Graphics stuff
        videoContainer = new javax.swing.JPanel();
        videoCanvas = new java.awt.Canvas();
        jLabel1 = new javax.swing.JLabel();
        urlTextfield = new javax.swing.JTextField();
        playButton = new javax.swing.JButton();
        stopButton = new javax.swing.JButton();
        startTrainingButton = new javax.swing.JButton();
        statsButton = new javax.swing.JButton();
        jComboBox1 = new javax.swing.JComboBox<>();
        jComboBoxTrainingTypes = new javax.swing.JComboBox<>();
        messagesButton = new javax.swing.JButton();
        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setPreferredSize(new java.awt.Dimension(800, 585));
        videoContainer.setPreferredSize(new java.awt.Dimension(551, 353));
        videoCanvas.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
        videoCanvas.setName("videoCanvas");
        javax.swing.GroupLayout videoContainerLayout = new javax.swing.GroupLayout(videoContainer);
        videoContainer.setLayout(videoContainerLayout);
        videoContainerLayout.setHorizontalGroup(
                videoContainerLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(videoCanvas, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        videoContainerLayout.setVerticalGroup(
                videoContainerLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(videoCanvas, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        jLabel1.setText("Media URL:");
        playButton.setText("Play");
        playButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                playButtonActionPerformed(evt);
            }
        });
        stopButton.setText("Stop");
        stopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                stopButtonActionPerformed(evt);
            }
        });
        messagesButton.setText("Messages");
        messagesButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                messagesButtonActionPerformed(evt);
            }
        });
        startTrainingButton.setText("Start Training");
        startTrainingButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                startTrainingButtonActionPerformed(evt);
            }
        });
        statsButton.setText("Stats");
        statsButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                statsButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(videoContainer, javax.swing.GroupLayout.DEFAULT_SIZE, 801, Short.MAX_VALUE)
                        .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                        .addGroup(layout.createSequentialGroup()
                                                .addGap(20, 20, 20)
                                                .addComponent(jLabel1)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(urlTextfield))
                                        .addGroup(layout.createSequentialGroup()
                                                .addContainerGap()
                                                .addComponent(playButton, javax.swing.GroupLayout.PREFERRED_SIZE, 90, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(stopButton, javax.swing.GroupLayout.PREFERRED_SIZE, 90, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(messagesButton, javax.swing.GroupLayout.PREFERRED_SIZE, 90, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(statsButton)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                                .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE))
                                        .addGroup(layout.createSequentialGroup()
                                                .addContainerGap()
                                                .addComponent(startTrainingButton, GroupLayout.DEFAULT_SIZE, 90, GroupLayout.DEFAULT_SIZE)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                                .addComponent(jComboBoxTrainingTypes, GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE)))
                                .addContainerGap())
        );
        layout.setVerticalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addGroup(layout.createSequentialGroup()
                                .addContainerGap()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                        .addComponent(jLabel1)
                                        .addComponent(urlTextfield, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(videoContainer, javax.swing.GroupLayout.DEFAULT_SIZE, 509, Short.MAX_VALUE)
                                .addGap(18, 18, 18)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                        .addComponent(jComboBox1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(playButton)
                                        .addComponent(stopButton)
                                        .addComponent(messagesButton)
                                        .addComponent(statsButton))

                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                        .addComponent(jComboBoxTrainingTypes, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(startTrainingButton))
                                .addContainerGap())
        );

        // Set algorithms names to combobox
            jComboBox1.setModel(new javax.swing.DefaultComboBoxModel<>(new String[]{"Bitrate-based", "Buffer-based", "FESTIVE", "D-DASH: MLP2", "MPC", "PANDA"}));

        // Set type of training to combobox
        jComboBoxTrainingTypes.setModel(new javax.swing.DefaultComboBoxModel<>(new String[]{"server", "local"}));

        // Set default link to the interface
//        TRAINING
//        urlTextfield.setText("http://www.secpoint.it/matteo/dash/video/");
        urlTextfield.setText("http://www.secpoint.it/matteo/dash/video/bigbuckbunny/bigbuckbunny-simple.mpd");


        // End graphics stuff
        //**************************************************//


        List<String> vlcArgs = new ArrayList<String>();
//        vlcArgs.add("-vvv");
//        vlcArgs.add("--no-plugins-cache");
//        vlcArgs.add("--avcodec-fast");
        vlcArgs.add("--avcodec-skiploopfilter=4");
        vlcArgs.add("--file-caching=70");
        vlcArgs.add("--avcodec-hw=any");
//        vlcArgs.add("--avcodec-codec=H264");
        vlcArgs.add("--avcodec-threads=1");
//        vlcArgs.add("--no-avcodec-hurry-up");
//        vlcArgs.add(" --no-skip-frames");

        // MediaPlayer
        MediaPlayerFactory mediaPlayerFactory = new MediaPlayerFactory(vlcArgs);
//        MediaPlayerFactory mediaPlayerFactory = new MediaPlayerFactory();
        CanvasVideoSurface videoSurface = mediaPlayerFactory.newVideoSurface(videoCanvas);





        mediaPlayer = mediaPlayerFactory.newEmbeddedMediaPlayer();
        mediaPlayer.setVideoSurface(videoSurface);
        mediaList = mediaPlayerFactory.newMediaList();
        mediaListPlayer = mediaPlayerFactory.newMediaListPlayer();

        mediaListPlayer.setMediaList(mediaList);
        mediaListPlayer.setMediaPlayer(mediaPlayer);
        mediaPlayer.addMediaPlayerEventListener(new PlayerEventListener());

        pack();

    }



    /**
     * Resize properly the video canvas.
     * This method is called automatically when the form is resized.
     */
    private void formComponentResized(java.awt.event.ComponentEvent evt) {
        if (videoAlreadyStarted) {
            videoCanvas.setSize(getRootPane().getWidth(), getRootPane().getHeight() - 120);
        } else {
            videoCanvas.setSize(videoContainer.getWidth(), videoContainer.getHeight() - 120);
        }
    }

    /**
     * Define the behavior of the playButton.
     * This method is called automatically when the playButton is clicked.
     */
    private void playButtonActionPerformed(java.awt.event.ActionEvent evt) {


//        mediaList.addMedia( "/home/davidetalon/Scrivania/seg1.mp4");
//        mediaList.addMedia( "/home/davidetalon/Scrivania/seg2.mp4");
//        mediaList.addMedia( "/home/davidetalon/Scrivania/seg3.mp4");
//        mediaList.addMedia( "/home/davidetalon/Scrivania/seg4.mp4");
//
//
//        mediaListPlayer.play();



        if (!videoAlreadyStarted) {
            // This block configure the logger with handler and formatter
            try {
                fh = new FileHandler(tempFolderPath + File.separator + "DashAlgorithmLog.log");
                logger.addHandler(fh);
                fh.setFormatter(new Formatter() {
                    @Override
                    public String format(LogRecord record) {
                        SimpleDateFormat logTime = new SimpleDateFormat("MM-dd-yyyy HH:mm:ss");
                        Calendar cal = new GregorianCalendar();
                        cal.setTimeInMillis(record.getMillis());
                        return " " + logTime.format(cal.getTime())
                                + " || "
                                + record.getMessage() + "\n";
                    }
                });
                logger.setUseParentHandlers(false);
            } catch (SecurityException | IOException e) {
                System.err.println(e.getMessage());
            }
            videoAlreadyStarted = true;
            String algorithmChoice = (String) jComboBox1.getSelectedItem();
            switch (algorithmChoice) {
                case "Bitrate-based":
                    dash = new BitRateBasedDashAlgorithm(mediaListPlayer, tempFolderPath, urlTextfield.getText());
                    break;
                case "Buffer-based":
                    dash = new BufferBasedDashAlgorithm(mediaListPlayer, tempFolderPath, urlTextfield.getText());
                    break;
                case "FESTIVE":
                    dash = new FestiveDashAlgorithm(mediaListPlayer, tempFolderPath, urlTextfield.getText());
                    break;
                case "D-DASH: MLP2":
                    dash = new MLP2(mediaListPlayer, tempFolderPath, urlTextfield.getText());
                    break;
                case "MPC":
                    dash = new MPCDashAlgorithm(mediaListPlayer, tempFolderPath, urlTextfield.getText());
                    break;
                case "PANDA":
                    dash = new PANDADashAlgorithm(mediaListPlayer, tempFolderPath, urlTextfield.getText());
            }
            jComboBox1.setEnabled(false);

            bufferPlotter = new Plotter();
            rewardPlotter = new Plotter();
            qualityPlotter = new Plotter();


            bufferRender = new RenderPlot(bufferPlotter, RenderPlot.BUFFER_PLOT);
            rewardRender = new RenderPlot(rewardPlotter, RenderPlot.REWARD_PLOT);
            qualityRender = new RenderPlot(qualityPlotter, RenderPlot.QUALITY_PLOT);

            dash.setPlotters(bufferPlotter, qualityPlotter, rewardPlotter);

            dash.start();
            addMessage("PLAYER: new DASH algorithm started");
            addLog("PLAYER: new DASH algorithm started");
            playButton.setText("Pause");
        } else {
            if (mediaPlayer.isPlaying()) {
                playButton.setText("Play");
                mediaListPlayer.pause();
            } else {
                playButton.setText("Pause");
                mediaListPlayer.pause();
            }
        }
    }

    /**
     * Define the behavior of the stopButton.
     * This method is called automatically when the stopButton is clicked.
     */
    private void stopButtonActionPerformed(java.awt.event.ActionEvent evt) {
        mediaPlayer.stop();
        mediaListPlayer.stop();
        dash.closeMDPSession();
        mediaList.clear();
        playButton.setText("Play");
        videoAlreadyStarted = false;
        jComboBox1.setEnabled(true);
        messagesFrame.clear();
        PlayerEventListener.segIndex = 1;
        fh.close();
    }

    /**
     * Define the behavior of the messageButton.
     * This method is called automatically when the messageButton is clicked.
     */
    private void messagesButtonActionPerformed(java.awt.event.ActionEvent evt) {
        if (messagesFrame.isVisible()) {
            messagesFrame.setVisible(false);
        } else {
            messagesFrame.setLocation(getLocation().x + getWidth() - 16, getLocation().y);
            messagesFrame.setSize(new Dimension(messagesFrame.getWidth(), this.getHeight()));
            messagesFrame.setVisible(true);
        }

    }

    private void startTrainingButtonActionPerformed(java.awt.event.ActionEvent evt) {

        if (trainingAlreadyStarted) {

            trainer.forceInterrupt();
            startTrainingButton.setText("Start training");
            playButton.setEnabled(true);
            jComboBox1.setEnabled(true);
            jComboBoxTrainingTypes.setEnabled(true);


        } else {

            playButton.setEnabled(false);
            jComboBox1.setEnabled(false);
            jComboBoxTrainingTypes.setEnabled(false);
            trainingAlreadyStarted = true;
            startTrainingButton.setText("Stop training");
            String trainingType = (String) jComboBoxTrainingTypes.getSelectedItem();

            bufferPlotter = new Plotter();
            rewardPlotter = new Plotter();
            qualityPlotter = new Plotter();


            bufferRender = new RenderPlot(bufferPlotter, RenderPlot.BUFFER_PLOT);
            rewardRender = new RenderPlot(rewardPlotter, RenderPlot.REWARD_PLOT);
            qualityRender = new RenderPlot(qualityPlotter, RenderPlot.QUALITY_PLOT);

            //local training whith exponential scenes and markov channel
            if (trainingType.equals("local")) {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {

                        try {

                            trainer = new Trainer(1000, 540, 8, 400, false, false);
                            trainer.setPlotters(bufferPlotter, rewardPlotter, qualityPlotter);
                            trainer.start();


                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }

                });

                addMessage("TRAINER: new local training started");
                addLog("TRAINER: new local training started");

            //training using fake MPDs downloaded from server
            } else {

                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {

                        try {

                            trainer = new Trainer(1000, 540, 8, 400, false, true);
                            trainer.setPlotters(bufferPlotter, rewardPlotter, qualityPlotter);
                            trainer.setSourceUrl(urlTextfield.getText());
                            trainer.setTempFolderPath(tempFolderPath);
                            trainer.start();


                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }

                });

                addMessage("TRAINER: new local training started");
                addLog("TRAINER: new local training started");

            }



        }

    }

    private void statsButtonActionPerformed(java.awt.event.ActionEvent evt) {

        if (bufferRender.isVisible() || rewardRender.isVisible() || qualityRender.isVisible()) {

            bufferRender.setVisible(false);
            rewardRender.setVisible(false);
            qualityRender.setVisible(false);

        } else {


            //TODO posizionare statistiche
            bufferRender.setLocation(getLocation().x + getWidth() - 16, getLocation().y);
            bufferRender.setVisible(true);

            rewardRender.setLocation(getLocation().x + getWidth() - 16, getLocation().y);
            rewardRender.setVisible(true);

            qualityRender.setLocation(getLocation().x + getWidth() - 16, getLocation().y);
            qualityRender.setVisible(true);
        }


    }


    /**
     * Append a new message to the messages window.
     *
     * @param message text to append
     */
    public static void addMessage(String message) {
        messagesFrame.addMessage(message);
    }

    /**
     * Append a new message to the log file.
     *
     * @param message text to append
     */
    public static void addLog(String message) {
        logger.info(message);
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        // Set look of OS
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e) {
            System.err.println("Error on setting look and feel");
        }
        // Find needed VLC library
        new NativeDiscovery().discover();

        // Create and display the form
        java.awt.EventQueue.invokeLater(() -> {
            Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
            Player g = new Player("DASH_Player");

            g.setLocation(dim.width / 2 - g.getSize().width / 2, dim.height / 2 - g.getSize().height / 2);
            g.setVisible(true);

        });
    }


}


