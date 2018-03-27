/**
 * @file Messages.java
 * @brief Massages manager for DASH player
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

import javax.swing.*;

public class Messages extends JFrame {

    private JList<String> jList1;
    private javax.swing.JScrollPane jScrollPane1;
    private DefaultListModel<String> list;

    /**
     * Creates new form Messages
     */
    public Messages(String title) {
        super(title);
        initComponents();
    }

    /**
     * Append a new message to the messages window.
     *
     * @param message text to append
     */
    public void addMessage(String message) {
        list.addElement(message);
        int lastIndex = list.getSize() - 1;
        if (lastIndex >= 0) {
            jList1.ensureIndexIsVisible(lastIndex);
        }
    }

    /**
     * Clear all the messages
     */
    public void clear() {
        list.clear();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     */
    private void initComponents() {
        jScrollPane1 = new javax.swing.JScrollPane();
        list = new DefaultListModel<>();
        jList1 = new JList<String>(list);
        setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
        jScrollPane1.setViewportView(jList1);
        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 430, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 300, Short.MAX_VALUE)
        );
        pack();
    }
}
