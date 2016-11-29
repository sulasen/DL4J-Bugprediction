import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by Sebi on 11/20/2016.
 */
public class SQLConnector {
    private List<String> fixedList; //contains the fixed strings
    private List<String> bugList; //contains the buggy strings
    private List<String> labelList; //contains the label strings
    private List<String> mixedList; //contains the all (fixed + buggy) strings
    private final String url = "jdbc:mysql://localhost:3306/";
    private String database = "bugfixes_ecplise"; //"bugfixes" or "bugfixes_ecplise"
    private final String user = "root";
    private String password = "admin";
    private int counter;
    private static Logger log = LoggerFactory.getLogger(Word2Vector.class);


    public SQLConnector() throws Exception {
        this.counter = 0;
    }

    public SQLConnector(int rows) throws Exception {
        this.counter = 0;
        getnextRows(rows);
    }

    public void getnextRows(int rows) throws SQLException {
        String selectStatement = "SELECT * FROM method_change LIMIT " + counter + ", " + rows;
        getResults(selectStatement);
        counter += rows;
    }


    public void getnextPercent(double percent) throws SQLException {
        String selectStatement = "SELECT * FROM method_change";
        getResults(selectStatement);
        int total = this.fixedList.size();
        int rows = (int) (total * percent) - 1;
        fixedList = fixedList.subList(counter, counter+rows);
        bugList = bugList.subList(counter, counter+rows);
        labelList = labelList.subList(2*counter, 2*counter+2*rows);
        mixedList = mixedList.subList(2*counter, 2*counter+2*rows);
        counter = counter + rows;
    }

    private void getResults(String selectStatement){
        this.fixedList = new LinkedList<String>();
        this.bugList = new LinkedList<String>();
        this.labelList = new LinkedList<String>();
        this.mixedList = new LinkedList<String>();
        try {
            //Connection to DB
            Connection conn = DriverManager.getConnection(url  + this.database, user, password);
            Statement st = conn.createStatement();
            ResultSet rs = st.executeQuery(selectStatement);
            //Convert into usable Data
            while (rs.next()) {
                String text_complete_bug = cleanString(rs.getString("pre_context") + rs.getString("old_content") + rs.getString("post_context"));
                String text_complete_fix = cleanString(rs.getString("pre_context") + rs.getString("new_content") + rs.getString("post_context"));

                this.bugList.add(text_complete_bug);
                this.fixedList.add(text_complete_fix);
                this.mixedList.add(text_complete_bug);
                this.labelList.add("bug");
                this.mixedList.add(text_complete_fix);
                this.labelList.add("fix");
            }
            conn.close();
        } catch (Exception e) {
            log.error(e.getMessage());
            System.err.println("Got an exception! ");
            System.err.println(e.getMessage());
        }
    }

    public List<String> getFixedList(){
        return this.fixedList;
    }

    public List<String> getBugList(){
        return this.bugList;
    }

    public List<String> getLabelList(){
        return this.labelList;
    }

    public List<String> getMixedList(){ return this.mixedList; }

    private String cleanString(String string){
        //Parse the String into a more useable format

        return string;
    }
}
