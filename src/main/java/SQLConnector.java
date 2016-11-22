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
    private List<String> allList; //contains the all (fixed + buggy) strings
    private String url;
    private int counter;
    private static Logger log = LoggerFactory.getLogger(Word2Vector.class);


    public SQLConnector(int rows) throws Exception {
        this.counter = 0;
        this.url = "jdbc:mysql://localhost:3306/bugfixes";
        getnextRows(rows);
    }

    public void getnextRows(int rows) throws SQLException {
        String selectStatement = "SELECT * FROM method_change LIMIT " + counter + ", " + rows;
        getResults(selectStatement);
        counter += rows;
    }

    private void getResults(String selectStatement){
        this.fixedList = new LinkedList<String>();
        this.bugList = new LinkedList<String>();
        this.labelList = new LinkedList<String>();
        this.allList = new LinkedList<String>();
        try {
            //Connection to DB
            Connection conn = DriverManager.getConnection(url, "root", "admin");
            Statement st = conn.createStatement();
            ResultSet rs = st.executeQuery(selectStatement);
            //Convert into usable Data
            while (rs.next()) {
                String text_complete_bug = rs.getString("pre_context") + " " + rs.getString("old_content") + " " + rs.getString("post_context");
                String text_complete_fix = rs.getString("pre_context") + " " + rs.getString("new_content") + " " + rs.getString("post_context");

                this.bugList.add(text_complete_bug);
                this.fixedList.add(text_complete_fix);
                this.allList.add(text_complete_bug);
                this.labelList.add("bug");
                this.allList.add(text_complete_fix);
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

    public List<String> getMixedList(){ return this.allList; }

    private String cleanString(String string){
        /*
        text_complete=text_complete.replace("{", " ");
        text_complete=text_complete.replace("}", " ");
        text_complete=text_complete.replace("(", " ( ");
        text_complete=text_complete.replace(")", " ) ");
        text_complete=text_complete.replace(";", " ; ");
        String[] lines = text_complete.split("\\s+");
        for(String line: lines){
            if (rowList.size()<100000){
                rowList.add(line);
            }
        }
        */
        return string;
    }
}
