package Bugprediction;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.SimpleName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.util.*;

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


    public void getnextPercentFiltered(double percent) throws SQLException {
        getResults();
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
                try {
                    log.info("Case: " + rs.getRow());
                    String text_complete_bug = cleanString(rs.getString("pre_context") + rs.getString("old_content") + rs.getString("post_context"));
                    String text_complete_fix = cleanString(rs.getString("pre_context") + rs.getString("new_content") + rs.getString("post_context"));

                    this.bugList.add(text_complete_bug);
                    this.fixedList.add(text_complete_fix);
                    this.mixedList.add(text_complete_bug);
                    this.labelList.add("bug");
                    this.mixedList.add(text_complete_fix);
                    this.labelList.add("fix");
                }
                catch (Exception e){}
            }
            conn.close();
        } catch (Exception e) {
            log.error(e.getMessage());
            System.err.println("Got an exception! ");
            System.err.println(e.getMessage());
        }
    }



    private void getResults(){
        this.fixedList = new LinkedList<String>();
        this.bugList = new LinkedList<String>();
        this.labelList = new LinkedList<String>();
        this.mixedList = new LinkedList<String>();
        try {
            //Connection to DB
            Connection conn = DriverManager.getConnection(url  + this.database, user, password);
            PreparedStatement pstmt = conn.prepareStatement(
                    "SELECT * FROM method_change WHERE new_content like ?");
            String transistion = "%if%(%null%)%";
            pstmt.setString(1, transistion);
            ResultSet rs = pstmt.executeQuery();
            //Convert into usable Data
            while (rs.next()) {
                try {
                    log.info("Case: " + rs.getRow());
                    String text_complete_bug = cleanString(rs.getString("pre_context") + rs.getString("old_content") + rs.getString("post_context"));
                    String text_complete_fix = cleanString(rs.getString("pre_context") + rs.getString("new_content") + rs.getString("post_context"));

                    this.bugList.add(text_complete_bug);
                    this.fixedList.add(text_complete_fix);
                    this.mixedList.add(text_complete_bug);
                    this.labelList.add("bug");
                    this.mixedList.add(text_complete_fix);
                    this.labelList.add("fix");
                }
                catch (Exception e){}
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

    private String cleanString(String text){
        //Parse the String into a more useable format
        int pCount = 0;
        int vCount = 0;
        String tempText = "";
        String finalText = "";
        //Get BodyAST
        BodyDeclaration bd = JavaParser.parseInterfaceBodyDeclaration(text);

        text = text.replace("(", " ( ");
        text = text.replace(")", " ) ");
        text = text.replace("{", " { ");
        text = text.replace("}", " } ");
        text = text.replace(",", " , ");


        //Replace Variables => TODO: Make it better
        Map<String, String> vocab = new HashMap<>();
        List<NameExpr> nameExprs = bd.getNodesByType(NameExpr.class);
        for (NameExpr variable: nameExprs) {
            List<SimpleName> nodes = variable.getNodesByType(SimpleName.class);
            SimpleName node = nodes.get(0);
            String strName = node.getIdentifier().toString();
            if (!vocab.containsKey(strName)){
                vocab.put(strName, "v"+vCount);
                text = text.replace(" " + strName + " ", " v ");
                vCount++;
            }
        }

        text = text.replace(".", " . ");

        if (text.contains("\"")){
            String[] parts = text.split("\"");
            for (int i=1; i<parts.length-1; i += 2){
                parts[i] = "\"s\"";
            }
            for (String part: parts) {
                tempText += part;
            }
        }
        if (tempText==""){
            tempText = text;
        }

        String[] parts = tempText.trim().split("\\s+");;
        for (int i=0; i<parts.length-1; i ++){
            if(!parts[i].contains("init") && !parts[i].contains("=") && !parts[i].equals("v")
                    && !parts[i].equals("if") && !parts[i].equals("else")
                    && !parts[i].contains(".") && !parts[i].contains(",")
                    && !parts[i].contains("{") && !parts[i].contains("}")
                    && !parts[i].contains("(") && !parts[i].contains(")")){
                parts[i] = "T";
            }
        }
        for (String part: parts) {
            finalText += " " + part;
        }


        text = finalText.replaceAll(" {2,}", " ");

        return finalText;
    }
}
