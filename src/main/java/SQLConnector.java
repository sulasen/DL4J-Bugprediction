import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by Sebi on 11/20/2016.
 */
public class SQLConnector {
    private List<String> rowList;
    private static Logger log = LoggerFactory.getLogger(Word2Vector.class);

    public SQLConnector(int rows) throws Exception {
        this.rowList = new LinkedList<String>();
        try {
            //Connection to DB
            String url = "jdbc:mysql://localhost:3306/bugfixes";
            Connection conn = DriverManager.getConnection(url, "root", "admin");
            Statement st = conn.createStatement();
            ResultSet rs = st.executeQuery("SELECT * FROM method_change");
            ResultSetMetaData meta = rs.getMetaData();

            //Convert into usable Data
            while (rs.next()) {
                String text_complete = rs.getString("pre_context") + " " + rs.getString("new_content") + " " + rs.getString("post_context");
                if (this.rowList.size()<rows) {
                    this.rowList.add(text_complete);
                }
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
            }
            conn.close();
        } catch (Exception e) {
            log.error(e.getMessage());
            System.err.println("Got an exception! ");
            System.err.println(e.getMessage());
        }
    }

    public List<String> getRowList(){
        return this.rowList;
    }
}
