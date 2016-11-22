import java.util.List;

/**
 * Created by Sebi on 11/22/2016.
 */
public class BugClassifier {

    public static void main( String[] args ) throws Exception {
        //Make Connection to DB and get Data
        SQLConnector sqlConnector = new SQLConnector(10000);
        List<String> rowList = sqlConnector.getMixedList();
        List<String> labelList = sqlConnector.getLabelList();
        sqlConnector.getnextRows(100);
        List<String> unlabeledList1 = sqlConnector.getFixedList();
        List<String> unlabeledList2 = sqlConnector.getBugList();

        //Vectorize the Words
        Doc2Vector doc2vec = new Doc2Vector(rowList, labelList);
        doc2vec.makeParagraphVectors();
        doc2vec.checkUnlabeledData(unlabeledList1, "bug");
        doc2vec.checkUnlabeledData(unlabeledList2, "fix");

    }
}
