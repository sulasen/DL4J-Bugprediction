import java.util.List;

/**
 * Created by Sebi on 11/22/2016.
 */
public class BugClassifier {

    public static void main( String[] args ) throws Exception {
        double percentTraining = 0.9;
        double percentTesting = 0.1;
        //Make Connection to DB and get Data
        SQLConnector sqlConnector = new SQLConnector();

        //Get List with bugs & their fixes, labelled accordingly
        sqlConnector.getnextPercent(percentTraining);
        List<String> rowList = sqlConnector.getMixedList();
        List<String> labelList = sqlConnector.getLabelList();

        //Get List with 'unlabelled' data to test accuracy
        sqlConnector.getnextPercent(percentTesting);
        List<String> unlabeledList1 = sqlConnector.getFixedList();
        List<String> unlabeledList2 = sqlConnector.getBugList();

        //Vectorize the Paragraphs
        Doc2Vector doc2vec = new Doc2Vector(rowList, labelList);
        doc2vec.makeParagraphVectors();
        //Check accuracy
        doc2vec.checkUnlabeledData(unlabeledList1, "fix");
        doc2vec.checkUnlabeledData(unlabeledList2, "bug");
    }
}
