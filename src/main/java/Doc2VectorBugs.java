import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Doc2VectorBugs {

    private static Logger log = LoggerFactory.getLogger(Doc2VectorBugs.class);

    public static void main(String[] args) throws Exception {
        List<String> rowList = null;
        List<String> labelList = null;
        try {
            String url = "jdbc:mysql://localhost:3306/bugfixes";
            Connection conn = DriverManager.getConnection(url, "root", "admin");
            Statement st = conn.createStatement();
            ResultSet rs = st.executeQuery("SELECT * FROM method_change");
            rowList = new LinkedList<String>();
            labelList = new LinkedList<String>();
            while (rs.next()) {
                //String[] lines = rs.getString("pre_context").split(System.getProperty("line.separator"));
                String buggy = rs.getString("pre_context")+rs.getString("old_content")+rs.getString("post_context");
                String fixed = rs.getString("pre_context")+rs.getString("new_content")+rs.getString("post_context");
                if (rowList.size()<100){
                    rowList.add(buggy);
                    labelList.add("bug");
                    rowList.add(fixed);
                    labelList.add("fixed");
                }
            }
            conn.close();
        } catch (Exception e) {
            log.error(e.getMessage());
            System.err.println("Got an exception! ");
            System.err.println(e.getMessage());
        }



        log.info("Load & Vectorize Sentences....");

        /*
        // Strip white space before and after for each line
        SentenceIterator iter = new LineSentenceIterator(new File("/Users/ivo/Desktop/raw_sentences.txt"));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        */


        SentenceIterator iter = new RowIterator(rowList);

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors to file
        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println("10 Words closest to 'day': " + lst);

        UiServer server = UiServer.getInstance();
        System.out.println("Started on port " + server.getPort());
    }

    private static class RowIterator extends BaseSentenceIterator {
        private List<String> rows;

        public RowIterator(List<String> rows) throws Exception {
            if (rows.isEmpty()){
                throw new Exception("Please enter a filled List");
            }
            this.rows = rows;
        }
        @Override
        public String nextSentence(){
            if (rows.listIterator().hasNext()){
                String line = rows.listIterator().next();
                return line.toLowerCase();
            }
            return null;
        }

        @Override
        public void reset(){
            while (rows.listIterator().hasPrevious()){
                rows.listIterator().previous();
            }
        }

        @Override
        public boolean hasNext(){
            return rows.listIterator().hasNext();
        }
    }
}