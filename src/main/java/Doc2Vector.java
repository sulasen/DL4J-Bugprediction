import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.LabelSeeker;
import tools.MeansBuilder;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Doc2Vector {
    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;
    List<String> rowList;
    Double right;
    int total;
    private static Logger log = LoggerFactory.getLogger(Doc2Vector.class);

    public Doc2Vector(List<String> rowList, List<String> labelList) throws Exception {
        this.rowList = rowList;
        this.right = 0.0;
        this.total = 0;
        iterator = new DocIterator(rowList, labelList);
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public void makeParagraphVectors()  throws Exception {
        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.00001)
                .batchSize(1000)
                .epochs(5)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .sequenceLearningAlgorithm(new DM())
                .build();

        // Start model training
        paragraphVectors.fit();
    }

    public void checkUnlabeledData(List<String> unlabeledList, String label) throws Exception {
        List<String> availabels = new LinkedList<String>();
        availabels.add("bug");
        availabels.add("fix");
        List<String> labels = new LinkedList<String>();
        for (String unlabeled:unlabeledList) {
            labels.add(label);
        }
        DocIterator unlabeledIterator = new DocIterator(unlabeledList, labels);
         /*
          Now we'll iterate over unlabeled data, and check which label it could be assigned to
          Please note: for many domains it's normal to have 1 document fall into few labels at once,
          with different "weight" for each.
         */
        MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(availabels, (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        Double right = 0.0;
        while (unlabeledIterator.hasNextDocument()) {
            LabelledDocument document = unlabeledIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          Print out the result
         */
            log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
            Double highest = 0.0;
            String strHighest = "";
            for (Pair<String, Double> score : scores) {
                log.info("        " + score.getFirst() + ": " + score.getSecond());
                if (score.getSecond()>highest){
                    highest = score.getSecond();
                    strHighest = score.getFirst();
                }
            }
            if (strHighest == document.getLabel()){
                right++;
            }
        }
        this.right += right;
        this.total += unlabeledList.size();

        Double accuracy = this.right / this.total;
        log.info("Total accuracy :" + accuracy);
    }
}