import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Created by Sebi on 11/19/2016.
 */
public class WordIterator implements DataSetIterator {
    //Valid words
    private Word2Vec word2Vec;
    //All words of the input file (after filtering to only those that are valid
    private List<String> allWords;
    //Length of each example/minibatch (number of characters)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;

    private int position;
    private Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();


    public WordIterator(Word2Vec vec, List<String> rowList, int miniBatchSize, int exampleLength){
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        this.miniBatchSize = miniBatchSize;
        this.word2Vec = vec;
        this.exampleLength = exampleLength;

        int maxSize = rowList.size();	//add lines.size() to account for newline characters at end of each line
        for( String s : rowList ) maxSize += s.split(" ").length;
        List<String> words = new LinkedList<String>();
        int currIdx = 0;
        for( String line : rowList ){
            String[] wordsInLine = line.split("\\s+");
            for (String word : wordsInLine){
                double[] wordVector = vec.getWordVector(word);
                if (wordVector != null && wordVector.length > 1){
                    words.add(currIdx++, word);
                }
            }
        }
        this.allWords = words;
    }


    @Override
    public DataSet next(int num) {
        int currMinibatchSize = Math.min(num, allWords.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,word2Vec.getVocab().numWords(),exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,word2Vec.getVocab().numWords(),exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            for( int j=0; j<exampleLength; j++){
                double[] currWord = word2Vec.getWordVector(allWords.get(position));	//Current input
                int wordArrayIndex = 0;
                for (double value:currWord){
                    input.putScalar(new int[]{i,wordArrayIndex,j}, value);
                    labels.putScalar(new int[]{i,wordArrayIndex++,j}, value);
                }
                position++;
            }
        }

        return new DataSet(input,labels);
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return position<allWords.size();
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public void remove() {

    }
}
