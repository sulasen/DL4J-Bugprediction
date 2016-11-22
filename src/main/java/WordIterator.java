import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.*;
import org.nd4j.linalg.api.rng.Random;
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
    //position of the cursor
    private int position;
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
        int currMinibatchSize = Math.min(num, allWords.size()-1);
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,word2Vec.getLayerSize(),exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,word2Vec.getLayerSize(),exampleLength}, 'f');

        if (position + currMinibatchSize > allWords.size()-1){
            currMinibatchSize = allWords.size() - position-1;
        }

        for( int i=0; i<currMinibatchSize; i++ ){
            for( int j=0; j<exampleLength; j++){
                double[] currWord = word2Vec.getWordVector(allWords.get(position));	//Current input
                int wordArrayIndex = 0;
                for (double value:currWord){
                    input.putScalar(new int[]{i,wordArrayIndex,j}, value);
                    labels.putScalar(new int[]{i,wordArrayIndex++,j}, value);
                }
                if (position<allWords.size()-1)  position++;
            }
        }

        return new DataSet(input,labels);
    }

    @Override
    public int totalExamples() {
        return (allWords.size()-1)/miniBatchSize;
    }

    @Override
    public int inputColumns() {
        return word2Vec.getLayerSize();
    }

    @Override
    public int totalOutcomes() {
        return word2Vec.getLayerSize();
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
        position = 0;
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return position;
    }

    @Override
    public int numExamples() {
        return totalExamples();
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
        return position<allWords.size()-1;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public void remove() {

    }

    public String getRandomWord(){
        java.util.Random rng = new java.util.Random(allWords.size());
        String word = allWords.get(rng.nextInt(allWords.size()-1));
        return word;
    }

    public double[] getVector(String word){
        double[] vector = word2Vec.getWordVector(word);
        if (vector == null || vector.length<1){
            vector = new double[word2Vec.getLayerSize()];
        }
        return vector;
    }

    public String getWord(double[] vector){
        INDArray wordArray = Nd4j.create(vector.length);
        int i = 0;
        for (double value : vector) {
            wordArray.putScalar(i, value);
            i++;
        }
        Collection<String> word = word2Vec.wordsNearest(wordArray, 1);
        String strWord = word.toString();
        return strWord;
    }
}
