import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;

import java.util.List;

/**
 * Created by Sebi on 11/19/2016.
 */
public class WordIterator extends BaseSentenceIterator {
    private List<String> rows;
    private Word2Vec vec;
    private int counter;

    public WordIterator(List<String> rows, Word2Vec vec) throws Exception {
        if (rows.isEmpty()){
            throw new Exception("Please enter a filled List");
        }
        this.rows = rows;
        this.counter = 0;
    }
    @Override
    public String nextSentence(){
        String line = rows.get(counter);
        counter++;
        return line.toLowerCase();
    }

    @Override
    public void reset(){
        counter = 0;
    }

    @Override
    public boolean hasNext(){
        return rows.size() > counter;
    }

    public int inputColumns(){
        return rows.size();
    }
}
