import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;

import java.util.List;

/**
 * Created by Sebi on 11/19/2016.
 */
public class DocIterator implements LabelAwareIterator {
    private List<String> rows;
    private List<String> labels;
    private int counter;

    public DocIterator(List<String> rows, List<String> labels) throws Exception {
        if (rows.isEmpty() || labels.isEmpty()){
            throw new Exception("Please enter a filled List");
        }
        this.rows = rows;
        this.labels = labels;
        this.counter = 0;
    }

    @Override
    public boolean hasNextDocument() {
        return rows.size() > counter;
    }

    @Override
    public LabelledDocument nextDocument() {
        String line = rows.get(counter);
        String label = labels.get(counter);
        counter++;
        LabelledDocument doc = new LabelledDocument();
        doc.setContent(line.toLowerCase());
        doc.setLabel(label);
        return doc;
    }

    @Override
    public void reset(){
        counter = 0;
    }

    @Override
    public LabelsSource getLabelsSource() {
        return new LabelsSource(labels);
    }

    public List<String> getLabels(){ return labels; }
}
