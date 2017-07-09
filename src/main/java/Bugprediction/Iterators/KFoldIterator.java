package Bugprediction.Iterators;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sebi on 1/29/2017.
 */
public class KFoldIterator implements DataSetIterator{
    private DataSet singleFold;
    private int k;
    private int batch;
    private int buggyCount;
    private int fixedCount;
    private int lastBatch;
    private int kCursor=0;
    private DataSet test;
    private DataSet train;
    private DataSet buggy;
    private DataSet fixed;
    protected DataSetPreProcessor preProcessor;

    public KFoldIterator(DataSet singleFold) {
        this(10,singleFold);
    }
    /**Create an iterator given the dataset and a value of k (optional, defaults to 10)
     * If number of samples in the dataset is not a multiple of k, the last fold will have less samples with the rest having the same number of samples.
     *
     * @param k number of folds (optional, defaults to 10)
     * @param singleFold DataSet to split into k folds
     */

    public KFoldIterator (int k, DataSet singleFold) {
        this.k = k;
        this.singleFold = singleFold.copy();
        if (k <= 1) throw new IllegalArgumentException();
        if (singleFold.numExamples() % k !=0 ) {
            this.batch = Math.round(singleFold.numExamples()/k);
            this.lastBatch = this.batch + singleFold.numExamples() % this.batch;
        }
        else {
            this.batch = singleFold.numExamples() / k;
            this.lastBatch = singleFold.numExamples() / k;
        }

        //Stratification:
        //Calculate how much percentage the different labels (buggy/not buggy) are in the set and separate them
        INDArray labels = singleFold.getLabels();
        Integer[] values = new Integer[labels.rows()];
        for (int i=0; i<labels.rows();i++) {
            float highestValue = 0;
            for (int j = 0; j < labels.columns(); j++) {
                float value = labels.getFloat(i,j);
                if (value > highestValue) {
                    highestValue = value;
                    values[i] = j;
                }
            }
        }

        List<DataSet> fixedDataSets = new ArrayList<DataSet>();
        List<DataSet> buggyDataSets = new ArrayList<DataSet>();
        for (int i=0; i<labels.rows();i++) {
            if (singleFold.get(i) != null){
                if (values[i]>0){
                    buggyDataSets.add(singleFold.get(i));
                    buggyCount++;
                }
                else{
                    fixedDataSets.add(singleFold.get(i));
                    fixedCount++;
                }
            }
        }

        fixed = DataSet.merge(fixedDataSets);
        buggy = DataSet.merge(buggyDataSets);
    }

    @Override
    public DataSet next(int num) throws UnsupportedOperationException {
        return null;
    }

    /**
     * Returns total number of examples in the dataset (all k folds)
     *
     * @return total number of examples in the dataset including all k folds
     */
    @Override
    public int totalExamples() {
        return singleFold.getLabels().size(0);
    }

    @Override
    public int inputColumns() {
        return singleFold.getFeatures().size(1);
    }

    @Override
    public int totalOutcomes() {
        return singleFold.getLabels().size(1);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    /**
     * Shuffles the dataset and resets to the first fold
     *
     * @return void
     */
    @Override
    public void reset() {
        //shuffle and return new k folds
        singleFold.shuffle();
        kCursor = 0;
    }


    /**
     * The number of examples in every fold, except the last if totalexamples % k !=0
     *
     * @return examples in a fold
     */
    @Override
    public int batch() {
        return batch;
    }

    /**
     * The number of examples in the last fold
     * if totalexamples % k == 0 same as the number of examples in every other fold
     *
     * @return examples in the last fold
     */
    public int lastBatch() {
        return lastBatch;
    }

    /**
     * cursor value of zero indicates no iterations and the very first fold will be held out as test
     * cursor value of 1 indicates the the next() call will return all but second fold which will be held out as test
     * curson value of k-1 indicates the next() call will return all but the last fold which will be held out as test
     *
     * @return the value of the cursor which indicates which fold is held out as a test set
     */

    @Override
    public int cursor() {
        return kCursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return singleFold.getLabelNamesList();
    }

    @Override
    public boolean hasNext() {
        return kCursor < k;
    }

    @Override
    public DataSet next() {
        nextFold();
        return train;
    }

    @Override
    public void remove() {
        // no-op
    }

    private void nextFold() {
        int left;
        int right;
        if (kCursor == k - 1) {
            left = totalExamples() - lastBatch;
            right = totalExamples();
        }
        else {
            left = kCursor*batch;
            right = left+batch;
        }

        List<DataSet> kMinusOneFoldList = new ArrayList<DataSet>();
        int buggyLeft = left * buggyCount / singleFold.numExamples();
        int buggyRight = right * buggyCount / singleFold.numExamples();
        int fixedLeft = left * fixedCount / singleFold.numExamples();
        int fixedRight = right * fixedCount / singleFold.numExamples();

        //Factor by which there are more cases of one type for oversampling
        float factor1=1;
        float factor2=1;
        if (fixedCount>2*buggyCount){
            factor1 = (float)(fixedCount/buggyCount)/1.5f;
        }
        else if (buggyCount>2*fixedCount){
            factor2 = (float)(buggyCount/fixedCount)/1.5f;
        }
        if (buggyRight<buggy.getLabels().size(0) && fixedRight<fixed.getLabels().size(0)) {
            //fixed examples
            for (int i=0; i<factor2; i++) {
                if (fixedLeft > 0)
                    kMinusOneFoldList.add((DataSet) fixed.getRange(0, fixedLeft));
                kMinusOneFoldList.add((DataSet) fixed.getRange(fixedRight, fixed.getLabels().size(0)));
            }
            //Buggy examples
            for (int i=0; i<factor1; i++){
                if (buggyLeft>0)
                    kMinusOneFoldList.add((DataSet) buggy.getRange(0,buggyLeft));
                kMinusOneFoldList.add((DataSet) buggy.getRange(buggyRight,buggy.getLabels().size(0)));
            }
            train = DataSet.merge(kMinusOneFoldList);
        }
        else {
            List<DataSet> trainset = new ArrayList<DataSet>();
            for (int i=0; i<factor2; i++) {
                if (fixedLeft > 0)
                    trainset.add((DataSet) fixed.getRange(0, fixedLeft));
            }
            for (int i=0; i<factor1; i++){
                if (buggyLeft>0)
                    trainset.add((DataSet) buggy.getRange(0,buggyLeft));
            }
            train = DataSet.merge(trainset);
        }
        List<DataSet> testsets = new ArrayList<DataSet>();
        testsets.add((DataSet) buggy.getRange(buggyLeft,buggyRight));
        testsets.add((DataSet) fixed.getRange(fixedLeft,fixedRight));
        test = DataSet.merge(testsets);

        train.shuffle();
        test.shuffle();

        kCursor++;
    }

    public DataSet testFold() {
        return test;
    }
}
