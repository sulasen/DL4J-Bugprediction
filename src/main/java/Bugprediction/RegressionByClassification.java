package Bugprediction;

import Bugprediction.Iterators.KFoldIterator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sébastien Broggi
 */
public class RegressionByClassification {

    private static Logger log = LoggerFactory.getLogger(RegressionByClassification.class);

    public static void main(String[] args) throws  Exception {
        int labelIndex = 32;     //32 Features
        int numClasses = 10;     //10 "Bugclassses" (0-9 Bugs)
        int batchSize = 900;
        final int numInputs = 32;
        int outputNum = 10;
        int iterations = 800;
        long seed = 6;

        //Get Data
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("luceneAllMetrics.csv").getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        KFoldIterator kFoldIter = new KFoldIterator(allData);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("tanh")
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numClasses)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation("softmax")
                        .nIn(numClasses).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));


        //Train and Evaluate
        float totalError = 0;
        float examples = 0;
        while (kFoldIter.hasNext()){
            DataSet trainingData = kFoldIter.next();
            DataSet testData = kFoldIter.testFold();
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainingData);     //Apply normalization to the training data
            normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

            model.fit(trainingData);

            //evaluate the model on the test set
            INDArray output = model.output(testData.getFeatureMatrix());
            Integer[] bugs = new Integer[output.rows()];
            Integer[] values = new Integer[output.rows()];
            double[] error = new double[output.rows()];
            INDArray labels = testData.getLabels();
            for (int i=0; i<output.rows();i++) {
                float highestGuess = 0;
                float highestValue = 0;
                for (int j = 0; j < output.columns(); j++) {
                    float guessValue = output.getFloat(i, j);
                    if (guessValue > highestGuess) {
                        highestGuess = guessValue;
                        bugs[i] = j;
                    }
                    float value = labels.getFloat(i,j);
                    if (value > highestValue) {
                        highestValue = value;
                        values[i] = j;
                    }
                }
            }

            //Calculate RMSE (over all KFold-Datasets)
            for (int i=0; i<bugs.length;i++){
                error[i] = Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
                totalError += error[i];
                examples += 1;
            }
            double rmse = Math.sqrt(totalError/examples);
            log.info("RMSE: " + rmse);

            /*
            Evaluation eval = new Evaluation(numClasses);
            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());
            */
        }
    }

}