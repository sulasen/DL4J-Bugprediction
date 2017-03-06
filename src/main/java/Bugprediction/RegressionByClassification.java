package Bugprediction;

import Bugprediction.Iterators.KFoldIterator;
import Bugprediction.tools.CSVWriter;
import Bugprediction.tools.CSVRecordReader;
import org.datavec.api.records.reader.RecordReader;
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
import org.jfree.util.Log;
import org.joda.time.DateTime;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

/**
 * @author Sébastien Broggi
 */
public class RegressionByClassification {
    private static Logger log = LoggerFactory.getLogger(RegressionByClassification.class);

    public static void main(String[] args) throws  Exception {
        SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH.mm");
        Date date = new Date();
        String dateString = sdfDate.format(date);
        float sumRmse = 0;
        int labelIndex = 32;     //32 Features
        int numClasses = 5;     //Number of Bugs (0-4+ Bugs)
        int numHiddenLayer = Math.round((labelIndex + numClasses)/2);
        int iterations = 250;
        long seed = 6;
        int repetitions = 100;
        //String[] projects = new String[] { "equinoxAllMetrics.csv" };
        String[] projects = new String[] { "jdtAllMetrics.csv", "luceneAllMetrics.csv", "mylynAllMetrics.csv", "pdeAllMetrics.csv", "equinoxAllMetrics.csv" };

        for (String project: projects) {
            //Get Data
            int numLinesToSkip = 0;
            float examples = 0;
            String delimiter = ",";
            CSVRecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
            recordReader.initialize(new FileSplit(new ClassPathResource(project).getFile())); //jdtAllMetrics, luceneAllMetrics, mylynAllMetrics
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 10000, labelIndex, numClasses);

            CSVWriter writer = new CSVWriter("results_" + dateString + " " + project);

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(numHiddenLayer).build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenLayer).nOut(numHiddenLayer).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(numHiddenLayer).nOut(numClasses).build())
                    .backprop(true).pretrain(false)
                    .build();

            //Set Score Listener
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            //Train and Evaluate
            DataSet allData = dataSetIterator.next();

            //Shuffle, Train and Evaluate k-foldIterations, repeat
            for (int k = 0; k < repetitions; k++) {
                allData.shuffle();
                KFoldIterator kFoldIter = new KFoldIterator(allData);
                int kCount = 0;
                while (kFoldIter.hasNext()) {
                    Log.info("KFold: " + kCount);
                    DataSet trainingData = kFoldIter.next();
                    DataSet testData = kFoldIter.testFold();
                    model.fit(trainingData);

                    //Evaluate the model on the test set
                    INDArray output = model.output(testData.getFeatureMatrix());
                    Integer[] bugs = new Integer[output.rows()];
                    Integer[] values = new Integer[output.rows()];
                    double[] error = new double[output.rows()];
                    INDArray labels = testData.getLabels();
                    for (int i = 0; i < output.rows(); i++) {
                        float highestGuess = 0;
                        float highestValue = 0;
                        for (int j = 0; j < output.columns(); j++) {
                            float guessValue = output.getFloat(i, j);
                            if (guessValue > highestGuess) {
                                highestGuess = guessValue;
                                bugs[i] = j;
                            }
                            float value = labels.getFloat(i, j);
                            if (value > highestValue) {
                                highestValue = value;
                                values[i] = j;
                            }
                        }
                    }
                    sumRmse += evaluate(bugs, values, writer, examples, project);
                    examples++;
                    kCount++;
                }
            }
            double rmse = sumRmse / examples;
            Log.info("TOTAL Average RMSE: " + rmse);
            writer.writeLine(-1, "TOTAL RMSE", 0, 0, 0, 0, rmse);
            writer.close();
        }
    }

    private static double evaluate(Integer[] bugs, Integer[] values, CSVWriter writer, float examples, String project) throws FileNotFoundException {
        //Calculate RMSE, accuracy,
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float correctBuggy = 0;
        float predictedBuggy = 0;
        float totalBuggy = 0;
        float total = 0;
        for (int i=0; i<bugs.length;i++){
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
            if (bugs[i]==values[i]){
                correct++;
            }
            if ((values[i]!=0)){
                totalBuggy++;
            }
            if ((bugs[i]!=0)){
                predictedBuggy++;
            }
            if ((bugs[i]!=0) && (values[i]!=0)){
                correctBuggy++;
            }
            if (((bugs[i]==0) && (values[i]==0))||((bugs[i]!=0) && (values[i]!=0))){
                correctClassified++;
            }
            total++;
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);
        float accuracy = correctClassified/total;
        float accuracyNumber = correct/total;
        float precision = correctBuggy/predictedBuggy;
        float recall = correctBuggy/totalBuggy;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);
        log.info("\n\t\t\t\t\t\t\tAccuracy Number: "+accuracyNumber+
                "; \n\t\t\t\t\t\t\tAccuracy Classification: "+accuracy+
                "; \n\t\t\t\t\t\t\tPrecision: "+precision+
                "; \n\t\t\t\t\t\t\tRecall: " +recall);

        writer.writeLine(((int) examples), project, accuracyNumber, accuracy, precision, recall, curRmse);

        return curRmse;
    }
}