package Bugprediction;

import Bugprediction.Iterators.KFoldIterator;
import Bugprediction.tools.CSVWriter;
import Bugprediction.tools.CSVRecordReader;
import Bugprediction.tools.Evaluator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.util.Log;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * @author Sébastien Broggi
 */
public class BugClassificationMetrics {
    private static Logger log = LoggerFactory.getLogger(BugClassificationMetrics.class);

    public static void main(String[] args) throws  Exception {
        Evaluator classifier = new Evaluator();
        Evaluator regression = new Evaluator();
        Evaluator regByClass = new Evaluator();
        Evaluator classByReg1 = new Evaluator();
        Map<String, Integer> projectDict = new HashMap<>();
        projectDict.put("mylynAllMetrics.csv",32);
        projectDict.put("jdtAllMetrics.csv", 32);
        projectDict.put("luceneAllMetrics.csv", 32);
        projectDict.put("pdeAllMetrics.csv", 32);
        projectDict.put("equinoxAllMetrics.csv", 32);
        projectDict.put("camel-1.2.csv", 20);
        //projectDict.put("camel-1.6.csv", 20);
        projectDict.put("prop-1.csv", 20);
        projectDict.put("prop-2.csv", 20);
        projectDict.put("prop-3.csv", 20);
        projectDict.put("prop-4.csv", 20);
        projectDict.put("prop-5.csv", 20);
        projectDict.put("xalan-2.6.csv", 20);
        projectDict.put("xalan-2.5.csv", 20);
        projectDict.put("xerces-1.4.csv", 20);

        setupDir("summary");
        SimpleDateFormat sdfDate1 = new SimpleDateFormat("yyyy-MM-dd_HH.mm");
        String dateString1 = sdfDate1.format(new Date());
        CSVWriter writerSummary = new CSVWriter("results/summary/Summary_"+dateString1+".csv");
        writerSummary.setupSummary();

        int[] numClassesArr = {2,3,5,10};

        Iterator it = projectDict.entrySet().iterator();
        while (it.hasNext()) {

            Map.Entry<String, Integer> pair = (Map.Entry) it.next();
            String project = pair.getKey();
            int labelIndex = pair.getValue();     //Number of Features
            for (int c : numClassesArr) {
                int numClasses = c;         //Different numbers of Bugs
                int numHiddenLayer = Math.round((labelIndex + numClasses) / 2);
                int iterations = 200;
                long seed = 6;
                int repetitions = 100;  //Number of repetitions per project
                int kFold = 5;
                double learningRate = 0.1;

                //Get Data from csv
                CSVRecordReader recordReader = new CSVRecordReader(0, ",", numClasses);
                recordReader.initialize(new FileSplit(new ClassPathResource(project).getFile()));
                DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 10000, labelIndex, numClasses);
                DataSet allData = dataSetIterator.next();

                //Generate writers
                SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd_HH.mm");
                String dateString = sdfDate.format(new Date());
                setupDir(project);
                CSVWriter writerRbC = new CSVWriter("results/" + project.replace(".csv", "") + "/results_" + dateString + " RbC - " + project);
                writerRbC.writeConfig(numClasses, numHiddenLayer, iterations, kFold, learningRate);
                CSVWriter writerCbR = new CSVWriter("results/" + project.replace(".csv", "") + "/results_" + dateString + " CbR - " + project);
                writerCbR.writeConfig(numClasses, numHiddenLayer, iterations, kFold, learningRate);
                CSVWriter writerReg = new CSVWriter("results/" + project.replace(".csv", "") + "/results_" + dateString + " Regression - " + project);
                writerReg.writeConfig(numClasses, numHiddenLayer, iterations, kFold, learningRate);
                CSVWriter writerCla = new CSVWriter("results/" + project.replace(".csv", "") + "/results_" + dateString + " Classification - " + project);
                writerCla.writeConfig(numClasses, numHiddenLayer, iterations, kFold, learningRate);

                log.info("Build model....");
                // 3 Layer Config
                MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .iterations(iterations)
                        .weightInit(WeightInit.XAVIER)
                        .learningRate(learningRate)
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(labelIndex).build())
                        .layer(1, new DenseLayer.Builder().nIn(labelIndex).nOut(numHiddenLayer).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation("softmax").nIn(numHiddenLayer).nOut(numClasses).build())
                        .backprop(true)
                        .pretrain(false)
                        .build();


                // 2 Layer Config
                MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .iterations(iterations)
                        .weightInit(WeightInit.XAVIER)
                        .learningRate(0.1)
                        .regularization(true).l2(1e-3)
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(numHiddenLayer).build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation("softmax").nIn(numHiddenLayer).nOut(numClasses).build())
                        .backprop(true)
                        .pretrain(false)
                        .build();

                //Shuffle, Train and Evaluate k-foldIterations, repeat
                for (int k = 0; k < repetitions; k++) {
                    //Shuffle Data and initialize K-Fold-Iterator
                    allData.shuffle();
                    KFoldIterator kFoldIter = new KFoldIterator(kFold, allData);
                    int kCount = 0;

                    //Iterate trough Dataset => Train, then get outputs
                    while (kFoldIter.hasNext()) {
                        //Initialize Model and set Score Listener
                        MultiLayerNetwork model = new MultiLayerNetwork(conf3);
                        model.init();
                        model.setListeners(new ScoreIterationListener(1000));


                        Log.info("-----------------------------");
                        Log.info("KFold: " + kCount);
                        DataSet trainingData = kFoldIter.next();
                        DataSet testData = kFoldIter.testFold();
                        model.fit(trainingData);

                        //Get outputs of the model on the test set
                        INDArray output = model.output(testData.getFeatureMatrix());
                        Integer[] bugs = new Integer[output.rows()];
                        Integer[] values = new Integer[output.rows()];
                        double[] bugsFloat = new double[output.rows()];
                        double[] valuesFloat = new double[output.rows()];
                        INDArray labels = testData.getLabels();
                        for (int i = 0; i < output.rows(); i++) {
                            float highestGuess = 0;
                            float highestValue = 0;
                            for (int j = 0; j < output.columns(); j++) {
                                float guessValue = output.getFloat(i, j);
                                float value = labels.getFloat(i, j);
                                if (guessValue > highestGuess) {
                                    highestGuess = guessValue;
                                    bugs[i] = j;
                                }
                                if (value > highestValue) {
                                    highestValue = value;
                                    values[i] = j;
                                }
                                bugsFloat[i] += guessValue * j;
                                valuesFloat[i] += value * j;
                            }
                        }

                        //Regression
                        regression.regression(bugsFloat, valuesFloat, writerReg, project);

                        //Classification
                        classifier.classify(bugs, values, writerCla, project);

                        //Classification  by Regression (Do Regression on Classified values)
                        regByClass.regressionByClasses(bugs, values, writerRbC, project);

                        classByReg1.classificationByRegression(bugsFloat, valuesFloat, writerCbR, project, 1);

                        kCount++;
                    }
                }

            /* Regression by Classification Eval. */
                regByClass.sumUp(writerRbC, writerSummary, "RbC");
                writerRbC.close();
                regByClass.clear();

            /* Classification Eval. */
                classifier.sumUp(writerCla, writerSummary, "Cla");
                writerCla.close();
                classifier.clear();

            /* Regression Eval. */
                regression.sumUp(writerReg, writerSummary, "Reg");
                writerReg.close();
                regression.clear();

            /* Classification by Regression Eval. */
                classByReg1.sumUp(writerCbR, writerSummary, "CbR");
                writerCbR.close();
                classByReg1.clear();

                writerSummary.writeSummary(project, numClasses);
            }
        }
        writerSummary.close();
    }

    private static void setupDir(String project){
        File theDir = new File("results/"+project.replace(".csv", ""));
        // if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("creating directory: " + theDir.getName());
            boolean result = false;

            try{
                theDir.mkdir();
                result = true;
            }
            catch(SecurityException se){
                //handle it
            }
            if(result) {
                System.out.println("DIR created");
            }
        }
    }
}