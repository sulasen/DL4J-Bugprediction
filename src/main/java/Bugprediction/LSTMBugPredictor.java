package Bugprediction;

import Bugprediction.Iterators.WordIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.List;
import java.util.Random;

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	from Project Gutenberg. Training on other text sources should be relatively easy to implement.

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class LSTMBugPredictor {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 100;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 100;				//Length of each sample to generate
		String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in Bugprediction.Iterators.CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);

		//Make Connection to DB and get Data
		SQLConnector sqlConnector = new SQLConnector(10000);
		List<String> rowList = sqlConnector.getFixedList();

		//Vectorize the Words
		Word2Vector word2Vector = new Word2Vector(rowList);
		word2Vector.test("boolean", 3);
		Word2Vec vec = word2Vector.getVec();

		WordIterator iter = new WordIterator(vec, rowList, miniBatchSize, 100);

		//Bugprediction.Iterators.CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("tanh")        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		//Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

			iter.reset();	//Reset iterator for another epoch
		}

		System.out.println("\n\nExample complete");
	}

	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter Bugprediction.Iterators.CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        WordIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = iter.getRandomWord();
		}
		double[] init = iter.getVector(initialization);

		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), init.length);
		for( int i=0; i<init.length; i++ ){
			for( int j=0; j<numSamples; j++ ){
				int vectorIndex = 0;
				for (double value:init){
					initializationInput.putScalar(new int[]{j,vectorIndex,i}, value);
					vectorIndex++;
				}
			}
		}

		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

		//Sample from network (and feed samples back into input) one word at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int row = 0;
				for (double outputValue: outputProbDistribution) {
					nextInput.putScalar(new int[]{s, row}, outputValue);
					row++;
				}
				//Prepare next time step input
				sb[s].append(iter.getWord(outputProbDistribution));	//Add sampled character to StringBuilder (human readable output)
			}

			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}