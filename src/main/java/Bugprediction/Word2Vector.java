package Bugprediction;

import Bugprediction.Iterators.RowIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.*;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import java.util.Collection;
import java.util.List;

/**
 * Part of the Bachelor's thesis of Sébastien Broggi
 * Based on work by agibsonccc.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2Vector {
    private Word2Vec vec;

    private static Logger log = LoggerFactory.getLogger(Word2Vector.class);

    public Word2Vector(List<String> rowList) throws Exception {
        //Building Word2Vec
        log.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new RowIterator(rowList);
        this.vec = buildVector(iter);

        UiServer server = UiServer.getInstance();
        System.out.println("Started on port " + server.getPort());
    }

    public void test(String word, int nearest){
        Collection<String> lst = this.vec.wordsNearest(word, nearest);
        System.out.println(nearest +" Words closest to '" + word + "': " + lst);
    }

    public Word2Vec getVec(){
        return this.vec;
    }

    private static Word2Vec buildVector(SentenceIterator iter) throws IOException {
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*  CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.         */
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");
        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        return vec;
    }




}