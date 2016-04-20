package com.neural.network;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by efetsko on 4/4/2016.
 */
public class Network {

    int inputLayers = 2;
    //must be equal to the amount of examples
    int hiddenLayers = 3;
    //
    int outputLayers = 1;

    //Weights
    DoubleMatrix W1 = DoubleMatrix.randn(inputLayers, hiddenLayers);
    DoubleMatrix W2 = DoubleMatrix.randn(hiddenLayers, outputLayers);

    //Node and weighted output values
    DoubleMatrix z2;
    DoubleMatrix z3;
    DoubleMatrix a2;

    //dJdW
    DoubleMatrix dJdW1;
    DoubleMatrix dJdW2;

    //training variables
    double learningRate = 0.5;
    int trainAmount = 100000;

    public DoubleMatrix forward(DoubleMatrix X) {

        this.z2 = X.mmul(this.W1);
        this.a2 = sigmoid(z2);
        this.z3 = a2.mmul(this.W2);
        DoubleMatrix yHat = sigmoid(this.z3);


        return yHat;


    }

    public DoubleMatrix sigmoid(DoubleMatrix z) {

        DoubleMatrix one = DoubleMatrix.ones(z.rows, z.columns);

        return one.div(one.add(MatrixFunctions.exp(z.neg())));

    }

    public DoubleMatrix sigmoidPrime(DoubleMatrix z) {

        DoubleMatrix one = DoubleMatrix.ones(z.rows, z.columns);

        return sigmoid(z).mul(one.sub(sigmoid(z)));

    }

    public Double costFunction(DoubleMatrix data, DoubleMatrix expected) {
        DoubleMatrix yHat = forward(data);
        Double J = 0.5 * (MatrixFunctions.pow(expected.sub(yHat), 2).sum());

        return J;

    }

    public void costFunctionPrime(DoubleMatrix X, DoubleMatrix y) {
        DoubleMatrix yHat = forward(X);

        DoubleMatrix delta3 = y.sub(yHat).neg().mul(sigmoidPrime(this.z3));
        dJdW2 = this.a2.transpose().mmul(delta3);

        DoubleMatrix delta2 = delta3.mmul(W2.transpose()).mul(sigmoidPrime(this.z2));
        dJdW1 = X.transpose().mmul(delta2);

    }

    public void backProp() {
        W2 = W2.sub(dJdW2.mul(learningRate));
        W1 = W1.sub(dJdW1.mul(learningRate));
    }

    public void train(DoubleMatrix X, DoubleMatrix y) {
        for (int i = 0; i < trainAmount; i++) {
            costFunctionPrime(X, y);
            backProp();
        }

    }










}
