package com.neural.network;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
	// write your code here

        int[] i = {1, 2, 3};
        Network n = new Network();
        DoubleMatrix X = new DoubleMatrix(3, 2, 3, 5, 10, 5, 1, 2);
        //X = new DoubleMatrix(4, 2, 3, 5, 10, 5, 1, 2, 2, 3);
        DoubleMatrix normalize = X.columnMaxs();
        X = X.divRowVector(normalize);
        DoubleMatrix y = new DoubleMatrix(3, 1, 75, 82, 93);
        //y = new DoubleMatrix(4, 1, 75, 82, 93, 69);
        y = y.div(100);

        DoubleMatrix t = X.mmul(n.W1);

        //making test data
        //DoubleMatrix mine = new DoubleMatrix(1, 2, 1, 11110);
        //mine = mine.divRowVector(normalize);
        //

        //System.out.println(t.rows);
        //System.out.println(t.columns);

        n.train(X, y);
        //System.out.println(n.forward(mine));
        System.out.println(n.forward(X));
/**
        int scalar = 3;
        n.costFunctionPrime(X, y);
        //System.out.println(n.dJdW1);
        //System.out.println(n.dJdW2);

        System.out.println();
        System.out.println(n.costFunction(X, y));
        n.W1 = n.W1.sub(n.dJdW1.mul(scalar));
        n.W2 = n.W2.sub(n.dJdW2.mul(scalar));

        System.out.println(n.costFunction(X, y));

**/
    }
}
