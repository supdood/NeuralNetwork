package com.neural.network;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
	// write your code here

        double[][] t = new double[2][4];
        t[0][0] = 1;
        t[0][1] = 2;
        t[0][2] = 3;
        t[0][3] = 4;
        t[1][0] = 5;
        t[1][1] = 6;
        t[1][2] = 7;
        t[1][3] = 8;

        int[] i = {1, 2, 3};
        Network n = new Network();
        DoubleMatrix tData = new DoubleMatrix(3, 1, -1, 0, 1);
        DoubleMatrix test = new DoubleMatrix(3, 2, 3, 5, 10, 5, 1, 2);
        DoubleMatrix y = new DoubleMatrix(3, 1, 75, 82, 93);
        n.sigmoidPrime(tData);

        DoubleMatrix x = new DoubleMatrix(1, 3, 1, 2, 4);
        System.out.println(MatrixFunctions.pow(x, 2));

        System.out.println(test);
        System.out.println(n.costFunction(test, x));

        n.costFunctionPrime(test, y);
        System.out.println(n.dJdW1);
        System.out.println(n.dJdW2);

    }
}
