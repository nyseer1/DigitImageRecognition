import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class LinearRegressionMultipleInputs {
    static void main(String[] args) throws IOException {

        //read data
        File file = new File("pizza_3_vars.txt");
        List<Integer> X_array = new ArrayList<>();
        List<Integer> Y_array = new ArrayList<>();

        int input_features = 3;

        try (Scanner scanner = new Scanner(file)) {

            //skip the header
            scanner.nextLine();

            // Loop while there is another integer available in the file
            while (scanner.hasNextInt()) {

                //add input feature and label. there are 3 inputs so scan three times
                for(int i = 0; i < input_features; i++){
                    X_array.add(scanner.nextInt());
                }
//                System.out.println(X);
                Y_array.add(scanner.nextInt());
//                System.out.println(Y);
            }


        } catch (FileNotFoundException e) {
            // Handle the case where the file is not found
            System.err.println("Error: The file '" + file.getName() + "' was not found.");
        }

        //creating input feature matrix from the arraylist. arraylist is self expanding, java arrays are not

        //adding extra column for bias column
        int[][] X = new int[X_array.size() / input_features][input_features + 1];

        int count = 0;

        //each row
        for(int j = 0; j < X_array.size() / input_features; j++){

            //each row in bias column = 1
            X[j][0] = 1;
            //each column
            for(int i = 1; i < input_features + 1; i++){
                X[j][i] = X_array.get(count);
                count++;
            }

        }

        //create labels matrix
        //# of labels rows, 1 column
        int[][] Y = new int[Y_array.size()][1];
        //fill rows with the array list elements
        for(int i = 0; i < Y_array.size(); i++){
            Y[i][0] = Y_array.get(i);
        }


        //test if worked
//        for(int i = 0; i < X.length; i++){
//            System.out.println(X[i][0]);
//        }


        //uppercase letter means matrix


        //lr equals how many decimal places of precision
        double[][] W = train(X,Y,100000,0.001);
        double[][] test = predict(X,W);
        //print weights
        System.out.print("weights: ");
        for(int i = 0; i < W.length; i++){
            System.out.print(W[i][0] + " ");
        }
        System.out.println();

        System.out.println("test predictions: ");
        for(int i = 0; i < W.length; i++){
            System.out.print("X" + i + ":" + X[0][i] + " ");
        }
        System.out.println("Y_hat = " + Y[0][0]);

    }

    //returns predictions. every y_hat corresponds to an input x
    public static double[][] predict(int[][] X, double[][] W){

        //output is matrix
        double[][] product = new double[X.length][W[0].length];

        //matrix mult
//      // Multiply the two matrices
//        for (i = 0; i < row1; i++) {
//            for (j = 0; j < col2; j++) {
//                for (k = 0; k < row2; k++)
//                    C[i][j] += A[i][k] * B[k][j];
//            }
//        }
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                for (int k = 0; k < W.length; k++)
                    product[i][j] += X[i][k] * W[k][j];
            }
        }

        return product;
    }

    public static double mse_loss(int[][] X, int[][] Y, double[][] W){

        //call it before for loop to only do the calculation once
        double[][] y_hat = predict(X,W);

        //sum the squared errors
        double sum = 0.0;
        for(int i = 0; i < X.length; i++){
            //first column, ith row. Y and Y hat only have one column and i rows
            sum += Math.pow(y_hat[i][0] - Y[i][0], 2);
        }
        //divide by # of examples to get the mean
        return sum / X.length;

    }

    public static double[][] gradient(int[][] X, int[][] Y, double[][] W, int[][] transposed_X){


        //call it before for loop to only do the calculation once
        double[][] Y_hat = predict(X,W);

        //matrix multiplication on X_transposed and error( difference of Y_hat - Y). X needs to be transposed to be divided
        // has rows are not = Y columns, but X rows = Y rows. so X is transposed to get a gradient
        // gradient dimmensions will be rows = # inputs, 1 column.
        double[][] G = new double[X[0].length][1];

        //error matrix, y_hat - y. only one column
        double[][] E = new double[Y.length][1];
        for (int i = 0; i < Y.length; i++) {
            E[i][0] = Y_hat[i][0] - Y[i][0];
        }

        //matrix mult
        for (int i = 0; i < transposed_X.length; i++) {        // features
            for (int k = 0; k < X.length; k++) {               // samples
                G[i][0] += transposed_X[i][k] * E[k][0];
            }
        }

        //divide each gradient by examples(x length) and then multiply by two
        for(int i = 0; i < X[0].length; i++){
            G[i][0] /= X.length;
            G[i][0] *= 2;
        }

        return G;

    }

    public static double[][] train(int[][] X, int[][] Y, int iterations, double lr){

        //weight is one row per input variable, one column
        //https://nusco.medium.com/of-gradients-and-matrices-1b19de65e5cd
        double[][] W = new double[X[0].length][1];

        //transpose X early so that it can be re-used
        //rows = cols, cols = rows
        int[][] T_X = new int[X[0].length][X.length];
        // Fill the transposed matrix
        // by swapping rows with columns
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {

                // Assign transposed value
                T_X[j][i] = X[i][j];
            }
        }

        //train the data(find w) based off loss
        for(int i = 0; i < iterations; i++){
            double current_loss = mse_loss(X,Y,W);
            System.out.println("Iteration:  " + i + " => Loss: " + current_loss);


            //store gradient to subtract from W.
            double[][] G = gradient(X,Y,W, T_X);
            //matrix sub. W and G are one column, input feature # of rows
            for(int j = 0; j < W.length; j++){
                W[j][0] -= G[j][0] * lr;
            }


        }


        return W;
    }

}
