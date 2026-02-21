import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class BinaryClassifier {
    static void main(String[] args) throws IOException {

        //read data
        File file = new File("police.txt");
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

        //TEST THE DATA
        double[][] W = train(X,Y,10000,0.001);
        test(X,Y,W);


//        for(int i = 0; i < input_features + 1; i++){
//            System.out.print("w" + i + ": " + W[i][0] + " ");
//        }

//        System.out.println("weights: ");
//        System.out.println("predict: " + test[0][0] + " = " + predict(test,W)[0]);

    }

    //function to test data
    public static void test(int[][] X,int[][] Y, double[][] W){
        int total_examples = X.length;
        int correct_results = 0;

        int[][] y_hat = classify(X,W);

        for(int i = 0; i < total_examples; i++){
            if(y_hat[i][0] == Y[i][0]){
                correct_results++;
            }
        }
        double success_percent = (double) (correct_results * 100) / total_examples;
        System.out.println("Success: " + correct_results + "/" + total_examples + " (" + success_percent + "%)");
        System.out.print("weights: ");
        for(int i = 0; i < X[0].length; i++){
            System.out.print("w" + i + ": " + W[i][0] + " ");
        }
    }

    //returns predictions rounded to a classifier (in this case 0 or 1)
    public static int[][] classify(int[][] X, double[][] W){

        double[][] sigmoid = forward(X,W);
        int[][] rounded = new int[sigmoid.length][1];
        for(int i = 0; i < sigmoid.length; i++){
            rounded[i][0] = (int) Math.round(sigmoid[i][0]);
        }
        return rounded;
    }

    //returns predictions unrounded. every y_hat corresponds to an input x
    public static double[][] forward(int[][] X, double[][] W){

        //first find the weighted sum (y_hat)

        //output is matrix, one column, example # rows, one output per example of x values(inputs)
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


        //then find sigmoid of each y_hat
        double[][] sigmoid = new double[product.length][1];
        for(int i = 0; i < product.length; i++){
            sigmoid[i][0] = 1 / (1 + Math.exp(-product[i][0]));
        }



        return sigmoid;
    }

    //log loss function, (because mse cannot be used with sigmoid, the gradient will reach a local max/minimum and stop abruptly)
    public static double log_loss(int[][] X, int[][] Y, double[][] W){

        //call it before for loop to only do the calculation once
        double[][] y_hat = forward(X,W);

        //first term
        double[][] first = new double[Y.length][1];
        //second term
        double[][] second = new double[Y.length][1];
        for(int i = 0; i < Y.length; i++){
            first[i][0] = Y[i][0] * Math.log(y_hat[i][0]);
            second[i][0] = (1 - Y[i][0]) * Math.log(1 - y_hat[i][0]);
        }

        //average the two together, and then multiply the averages by -1

        //sum the terms for each example
        double sum = 0.0;
        for(int i = 0; i < X.length; i++){
            //first column, ith row. Y and Y hat only have one column and i rows
            sum += first[i][0] + second[i][0];
        }

        //divide by # of examples to get the mean, then multiply average by -1
        return -(sum / X.length);
    }

    public static double[][] gradient(int[][] X, int[][] Y, double[][] W, int[][] transposed_X){


        //call it before for loop to only do the calculation once
        double[][] Y_hat = forward(X,W);

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
//            G[i][0] *= 2; no longer needed in partial derivitive of log_loss with respect to weight
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
            double current_loss = log_loss(X,Y,W);
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
