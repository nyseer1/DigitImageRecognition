import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class LinearRegression {
    static void main(String[] args) throws IOException {

        //read data
        File file = new File("pizza.txt");
        List<Integer> X = new ArrayList<>();
        List<Integer> Y = new ArrayList<>();

        try (Scanner scanner = new Scanner(file)) {
            //skip the header
            scanner.nextLine();

            // Loop while there is another integer available in the file
            while (scanner.hasNextInt()) {

                //add input feature and label
                X.add(scanner.nextInt());
//                System.out.println(X);
                Y.add(scanner.nextInt());
//                System.out.println(Y);
            }


        } catch (FileNotFoundException e) {
            // Handle the case where the file is not found
            System.err.println("Error: The file '" + file.getName() + "' was not found.");
        }

        //lr equals how many decimal places of precision
        double[] wb = train(X,Y,20000,0.001);
        double w = wb[0];
        double b = wb[1];
        List<Integer> test = new ArrayList<>();
        test.add(20);
        System.out.println("weight = " + w + " bias: " + b);
        System.out.println("predict: " + test.get(0) + " = " + predict(test,w,b)[0]);

    }

    //returns predictions. every y_hat corresponds to an input x
    public static double[] predict(List<Integer> X, double w, double b){

        double[] product = new double[X.size()];
        for(int i = 0; i < X.size(); i++){
            product[i] = X.get(i) * w + b;
        }

        return product;
    }
    public static double loss(List<Integer> X, List<Integer> Y, double w, double b){

        //call it before for loop to only do the calculation once
        double[] y_hat = predict(X,w,b);

        //sum the squared errors
        double sum = 0.0;
        for(int i = 0; i < X.size(); i++){
             sum += Math.pow(y_hat[i] - Y.get(i), 2);
        }
        //divide to get the mean
        return sum / X.size();

    }
    public static double[] train(List<Integer> X, List<Integer> Y, int iterations, double lr){
        double w = 0.0;
        double b = 0.0;

        //initialize the weight bias values to be returned
        double[] wb = new double[2];
        wb[0] = w;
        wb[1] = b;

        for(int i = 0; i < iterations; i++){
            double current_loss = loss(X,Y,w,b);
            System.out.println("Iteration:  " + i + " => Loss: " + current_loss);
            //find gradient once then reuse the values
            double[] gradients = gradient(X,Y,w,b);
            w -= gradients[0] * lr;
            b -= gradients[1] * lr;
            wb[0] = w;
            wb[1] = b;

        }


        return wb;
    }
    public static double[] gradient(List<Integer> X, List<Integer> Y, double w, double b){

        double[] wb_gradients = new double[2];
        //call it before for loop to only do the calculation once
        double[] y_hat = predict(X,w,b);

        //sum the errors multiplied by X. one input corresponds to an output or predicted output
        double sum = 0.0;
        double sumb = 0.0;
        for(int i = 0; i < X.size(); i++){
            sum += (y_hat[i] - Y.get(i)) * X.get(i);
            sumb += y_hat[i] - Y.get(i);
        }
        //divide by X size to get mean, multiply by two
        wb_gradients[0] = (sum / X.size()) * 2;
        wb_gradients[1] = (sumb / X.size()) * 2;

        return wb_gradients;

    }
}
