//might not be necessary later
//to read file

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPInputStream;
//to create extracted file


//MNIST MATRIXES - this is how it needs to be set up

//IMAGES
//each image is an example, each row of the matrix is one image/example
//so at each row value, is the same exact pixel location in a grid of 28x28
//each pixel location is as column because each pixel location is an input_variable x/feature/weight
// (they influence the prediction of which number it is so they are all features)
//each pixel is stored in a byte (a single value equal to 4 bits, can be 0 to 255)
//there needs to be a bias column where each value is 1, for weight x sub 0
//28x28 pixel features plus bias is 785 features, so 785 columns,
// the # of rows depends on the number of examples. We want it to be flexible,
// so assume we dont know # of rows and keep going until we reach the end of the data

//The file format uses big-endian 4-byte integers for the header and unsigned bytes for the image data.
//The file format for image data (idx3):
//      0-3: Magic number (2051 integer, 0x00000803)
//      4-7: Number of images (integer)
//      8-11: Number of rows (integer)
//      12-15: Number of columns (integer)
//      16-end: Image data (unsigned bytes)



//LABELS
//the labels matrix is a one column matrix, each column is a label to an image (aka the y/output, the number it is)
// the # of rows is the same as images.

//The file format for label data (idx1):
//      0-3: Magic number 4 bytes (2049 integer, 0x00000801 hex) (MSB first) big-endian (big end the largest number is last)
//      4-7: Number of images (integer)
//      8-end: Image data (unsigned bytes)


//Byte Order: Endianness refers to the order in which bytes of a multi-byte value (like a 4-byte integer) are stored in computer memory.
//
//In big endian (BE) format, the "big end" or most significant byte (MSB) comes first, which is similar to how humans typically write numbers from left to right (e.g., the number 0x12345678 would be stored as 12 34 56 78).
//In little endian (LE) format (common in modern x86 and ARM processors), the least significant byte (LSB) is stored first (e.g., 0x12345678 would be stored as 78 56 34 12).
//
//Cross-Platform Compatibility: The original MNIST data files were created on a system that used big endian order. Most modern personal computers, however, use little endian architecture.
//
//When you read these files on a little endian machine (like most PCs or Macs), the bytes for multi-byte values (such such as the "magic number", number of images, rows, and columns defined in the file header) will be interpreted incorrectly if you simply read them as native integers. For example, the number 60,000 (number of training images) might be read as 1,625,948,160 if the byte order is not properly handled.
//
//The multi-byte data in the MNIST files are:
//
//Magic Number: A 4-byte integer at the beginning of each file that identifies the file type and data type (e.g., 0x00000803 for image files, 0x00000801 for label files).
//Number of Items: A 4-byte integer indicating the total count of images or labels in the file (e.g., 60,000 for the training set).
//Dimensions (Images only): For the image files, two additional 4-byte integers specify the number of rows and the number of columns (both are 28 for MNIST images).

//multiclass mnist classifier,
//does a binary classification on every class that the image can fall into (0-9)
//skip rounding to 0 or 1 after sigmoid & weighted (WSS) sum (what forward returns)
//result is the number thats (WSS)^ is closest to 1
//uses one hot encoding to encode training data labels

public class multiclass_MNIST_CLASSIFIER {


    static void main(String[] args) throws IOException {

        //unzip training data labels
        String path = "train-labels-idx1-ubyte.gz";
        String labelsFile  = "train-labels.idx1-ubyte";
        decompressGzipFile(path,labelsFile);

        //unzip training data images
        path = "train-images-idx3-ubyte.gz";
        String imagesFile = "train-images.idx3-ubyte";
        decompressGzipFile(path,imagesFile);

        //unzip test data labels
        path = "t10k-labels-idx1-ubyte.gz";
        String testlabelsFile  = "t10k-labels.idx1-ubyte";
        decompressGzipFile(path,testlabelsFile);

        //unzip test data images
        path = "t10k-images-idx3-ubyte.gz";
        String testimagesFile = "t10k-images.idx3-ubyte";
        decompressGzipFile(path,testimagesFile);

        //load the training data labels into training data matrix
        //use a fileinputstream object and call the .read method on it to read each byte one by one
        //and for headers, use .read(byte[] b) - Reads up to b.length bytes of data from this input stream into an array of bytes.
        // resource i found here https://docs.oracle.com/javase/8/docs/api/java/io/FileInputStream.html

        //convert training data matrices
        int[][] Y = readLabelsEncode(labelsFile);
        float[][] X = readImages(imagesFile);

        //convert training data matrices
        int[][] test_Y = readLabels(testlabelsFile);
        float[][] test_X = readImages(testimagesFile);

        //TEST THE TRAINING DATA
        double[][] W = train(X,Y,test_X,test_Y, 200, 0.5);
//        test(test_X,test_Y, W);
//        for(int i = 0; i < Y.length; i++){
////            System.out.println("num: " + Y[i][0] + " prediction: ");
//        }

//        for (int i = 0; i < test_X.length; i++) {
//            System.out.print(" "  + test_X[i][400]);
//
//        }

    }

    //functions to read the data

    public static void decompressGzipFile(String gzipFile, String newFile) {
        byte[] buffer = new byte[1024]; // Use an appropriate buffer size

        try (
                GZIPInputStream gis = new GZIPInputStream(new FileInputStream(gzipFile));
                FileOutputStream fos = new FileOutputStream(newFile)
        ) {
            int len;
            // Read uncompressed data into buffer and write to output file
            while ((len = gis.read(buffer)) > 0) {
                fos.write(buffer, 0, len);
            }
            System.out.println("File successfully decompressed to " + newFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static int[][] readLabelsEncode(String file) throws IOException {

        // Use try-with-resources to ensure the FileInputStream is closed automatically
        try (DataInputStream dis =
                     new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {

            // Read the magic number (4 bytes)

            int magicNumber = dis.readInt();
            // Verify magic number 0x00000801 or 2049 in decimal
            if (magicNumber != 2049) {
                throw new IOException("Invalid magic number: " + magicNumber);
            }
            // Read the number of examples (4 bytes)

            int numberOfExamples = dis.readInt();
            System.out.println("Number of labels in file: " + numberOfExamples);

            int[][] Y = new int[numberOfExamples][10];

            // Read the remaining bytes, the actual labels, into an array
            //create Y, one column, rows = # of examples. int[rows][columns]
            //read the values in from the array
            int digit;
            for (int i = 0; i < numberOfExamples; i++) {
                digit = dis.readUnsignedByte(); // perfect for labels

                /*
                one hot encoding (hot one, cold zero)
                we need 10 digits so we can make a matrix with 10 columns, one per digit.
                1 if the digit is at that position, 0 if not. So one 1 per row, the rest 0.
                because only one digit exists at each index for the original labels array
                 */
                switch(digit) {
                    case 0:
                        Y[i][0] = 1; break;
                    case 1:
                        Y[i][1] = 1; break;
                    case 2:
                        Y[i][2] = 1; break;
                    case 3:
                        Y[i][3] = 1; break;
                    case 4:
                        Y[i][4] = 1; break;
                    case 5:
                        Y[i][5] = 1; break;
                    case 6:
                        Y[i][6] = 1; break;
                    case 7:
                        Y[i][7] = 1; break;
                    case 8:
                        Y[i][8] = 1; break;
                    case 9:
                        Y[i][9] = 1; break;



                }
            }
            // test labels data
//            System.out.println("First 5 labels:");
//            for (int i = 0; i < 5 && i < numberOfExamples; i++) {
//                //probably not necessary but leaving incase an error occurs
//                // The labels are unsigned bytes, Java treats them as signed.
//                // Casting to int and using bitwise AND with 0xFF gets the unsigned value.
//                //int label = labels[i] & 0xFF;
//                System.out.println("Label at index " + i + ": " + Y[i][0]);
//            }
            return Y;
        }
    }

    public static int[][] readLabels(String file) throws IOException {

        // Use try-with-resources to ensure the FileInputStream is closed automatically
        try (DataInputStream dis =
                     new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {

            // Read the magic number (4 bytes)

            int magicNumber = dis.readInt();
            // Verify magic number 0x00000801 or 2049 in decimal
            if (magicNumber != 2049) {
                throw new IOException("Invalid magic number: " + magicNumber);
            }
            // Read the number of examples (4 bytes)

            int numberOfExamples = dis.readInt();
            System.out.println("Number of labels in file: " + numberOfExamples);

            int[][] Y = new int[numberOfExamples][1];

            // Read the remaining bytes, the actual labels, into an array
            //create Y, one column, rows = # of examples. int[rows][columns]
            //read the values in from the array
            for (int i = 0; i < numberOfExamples; i++) {
                Y[i][0] = dis.readUnsignedByte(); // perfect for labels



                }
            return Y;
            }
            // test labels data
//            System.out.println("First 5 labels:");
//            for (int i = 0; i < 5 && i < numberOfExamples; i++) {
//                //probably not necessary but leaving incase an error occurs
//                // The labels are unsigned bytes, Java treats them as signed.
//                // Casting to int and using bitwise AND with 0xFF gets the unsigned value.
//                //int label = labels[i] & 0xFF;
//                System.out.println("Label at index " + i + ": " + Y[i][0]);
//            }
        }


    //return X matrix (images matrix. each pixel of an image is a column, each row is an image)
    public static float[][] readImages(String file) throws IOException {

        try (DataInputStream dis =
                     new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {

            int magicNumber = dis.readInt();
            if (magicNumber != 2051) {
                throw new IOException("Invalid magic number: " + magicNumber);
            }

            int numberOfExamples = dis.readInt();
            System.out.println("Number of images in file: " + numberOfExamples);
            int numberOfRows = dis.readInt();
            int numberOfColumns = dis.readInt();

            int pixelsPerImage = numberOfRows * numberOfColumns;
            int matrixColumns = pixelsPerImage + 1; // +1 for bias

            float[][] X = new float[numberOfExamples][matrixColumns];

            byte[] image = new byte[pixelsPerImage];

            for (int i = 0; i < numberOfExamples; i++) {

                // bias term
                X[i][0] = 1.0f;

                // read exactly one image
                dis.readFully(image);

                for (int j = 0; j < pixelsPerImage; j++) {
                    int pixel = image[j] & 0xFF;       // unsigned
                    X[i][j + 1] = pixel / 255.0f;      // normalize
                }
            }

            return X;
        }
    }

    ////functions to train (find w's) based off of data

    //(-the book has a separate function for sigmoid but i put it inside forward)
    //returns matrix of confidence levels. One for each example of every column
    public static double[][] forward(float[][] X, double[][] W){

        //first find the weighted sum (y_hat)

        //output is matrix, one column per classifier, one example per row
        double[][] product = new double[X.length][W[0].length];

        //matrix mult reference
//        for (int i = 0; i < row1; i++) {
//            for (int j = 0; j < col2; j++) {
//                for (int k = 0; k < row2; k++)
//                    C[i][j] += A[i][k] * B[k][j];
//            }
//        }
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                for (int k = 0; k < W.length; k++)
                    product[i][j] += X[i][k] * W[k][j];
            }
        }
//        System.out.print("test product outputs: ");
//        for (int i = 0; i < X.length; i++) {
//                System.out.print(product[i][0] + " ");
//        }
//        System.out.println();


        //then find sigmoid for each y_hat for each classifier(classifiers are one per column)
        double[][] sigmoid = new double[product.length][10];
        for(int i = 0; i < product.length; i++){
            for (int j = 0; j < product[0].length; j++) {
                sigmoid[i][j] = 1 / (1 + Math.exp(-product[i][j]));

            }
        }
//        System.out.print("test sigmoid outputs: ");
//        for (int i = 0; i < X.length; i++) {
//            if(sigmoid[i][0] >= 0.5){
//                System.out.print(sigmoid[i][0] + " ");
//            }
//        }
//        System.out.println();



        return sigmoid;
    }

    //returns predictions rounded to a classifier (example: digits 0 - 9)
    public static int[][] classify(float[][] X, double[][] W){

        //matrix of examples, each with one column for each digit. the value at each index is the confidence level at that digit and example
        double[][] y_hat = forward(X,W);

        int[][] classification = new int[y_hat.length][1];

        //for each example make a prediction. (classify each example)
        for(int i = 0; i < y_hat.length; i++){

            double max = 0;

            //go through every column in an example,
            //find max of columns(the max is the highest confidence digit, determines which digit is predicted)
            for(int j = 0; j < y_hat[0].length; j++){
                //compare each index to find max
                if(y_hat[i][j] > max){
                    //if column with the largest number is the current column
                    //then set max to current column. current column is j, so max is number at i,j, and the digit it corresponds to is j
                    max = y_hat[i][j];
                    classification[i][0] = j;
                }
            }

        }
        return classification;
    }



    //log loss function, (because mse cannot be used with sigmoid, the gradient will reach a local max/minimum and stop abruptly)
    public static double log_loss(float[][] X, int[][] Y, double[][] W){

        //call it before for loop to only do the calculation once
        double[][] y_hat = forward(X,W);

        //// probably also needs to be the same dimmensions as hot encoded Y now, same as W
        //first term
        double[][] first = new double[Y.length][Y[0].length];
        //second term
        double[][] second = new double[Y.length][Y[0].length];

//        Y is now hot encoded (matrix with multiple columns) so now we need to matrix mult
        //matrix mult reference
//        for (i = 0; i < row1; i++) {
//            for (j = 0; j < col2; j++) {
//                for (k = 0; k < row2; k++)
//                    C[i][j] += A[i][k] * B[k][j];
//            }
//        }
        double sum = 0.0;

        //matrix mult reference
//        for (int i = 0; i < row1; i++) {
//            for (int j = 0; j < col2; j++) {
//                for (int k = 0; k < row2; k++)
//                    C[i][j] += A[i][k] * B[k][j];
//            }
//        }

        //sum the matrices
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < y_hat[0].length; j++) {
                first[i][j] += Y[i][j] * Math.log(y_hat[i][j]);
                second[i][j] += (1 - Y[i][j]) * Math.log(1 - y_hat[i][j]);


            }
        }

        //sum each matrix index together
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < y_hat[0].length; j++) {
                sum += first[i][j] + second[i][j];
            }

        }

        //takes the mean (the sum(first+second) / the # of examples), and then multiply the mean by -1
        return -(sum / X.length);
    }

    public static double[][] gradient(float[][] X, int[][] Y, double[][] W, float[][] transposed_X){


        //call it before for-loop to only do the calculation once
        double[][] Y_hat = forward(X,W);

        //matrix multiplication on X_transposed and error( difference of Y_hat - Y). X needs to be transposed to be divided
        // X rows are not = Y columns, but X rows = Y rows. so X is transposed to get a gradient
        // X (m * n) * W (n * k) = Y (m * k)
        //m = # of examples
        //k = # of classifiers
        //n = # of input variables
        // gradient dimmensions will be rows = # inputs, 1 column per classifier
        //row of 1st matrix, column of 2nd
        double[][] G = new double[transposed_X.length][W[0].length];

        //error matrix, y_hat - y. matrix sub
        double[][] E = new double[Y.length][Y[0].length];
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[0].length; j++) {
                E[i][j] = Y_hat[i][j] - Y[i][j];
            }
        }

//        for (i = 0; i < row1; i++) {
//            for (j = 0; j < col2; j++) {
//                for (k = 0; k < row2; k++)
//                    C[i][j] += A[i][k] * B[k][j];
//            }
//        }
        //matrix mult of X.T * E
        for (int i = 0; i < transposed_X.length; i++) {
            for (int j = 0; j < E[0].length; j++) {
                for (int k = 0; k < E.length; k++)
                    G[i][j] += transposed_X[i][k] * E[k][j];
            }
        }


        //divide each gradient by examples(x length) and then multiply by two
        for(int i = 0; i < transposed_X.length; i++){
            for (int j = 0; j < W[0].length; j++) {
                G[i][j] /= X.length;
            }
//            G[i][0] *= 2; no longer needed in partial derivitive of log_loss with respect to weight
        }

        return G;

    }

    public static double[][] train(float[][] X, int[][] Y,float[][] test_X, int[][] test_Y, int iterations, double lr) throws IOException {

        //weight is one row per input variable, one column per class
        double[][] W = new double[X[0].length][Y[0].length];

//        //test initializing weights to random stuff
//        for(int i = 0; i < W.length; i++){
//            W[i][0] = 0;
//        }

        //transpose X early so that it can be re-used
        //rows = cols, cols = rows
        float[][] T_X = new float[X[0].length][X.length];
        // Fill the transposed matrix
        // by swapping rows with columns
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                // Assign transposed value
                T_X[j][i] = X[i][j];
            }
        }

        //train the data(find w) based off gradient(loss with respect to weight)
        for(int i = 0; i < iterations; i++){

            report(i, X,Y,test_X, test_Y, W);

            //store gradient to subtract from W.
            double[][] G = gradient(X,Y,W, T_X);
            //matrix sub.
            for(int j = 0; j < W.length; j++){
                for (int k = 0; k < W[0].length; k++) {
                    W[j][k] -= G[j][k] * lr;
                }
            }


        }

        report(iterations, X,Y,test_X, test_Y, W);


        return W;
    }
    //function to test how well the classifier ml alg is learning
    public static void report(int iteration, float[][] train_X,int[][] train_Y, float[][] test_X,int[][] test_Y, double[][] W) throws IOException {
        int total_examples = test_X.length;
        int correct_predictions = 0;

        //count correct predictions
        int[][] y_hat = classify(test_X,W);
        for(int i = 0; i < total_examples; i++){
            if(y_hat[i][0] == test_Y[i][0]){
                correct_predictions++;
//                if(Y[i][0] != 0){
//                    System.out.println("y = " + Y[i][0]);
//                }

            }
        }
        double training_loss = log_loss(train_X, train_Y, W);

        //Write weights to JSON file
        PrintWriter writer = new PrintWriter("weights.txt", StandardCharsets.UTF_8);

        for (int i = 0; i < W.length; i++) {    //for each row
            for (int j = 0; j < W[0].length; j++) { //for each column

                writer.print(W[i][j] + " ");// add new weight and space at specified row and column
            }
            writer.println(); //add new row

        }
        writer.close();

        System.out.println();
        double success_percent = (double) (correct_predictions * 100) / total_examples;
        System.out.println("i: " + iteration + " - Loss: " + training_loss + ", (" + success_percent + "%)");

//        //print weights
//        System.out.print("weights: ");
//        for(int i = 0; i < X[0].length; i++){
//            System.out.print("w" + i + ": " + W[i][0] + " ");
//            if(i % 10 == 0) System.out.println();
//        }
    }

}

