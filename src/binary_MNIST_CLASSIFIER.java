//might not be necessary later
//to read file

import java.io.*;
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

//basic version of mnist classifier algorithm that only checks if an image is a 5 or not
public class binary_MNIST_CLASSIFIER {


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
        int[][] Y = readLabels(labelsFile);
        float[][] X = readImages(imagesFile);

        //convert training data matrices
        int[][] test_Y = readLabels(testlabelsFile);
        float[][] test_X = readImages(testimagesFile);



        //encode fives for training labels(Y)
        for(int i = 0; i < Y.length; i++){
            if(Y[i][0] == 5){
                Y[i][0] = 1;
            }
            else Y[i][0] = 0;
        }

        //encode fives for test labels(Y)
        for(int i = 0; i < test_Y.length; i++){
            if(test_Y[i][0] == 5){
                test_Y[i][0] = 1;
            }
            else test_Y[i][0] = 0;
        }

        //TEST THE TRAINING DATA
        double[][] W = train(X,Y,100,0.00001);
        test(test_X,test_Y, W);
        for(int i = 0; i < Y.length; i++){
//            System.out.println("num: " + Y[i][0] + " prediction: ");
        }

//        for (int i = 0; i < test_X.length; i++) {
//            System.out.print(" "  + test_X[i][400]);
//
//        }

    }

    //function to test data
    public static void test(float[][] X,int[][] Y, double[][] W){
        int total_examples = X.length;
        int correct_results = 0;

        int[][] y_hat = classify(X,W);

        for(int i = 0; i < total_examples; i++){
            if(y_hat[i][0] == Y[i][0]){
                correct_results++;
//                if(Y[i][0] != 0){
//                    System.out.println("y = " + Y[i][0]);
//                }

            }
        }
        System.out.println();
        double success_percent = (double) (correct_results * 100) / total_examples;
        System.out.println("Success: " + correct_results + "/" + total_examples + " (" + success_percent + "%)");

//        //print weights
//        System.out.print("weights: ");
//        for(int i = 0; i < X[0].length; i++){
//            System.out.print("w" + i + ": " + W[i][0] + " ");
//            if(i % 10 == 0) System.out.println();
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

    //returns predictions rounded to a classifie- ( b in this case 0 or 1)
    public static int[][] classify(float[][] X, double[][] W){

        double[][] sigmoid = forward(X,W);
        int[][] rounded = new int[sigmoid.length][1];
        for(int i = 0; i < sigmoid.length; i++){
            if(sigmoid[i][0] >= 0.5) rounded[i][0] = 1;
            else rounded[i][0] = 0;
        }
        return rounded;
    }

    //returns predictions unrounded. every y_hat corresponds to an input x
    public static double[][] forward(float[][] X, double[][] W){

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
//        System.out.print(" product outputs: ");
//        for (int i = 0; i < X.length; i++) {
//                System.out.print(product[i][0] + " ");
//        }
//        System.out.println();


        //then find sigmoid of each y_hat
        double[][] sigmoid = new double[product.length][1];
        for(int i = 0; i < product.length; i++){
            sigmoid[i][0] = 1 / (1 + Math.exp(-product[i][0]));
        }
//        System.out.print(" sigmoid outputs: ");
//        for (int i = 0; i < X.length; i++) {
//            if(sigmoid[i][0] >= 0.5){
//                System.out.print(sigmoid[i][0] + " ");
//            }
//        }
//        System.out.println();



        return sigmoid;
    }

    //log loss function, (because mse cannot be used with sigmoid, the gradient will reach a local max/minimum and stop abruptly)
    public static double log_loss(float[][] X, int[][] Y, double[][] W){

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

    public static double[][] gradient(float[][] X, int[][] Y, double[][] W, float[][] transposed_X){


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

    public static double[][] train(float[][] X, int[][] Y, int iterations, double lr){

        //weight is one row per input variable, one column
        //https://nusco.medium.com/of-gradients-and-matrices-1b19de65e5cd
        double[][] W = new double[X[0].length][1];

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



//no longer needed
//    public static int byteArrayToInt(byte[] bytes) {
//        if (bytes.length < 4) {
//            throw new IllegalArgumentException("Byte array must have at least 4 elements.");
//        }
//
//        int value = 0;
//        for (int i = 0; i < 4; i++) {
//            // Shift the existing value left by 8 bits (one byte) and add the next byte
//            value = (value << 8) | (bytes[i] & 0xFF);
//        }
//        return value;
//    }

//return Y

    //functions to train the model based off the read data




















    //old methods im replacing
//
//    //returns matrix of y-hat's
//    //return doubles because w and y-hat might be a decimal
//    //y_hat_matrix is product of x_matrix * w_matrix so output is (x_matrix rows length, w_matrix column length) so row,col = (examples,1)
//    //all the x inputs features result in one y_hat (each pixel of an image contributes to a singular output prediction)
//    //so then every example only has one y_hat
//    public static int[][] predict(int[][] x_matrix, double[][] w_matrix){
//        //first matrix is product rows, 2nd is product columns
//        int[][] product = new int[x_matrix.length][w_matrix[0].length];
//
//        //matrix mult that rounds to nearest int.
//        //multiply two corresponding indexes, then add to sum
//        //i is a_matrix rows, j is b_matrix columns
//        for(int i = 0; i < x_matrix.length; i++){
//            for(int j = 0; j < w_matrix[0].length; j++){
//                for (int k = 0; k < w_matrix.length; k++){
//                    product[i][j] += (int) Math.round(x_matrix[i][k] * w_matrix[k][j]);
//                }
//
//
//            }
//
//        }
//        return product;
//
//    }
//
//    //returns the mean squared loss.
//    //mean - all examples averaged (it dosent mattter how many input features there are because they always result to one y_hat(output))
//    //squared - difference is squared to avoid negatives
//    //loss - y_hat - y
//    public static double loss(int[][] x_matrix, int[][] y_matrix, double[][] w_matrix){
//
//        //prediction is a vertical matrix input feature # of rows, and one column
//        int[][] prediction = predict(x_matrix, w_matrix);
//
//        //for every row(x input feature)
//        double sum = 0.0;
//
//        //test first 5 predictions
//        for(int i = 0; i < 5; i++){
//            System.out.println("prediction: " + prediction[i][0] + ", actual: " + y_matrix[i][0]);
//        }
//
//        for(int i = 0; i < y_matrix.length; i++){
//
//            //add to the sum, the squared error: (y_hat - y)^2
//            sum += Math.pow(prediction[i][0] - y_matrix[i][0], 2);
//        }
//
//        //mean of the squared errors
//        return sum / y_matrix.length;
//    }
//
//    //returns the slope of the loss curve(aka gradient) for each weight,
//    // every column is its own weight, and its own x input feature, so also its own gradient.
//    // one weight corresponds to one gradient, because gradient is the derivitive of the loss with respect to weight
//    public static double[][] gradient(int[][] x_matrix, int[][] y_matrix, double[][] w_matrix, int[][] x_transposed){
//
//        //one row per input variable, one column total
//        double[][] gradient_matrix = new double[x_matrix[0].length][1];
//
//        //result is array of one column. this does not do the mult of 2 or divide by # of rows, so it is an inbetween phase
//        //we skip summing all the examples of x per input feature together because it is already done in matrix multiplication
//        int[][] y_hat = predict(x_matrix, w_matrix);
//        double[][] difference = new double[y_hat.length][1];
//
//        for(int i = 0; i < y_hat.length; i++){
//            difference[i][0] = y_hat[i][0] - y_matrix[i][0];
//        }
//        gradient_matrix = matrix_multiply(x_transposed, difference);
//
//        //does the rest of the arithmatic on every element of the matrix
//        // (dividing by each element by example length(rows) then multiplying each element by two)
//        for(int i = 0; i < x_matrix[0].length; i++){
//            gradient_matrix[i][0] /= x_matrix.length; //averages
//            gradient_matrix[i][0] *= 2;
//
//
//        }
//        return gradient_matrix;
//    }
//
//    //lr = learning rate, returns double because w can be decimal
//    public static double[][] train(int[][] x_matrix, int[][] y_matrix, int iterations, double lr){
//
//        //create weight matrix initialized all to zero, one row per input feature(x_matrix columns), and one column.
//        //this is flipped so we can do matrix mult that = the sum of each weight * its corresponding input variable.
//        // x_matrix[0].length gives the length of one of the inner arrays (column length)
//        double[][] w_matrix = new double[x_matrix[0].length][1];
////        System.out.println("this should be 785 for weight matrix rows: " + w_matrix.length);
//
//        //doing this early so that we dont have to keep transposing each time
//        int[][] x_transposed = matrix_transpose(x_matrix);
//
//        //for how many iterations specified, adjust w based off of its corresponding gradient, times the lr
//        for(int i = 0; i < iterations; i++){
//            System.out.println("Iteration    " + i + " => Loss: " + loss(x_matrix,y_matrix,w_matrix));
//
//            //using the same function in the inner loop so im calling it outside the inner loop to save time
//            double[][] gradient_matrix = gradient(x_matrix, y_matrix, w_matrix, x_transposed);
//
//            //matrix subtraction
//            for(int j = 0; j < w_matrix.length; j++){
//                //subtract corresponding gradients from weights
//                w_matrix[j][0] = w_matrix[j][0] - gradient_matrix[j][0] * lr;
//            }
//
//
//        }
//        System.out.println("test done");
//
//
//        return w_matrix;
//    }
//
//    //return double because y-hat might be a decimal, 2nd matrix is decimal because it may be w, which will prob be decimal
//    public static double[][] matrix_multiply(int[][] a_matrix, double[][] b_matrix){
//
//        //first matrix is product rows, 2nd is product columns
//        double[][] product = new double[a_matrix.length][b_matrix[0].length];
//
//        //multiply two corresponding indexes, then add to sum
//        //i is a_matrix rows, j is b_matrix columns
//        for(int i = 0; i < a_matrix.length; i++){
//            for(int j = 0; j < b_matrix[0].length; j++){
//                for (int k = 0; k < b_matrix.length; k++){
//                    product[i][j] += a_matrix[i][k] * b_matrix[k][j];
//                }
//
//
//            }
//
//        }
//
//        return product;
//    }
//
//    //int because x_matrix is an int and i only need it for that
//    public static int[][] matrix_transpose(int[][] matrix){
//
//        //swap size of col and row
//        int[][] transposed = new int[matrix[0].length][matrix.length];
//
//        //copy values of old matrix but transpose the values
//        //for rows
//        for (int i = 0; i < matrix.length; i++) {
//            //for cols
//            for (int j = 0; j < matrix[0].length; j++) {
//                transposed[j][i] = matrix[i][j];
//            }
//        }
//
//        return transposed;
//    }
//
//    //only subtracts rows of matrix a - rows of matrix b. 1st will be a double because y-hat is being subtracted and it prob a double
//    public static double[][] matrix_sub(double[][] a_matrix, int[][] b_matrix){
//
//        //only sub rows, so we can set col to 1
//        double[][] difference = new double[a_matrix.length][1];
//
//        for(int i = 0; i < a_matrix.length; i++){
//            //times a double to convert to double incase it isnt
//            difference[i][0] = 1.0 * a_matrix[i][0] - b_matrix[i][0];
//        }
//
//        return difference;
//
//    }
//
//
//
