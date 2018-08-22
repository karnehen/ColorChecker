package seedcounter.examples;

import javafx.util.Pair;
import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.common.Helper;
import seedcounter.common.Quad;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

class CalculateArea {
    private static final String INPUT_FILES = "src/seedcounter/examples/calculate_area_input_files.txt";
    private static final String RESULT_FILE = "src/seedcounter/examples/calculate_area_results.txt";
    private static final String REFERENCE_FILE = "reference.png";
    // targets and ranges
    private static final List<Pair<Scalar, Scalar>> POTATO_TYPES = Arrays.asList(
            new Pair<>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
            new Pair<>(new Scalar(17, 67, 232), new Scalar(50, 50, 50)),
            new Pair<>(new Scalar(45, 170, 220), new Scalar(30, 30, 30))
        );

    private static void printSeeds(PrintWriter outputFile, Mat image, Double scale) {
        List<MatOfPoint> contours = Helper.getContours(image);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        for (int i = 0; i < contours.size(); ++i) {
            MatOfPoint contour = contours.get(i);
            Double area = scale * Imgproc.contourArea(contour);
            outputFile.println("Object: " + i + "; Area: " + area);
            contour.release();
        }

        seedBuffer.release();
    }

    private static Mat getMask(Mat image) {
        Mat mask = Helper.binarizeSeed(image, POTATO_TYPES);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                new Size(150, 150));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        kernel.release();

        return mask;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MatchingModel MATCHING_MODEL = new MatchingModel(
            ORB.create(), ORB.create(),
            DescriptorMatcher.BRUTEFORCE_HAMMING, 0.9f
        );
        FindColorChecker f = new FindColorChecker(REFERENCE_FILE, MATCHING_MODEL);

        RegressionModel model = RegressionFactory.createModel(Order.FIRST);

        List<String> inputFiles = null;
        try {
            inputFiles = FileUtils.readLines(new File(INPUT_FILES), "utf-8");
        } catch (IOException e) {
            System.out.println("Can't read from file " + INPUT_FILES);
            System.exit(1);
        }

        PrintWriter outputFile = null;
        try {
            outputFile = new PrintWriter(RESULT_FILE);
        } catch (FileNotFoundException e) {
            System.out.println("Can't write to file " + RESULT_FILE);
            System.exit(1);
        }

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            outputFile.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

            Quad quad = f.findColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedColorChecker);

            Mat calibrated;
            try {
                calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.RGB);
            } catch (IllegalStateException e) {
                System.out.println("Couldn't calibrate the image " + fileName + " skipping...");
                continue;
            } finally {
                image.release();
            }

            Mat mask = getMask(calibrated);
            Mat filtered = Helper.filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();

            Double scale = checker.pixelArea(quad);
            printSeeds(outputFile, filtered, scale);
            filtered.release();
            extractedColorChecker.release();
        }

        outputFile.close();
    }
}
