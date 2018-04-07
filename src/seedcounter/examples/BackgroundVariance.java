package seedcounter.examples;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.xfeatures2d.SIFT;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.common.Clusterizer;
import seedcounter.common.Quad;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.IdentityModel;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

class BackgroundVariance {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_FILE = "src/seedcounter/examples/background_variance_results.tsv";
    private static final String REFERENCE_FILE = "reference.png";
    private static final Clusterizer clusterizer = new Clusterizer(2);

    private static void iterateColorspaces(String inputFile, PrintWriter outputFile, RegressionModel model, Mat image,
                                           ColorChecker checker, Mat beforeSamples, Mat[] beforeClusters) {
        for (ColorSpace featuresSpace : ColorSpace.values()) {
            for (ColorSpace targetSpace : ColorSpace.values()) {
                if (model.getClass() == IdentityModel.class &&
                        !(featuresSpace == ColorSpace.RGB && targetSpace == ColorSpace.RGB)) {
                    continue;
                }

                Mat calibrated = checker.calibrate(image, model, featuresSpace, targetSpace);
                Mat afterSamples = clusterizer.getClusteringSamples(calibrated);
                calibrated.release();

                double varianceChange = clusterizer.getBackgroundVariance(afterSamples, beforeClusters[0]) /
                        (clusterizer.getBackgroundVariance(beforeSamples, beforeClusters[0]) + 1e-5);
                afterSamples.release();

                outputFile.println(inputFile + "\t" + model.getName() + "\t" + featuresSpace.name() +
                        "\t" + targetSpace.name() + "\t" + varianceChange);
            }
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MatchingModel MATCHING_MODEL = new MatchingModel(
            SIFT.create(), SIFT.create(),
            DescriptorMatcher.FLANNBASED, 0.7f
        );
        FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, MATCHING_MODEL);

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
        outputFile.println("file\tmodel\tfeatures_space\ttarget_space\tvariance_change");

        for (String inputFile : inputFiles) {
            System.out.println(inputFile);
            Mat image = Imgcodecs.imread(inputFile,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

            Quad quad = findColorChecker.findColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedColorChecker);

            for (Order order : Order.values()) {
                System.out.println(order);
                RegressionModel model = RegressionFactory.createModel(order);

                Mat beforeSamples = clusterizer.getClusteringSamples(image);
                Mat[] beforeClusters = clusterizer.clusterize(beforeSamples);

                iterateColorspaces(inputFile, outputFile, model, image, checker, beforeSamples, beforeClusters);

                beforeSamples.release();
                beforeClusters[0].release();
                beforeClusters[1].release();
            }

            image.release();
            extractedColorChecker.release();
        }

        outputFile.close();
    }
}
