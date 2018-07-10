package seedcounter.examples;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.colormetric.CellColors;
import seedcounter.colormetric.ColorMetric;
import seedcounter.colormetric.EuclideanLab;
import seedcounter.colormetric.EuclideanRGB;
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

class ColorCheckerMetric {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_FILE = "src/seedcounter/examples/color_metric_results.tsv";
    private static final String REFERENCE_FILE = "reference.png";

    private static void iterateColorspaces(String inputFile, PrintWriter outputFile, RegressionModel model,
                                           Mat extractedChecker, ColorChecker checker,
                                           double rgbBaseline, double labBaseline) {
        ColorMetric rgb = new EuclideanRGB();
        ColorMetric lab = new EuclideanLab();

        for (ColorSpace featuresSpace : ColorSpace.values()) {
            for (ColorSpace targetSpace : ColorSpace.values()) {
                if (model.getClass() == IdentityModel.class &&
                        !(featuresSpace == ColorSpace.RGB && targetSpace == ColorSpace.RGB)) {
                    continue;
                }

                Mat calibrated;
                try {
                    calibrated = checker.calibrate(extractedChecker, model, featuresSpace, targetSpace);
                } catch (IllegalStateException e) {
                    System.out.println("Couldn't calibrate the image " + inputFile + " model: " + model.getName() +
                            " features " + featuresSpace.name() + " targets " + targetSpace.name() + " skipping...");
                    continue;
                }
                CellColors cellColors = checker.getCellColors(calibrated);
                calibrated.release();

                double rgbChange = cellColors.calculateMetric(rgb) / rgbBaseline;
                double labChange = cellColors.calculateMetric(lab) / labBaseline;

                outputFile.println(inputFile + "\t" + model.getName() + "\t" + featuresSpace.name() +
                        "\t" + targetSpace.name() + "\t" + rgbChange + "\t" + labChange);
            }
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MatchingModel matchingModel = new MatchingModel(
            ORB.create(), ORB.create(),
            DescriptorMatcher.BRUTEFORCE_HAMMING, 0.9f
        );
        FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, matchingModel);
        ColorMetric rgb = new EuclideanRGB();
        ColorMetric lab = new EuclideanLab();

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
        outputFile.println("file\tmodel\tfeatures_space\ttarget_space\trgb_change\tlab_change");

        for (String inputFile : inputFiles) {
            System.out.println(inputFile);
            Mat image = Imgcodecs.imread(inputFile,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

            Quad quad = findColorChecker.findColorChecker(image);
            Mat extractedChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedChecker);
            CellColors cellColors = checker.getCellColors(extractedChecker);
            double rgbBaseline = cellColors.calculateMetric(rgb);
            double labBaseline = cellColors.calculateMetric(lab);

            for (Order order : Order.values()) {
                System.out.println(order);
                RegressionModel model = RegressionFactory.createModel(order);
                iterateColorspaces(inputFile, outputFile, model, extractedChecker, checker, rgbBaseline, labBaseline);
            }

            image.release();
            extractedChecker.release();
        }

        outputFile.close();
    }
}
