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
import seedcounter.common.Quad;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.IOException;
import java.util.List;

class Calibration {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/calibration_results";
    private static final String REFERENCE_FILE = "reference.png";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MatchingModel matchingModel = new MatchingModel(
                ORB.create(), ORB.create(),
                DescriptorMatcher.BRUTEFORCE_HAMMING, 0.9f
        );
        FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, matchingModel);

        List<String> inputFiles = null;
        try {
            inputFiles = FileUtils.readLines(new File(INPUT_FILES), "utf-8");
        } catch (IOException e) {
            System.out.println("Can't read from file " + INPUT_FILES);
            System.exit(1);
        }

        File resultDirectory = new File(RESULT_DIR);
        resultDirectory.mkdir();

        RegressionModel model = RegressionFactory.createModel(Order.FIRST);

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

            Quad quad = findColorChecker.findColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedColorChecker);

            Mat calibrated;
            try {
                calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.XYZ);
            } catch (IllegalStateException e) {
                System.out.println("Couldn't calibrate the image " + fileName + " skipping...");
                continue;
            } finally {
                image.release();
                extractedColorChecker.release();
            }

            File inputFile = new File(fileName);
            Imgcodecs.imwrite(resultDirectory.getAbsolutePath() + "/" + inputFile.getName(), calibrated);
            calibrated.release();
        }
    }
}
