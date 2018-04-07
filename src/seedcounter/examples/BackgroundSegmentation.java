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
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.IOException;
import java.util.List;

class BackgroundSegmentation {
    private static final String INPUT_FILES = "src/seedcounter/examples/background_segmentation_input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/background_segmentation_results";
    private static final String REFERENCE_FILE = "reference.png";
    private static final Clusterizer clusterizer = new Clusterizer(3);

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

        File resultDirectory = new File(RESULT_DIR);
        resultDirectory.mkdir();

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

            Quad quad = findColorChecker.findColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedColorChecker);

            RegressionModel model = RegressionFactory.createModel(Order.THIRD);

            Mat calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.RGB);
            image.release();
            extractedColorChecker.release();

            Mat samples = clusterizer.getClusteringSamples(calibrated);
            Mat[] clusters = clusterizer.clusterize(samples);
            Mat segmentedBackground = clusterizer.getBackgroundSegmentation(calibrated, clusters);
            calibrated.release();
            samples.release();

            File inputFile = new File(fileName);
            Imgcodecs.imwrite(resultDirectory.getAbsolutePath() + "/" + inputFile.getName(),
                    segmentedBackground);
            segmentedBackground.release();
        }
    }
}
