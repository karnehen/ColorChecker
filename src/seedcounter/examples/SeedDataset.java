package seedcounter.examples;

import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.common.Quad;
import seedcounter.common.SeedUtils;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

class SeedDataset {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/seed_dataset_results";
    private static final String REFERENCE_FILE = "reference.png";

    public static void main(String[] args) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // BRUTEFORCE is used for reproducibility
        MatchingModel matchingModel = new MatchingModel(
                BRISK.create(), BRISK.create(),
                DescriptorMatcher.BRUTEFORCE_HAMMING, 0.75f
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

        RegressionModel model = RegressionFactory.createModel(Order.THIRD);

        PrintWriter seedLog = new PrintWriter(RESULT_DIR + "/seed_log.txt");
        Map<String, String> seedData = new HashMap<>();
        seedData.put("header", "1");

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
            seedData.put("file", fileName);

            Quad quad = findColorChecker.findBestFitColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            findColorChecker.fillColorChecker(image, quad);
            ColorChecker checker = new ColorChecker(extractedColorChecker, true, true);
            Double scale = checker.pixelArea(quad);

            Mat calibrated;
            try {
                calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.RGB);
            } catch (IllegalStateException e) {
                System.out.println("Couldn't calibrate the image " + fileName + " skipping...");
                image.release();
                extractedColorChecker.release();
                continue;
            }

            Mat mask = SeedUtils.getMask(calibrated, scale);
            if (mask == null) {
                System.out.println("error");
                image.release();
                extractedColorChecker.release();
                calibrated.release();
                continue;
            }

            Mat sourceFiltered = SeedUtils.filterByMask(image, mask);
            image.release();
            Mat calibratedFiltered = SeedUtils.filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();
            extractedColorChecker.release();

            seedData.put("type", "source");
            SeedUtils.printSeeds(sourceFiltered, sourceFiltered, seedLog, seedData, scale);
            seedData.put("type", "calibrated");
            SeedUtils.printSeeds(calibratedFiltered, sourceFiltered, seedLog, seedData, scale);
            sourceFiltered.release();
            calibratedFiltered.release();
        }
        seedLog.close();
    }
}
