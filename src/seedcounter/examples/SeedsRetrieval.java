package seedcounter.examples;

import org.opencv.core.Core;
import org.opencv.core.Mat;
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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

class SeedsRetrieval {
    private static final String REFERENCE_FILE = "reference.png";

    private static final Order ORDER = Order.THIRD;
    private static final ColorSpace FEATURE_SPACE = ColorSpace.RGB;
    private static final ColorSpace TARGET_SPACE = ColorSpace.RGB;
    private static final boolean MASK_BY_CALIBRATED = true;

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        boolean calibrate = args.length == 2 && "--calibrate".equals(args[0]);
        if (!calibrate && args.length != 1) {
            System.out.println("Arguments: [--calibrate] FILE_PATH");
            return;
        }

        String filePath = args[args.length - 1];

        MatchingModel matchingModel = new MatchingModel(
                BRISK.create(), BRISK.create(),
                DescriptorMatcher.BRUTEFORCE_HAMMING, 0.75f
        );
        FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, matchingModel);
        RegressionModel model = RegressionFactory.createModel(ORDER);

        PrintWriter seedLog = new PrintWriter("seed_log.txt");
        Map<String, String> seedData = new HashMap<>();
        seedData.put("header", "1");

        seedData.put("file", filePath);

        Mat image = Imgcodecs.imread(filePath,
                Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

        Mat extractedColorChecker = null;
        ColorChecker checker = null;
        // default scale is calculated by A4 size
        double scale = (297.0 / 0.8 / image.cols()) * (210.0 / 0.85 / image.rows());

        if (calibrate) {
            Quad quad = findColorChecker.findBestFitColorChecker(image);

            if (quad.getArea() / image.cols() / image.rows() < 0.15) {
                extractedColorChecker = quad.getTransformedField(image);
                checker = new ColorChecker(extractedColorChecker);
                if (checker.labDeviationFromReference() > 25) { // bad CC detection
                    checker = null;
                    System.out.println("Couldn't detect colorchecker: colors deviate too much from the reference");
                    extractedColorChecker.release();
                    extractedColorChecker = null;
                } else {
                    scale = checker.pixelArea(quad);
                }
            } else { // bad CC detection
                checker = null;
                System.out.println("Couldn't detect colorchecker: detected area is too large");
            }
        }

        Mat mask;
        Mat colorData;
        Mat forFilter;

        if (checker != null) {
            Mat calibrated;
            try {
                calibrated = checker.calibrate(image, model, FEATURE_SPACE, TARGET_SPACE);
                if (MASK_BY_CALIBRATED) {
                    mask = SeedUtils.getMask(calibrated, scale);
                } else {
                    mask = SeedUtils.getMask(image, scale);
                }

                colorData = SeedUtils.filterByMask(calibrated, mask);
                if (MASK_BY_CALIBRATED) {
                    forFilter = colorData;
                } else {
                    forFilter = SeedUtils.filterByMask(image, mask);
                }
                seedData.put("calibrated", "1");
                image.release();
            } catch (IllegalStateException e) {
                seedData.put("calibrated", "0");
                System.out.println("Couldn't calibrate the image " + filePath + " using the source...");
                mask = SeedUtils.getMask(image, scale);
                colorData = SeedUtils.filterByMask(image, mask);
                forFilter = colorData;
                calibrated = image;
            }

            calibrated.release();
        } else {
            seedData.put("calibrated", "0");
            mask = SeedUtils.getMask(image, scale);
            colorData = SeedUtils.filterByMask(image, mask);
            forFilter = colorData;
            image.release();
        }

        SeedUtils seedUtils = new SeedUtils(0.0, 255.0, false);
        seedUtils.printSeeds(colorData, forFilter, seedLog, seedData, scale);

        mask.release();
        colorData.release();
        if (colorData != forFilter) {
            forFilter.release();
        }

        if (extractedColorChecker != null) {
            extractedColorChecker.release();
        }

        image.release();

        seedLog.close();
    }
}
