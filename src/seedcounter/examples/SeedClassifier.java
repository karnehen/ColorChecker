package seedcounter.examples;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.common.Helper;
import seedcounter.common.Quad;
import seedcounter.common.SeedUtils;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.IOException;
import java.util.*;

class SeedClassifier {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/seed_dataset_results";
    private static final String REFERENCE_FILE = "reference.png";

    private final SVM purpleSvm;
    private final SVM redSvm;
    private final SVM whiteSvm;

    public SeedClassifier() {
        purpleSvm = SVM.load("purple.xml");
        redSvm = SVM.load("red.xml");
        whiteSvm = SVM.load("white.xml");
    }

    private SeedColor classifySingleSeed(List<Map<String, String>> seedData) {
        double[] redArray = new double[seedData.size()];
        double[] greenArray = new double[seedData.size()];
        double[] blueArray = new double[seedData.size()];
        Median median = new Median();

        for (int i = 0; i < seedData.size(); ++i) {
            redArray[i] = Double.parseDouble(seedData.get(i).get("red"));
            greenArray[i] = Double.parseDouble(seedData.get(i).get("green"));
            blueArray[i] = Double.parseDouble(seedData.get(i).get("blue"));
        }

        double red = median.evaluate(redArray);
        double green = median.evaluate(greenArray);
        double blue = median.evaluate(blueArray);

        Mat features = new Mat(1, 3, CvType.CV_32FC1);
        features.put(0, 0, blue);
        features.put(0, 1, green);
        features.put(0, 2, red);

        double purplePrediction = purpleSvm.predict(features);
        double redPrediction = redSvm.predict(features);
        double whitePrediction = whiteSvm.predict(features);

        if (whitePrediction > 0.0) {
            return SeedColor.WHITE;
        } else if (purplePrediction > 0.0) {
            return SeedColor.PURPLE;
        } else if (redPrediction > 0.0) {
            return SeedColor.RED;
        } else {
            return SeedColor.UNKNOWN;
        }
    }

    public SeedColor classifySeeds(Mat image, Double scale) {
        List<MatOfPoint> contours = Helper.getContours(image);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        int purpleCount = 0;
        int redCount = 0;
        int whiteCount = 0;

        for (MatOfPoint contour : contours) {
            Double area = scale * Imgproc.contourArea(contour);
            if (area < 30.0 && area > 5.0) {
                List<Map<String,String>> seedData = SeedUtils.getSeedData(contour, image, seedBuffer, 0.5);
                if (!seedData.isEmpty()) {
                    SeedColor prediction = classifySingleSeed(seedData);

                    switch (prediction) {
                        case PURPLE:
                            ++purpleCount;
                            break;
                        case RED:
                            ++redCount;
                            break;
                        case WHITE:
                            ++whiteCount;
                            break;
                    }
                }
            }
            contour.release();
        }

        seedBuffer.release();

        if (purpleCount > redCount && purpleCount > whiteCount) {
            return SeedColor.PURPLE;
        } else if (redCount > purpleCount && redCount > whiteCount) {
            return SeedColor.RED;
        } else if (whiteCount > purpleCount && whiteCount > redCount) {
            return SeedColor.WHITE;
        }

        return SeedColor.UNKNOWN;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        SeedClassifier classifier = new SeedClassifier();

        // BRUTEFORCE is used for reproducibility
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

            image.release();
            extractedColorChecker.release();
            findColorChecker.fillColorChecker(calibrated, quad);

            Mat mask = SeedUtils.getMask(calibrated, scale);
            Mat filtered = SeedUtils.filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();

            System.out.println(classifier.classifySeeds(filtered, scale).getValue());
            filtered.release();
        }
    }

    private enum SeedColor {
        PURPLE("Purple"),
        RED("Red"),
        UNKNOWN("Unknown"),
        WHITE("White");

        private String value;

        SeedColor(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }
    }
}
