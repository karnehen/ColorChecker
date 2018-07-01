package seedcounter.examples;

import javafx.util.Pair;
import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SIFT;
import seedcounter.colorchecker.ColorChecker;
import seedcounter.colorchecker.FindColorChecker;
import seedcounter.colorchecker.MatchingModel;
import seedcounter.colormetric.ColorMetric;
import seedcounter.colormetric.EuclideanLab;
import seedcounter.colormetric.EuclideanRGB;
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
import java.util.*;

class SeedDataset {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/seed_dataset_results";
    private static final String REFERENCE_FILE = "reference.png";
    // targets and ranges
    private static final List<Pair<Scalar, Scalar>> SEED_TYPES = Arrays.asList(
            new Pair<>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
            new Pair<>(new Scalar(17, 67, 232), new Scalar(50, 50, 50))
        );

    private static void printMap(PrintWriter writer, Map<String, String> map) {
        boolean header = map.containsKey("header");
        map.remove("header");
        StringBuilder builder = new StringBuilder();
        if (header) {
            for (String key : map.keySet()) {
                builder.append(key);
                builder.append("\t");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("\n");
        }
        for (String key : map.keySet()) {
            builder.append(map.get(key));
            builder.append("\t");
        }
        builder.deleteCharAt(builder.length() - 1);
        writer.println(builder.toString());
    }

    private static void printSingleSeed(MatOfPoint contour, Mat image, Mat seedBuffer,
            PrintWriter writer, Map<String, String> data) {
        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
                new Scalar(255.0), Core.FILLED);

        int minX = image.cols() - 1;
        int maxX = 0;
        int minY = image.rows() - 1;
        int maxY = 0;

        for (Point point : contour.toList()) {
            if (point.x < minX) {
                minX = (int) point.x;
            }
            if (point.x > maxX) {
                maxX = (int) point.x;
            }
            if (point.y < minY) {
                minY = (int) point.y;
            }
            if (point.y > maxY) {
                maxY = (int) point.y;
            }
        }

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                if (seedBuffer.get(y, x)[0] > 0.0) {
                    double[] color = image.get(y, x);
                    if (color[0] + color[1] + color[2] > 0.0) {
                        data.put("x", String.valueOf(x));
                        data.put("y", String.valueOf(y));
                        data.put("blue", String.valueOf(color[0]));
                        data.put("green", String.valueOf(color[1]));
                        data.put("red", String.valueOf(color[2]));
                        printMap(writer, data);
                    }
                }
            }
        }
        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0, new Scalar(0.0));
    }

    private static void printSeeds(Mat image, PrintWriter writer,
            Map<String, String> data, Double scale) {
        List<MatOfPoint> contours = Helper.getContours(image);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        for (int i = 0; i < contours.size(); ++i) {
            data.put("seed_number", String.valueOf(i));
            MatOfPoint contour = contours.get(i);
            Double area = scale * Imgproc.contourArea(contour);
            if (area < 50.0) {
                data.put("area", area.toString());
                printSingleSeed(contours.get(i), image, seedBuffer, writer, data);
            }
            contour.release();
        }

        seedBuffer.release();
    }

    private static Mat filterByMask(Mat image, Mat mask) {
        Mat filtered = Helper.filterByMask(image, mask);
        Range rows = new Range(filtered.rows() / 4, 3 * filtered.rows() / 4);
        Range cols = new Range(filtered.cols() / 4, 3 * filtered.cols() / 4);

        Mat result = new Mat(filtered, rows, cols);
        filtered.release();

        return result;
    }

    private static Mat getMask(Mat image) {
        Mat mask = Helper.binarizeSeed(image, SEED_TYPES);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(50, 50));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

        Mat whiteMask = Helper.whiteThreshold(image);
        kernel.release();
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
        Imgproc.morphologyEx(whiteMask, whiteMask, Imgproc.MORPH_CLOSE, kernel);

        Core.bitwise_and(mask, whiteMask, mask);
        whiteMask.release();
        kernel.release();

        return mask;
    }

    public static void main(String[] args) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // BRUTEFORCE is used for reproducibility
        MatchingModel matchingModel = new MatchingModel(
                SIFT.create(), SIFT.create(),
                DescriptorMatcher.BRUTEFORCE, 0.7f
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

        List<ColorMetric> metrics = new ArrayList<>();
        metrics.add(EuclideanRGB.create());
        metrics.add(EuclideanLab.create());

        PrintWriter calibrationLog = new PrintWriter(RESULT_DIR + "/calibration_log.txt");
        Map<String, String> calibrationData = new HashMap<>();
        calibrationData.put("header", "1");

        PrintWriter seedLog = new PrintWriter(RESULT_DIR + "/seed_log.txt");
        Map<String, String> seedData = new HashMap<>();
        seedData.put("header", "1");

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
            calibrationData.put("file", fileName);
            seedData.put("file", fileName);

            Quad quad = findColorChecker.findColorChecker(image);
            Mat extractedColorChecker = quad.getTransformedField(image);
            ColorChecker checker = new ColorChecker(extractedColorChecker);
            Double scale = checker.pixelArea(quad);
            calibrationData.put("scale", scale.toString());

            for (ColorMetric cm : metrics) {
                String metricName = cm.getClass().getSimpleName();
                calibrationData.put("source:" + metricName,
                        String.valueOf(checker.getCellColors(extractedColorChecker).
                                calculateMetric(cm)));
            }

            String name = model.getClass().getSimpleName();
            calibrationData.put("model", name);
            seedData.put("model", name);

            try {
                Mat calibratedChecker = checker.calibrate(extractedColorChecker, model, ColorSpace.RGB, ColorSpace.RGB);
                for (ColorMetric cm : metrics) {
                    String metricName = cm.getClass().getSimpleName();
                    calibrationData.put("calibrated:" + metricName,
                            String.valueOf(checker.getCellColors(calibratedChecker).
                                    calculateMetric(cm)));
                }
                calibratedChecker.release();;
            } catch (IllegalStateException e) {
                System.out.println("Couldn't calibrate the image " + fileName + " skipping...");
                image.release();
                extractedColorChecker.release();
                continue;
            }

            printMap(calibrationLog, calibrationData);
            Mat calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.RGB);
            image.release();
            extractedColorChecker.release();
            findColorChecker.fillColorChecker(calibrated, quad);

            Mat mask = getMask(calibrated);
            Mat filtered = filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();

            printSeeds(filtered, seedLog, seedData, scale);
            filtered.release();
        }
        calibrationLog.close();
        seedLog.close();
    }
}
