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

    private static List<Map<String,String>> getSeedData(MatOfPoint contour, Mat image, Mat seedBuffer,
                                                        double threshold) {
        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
                new Scalar(255.0), Core.FILLED);
        List<Map<String,String>> result = new ArrayList<>();

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
                        Map<String,String> map = new HashMap<>();
                        map.put("x", String.valueOf(x));
                        map.put("y", String.valueOf(y));
                        map.put("blue", String.valueOf(color[0]));
                        map.put("green", String.valueOf(color[1]));
                        map.put("red", String.valueOf(color[2]));
                        result.add(map);
                    }
                }
            }
        }

        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
                new Scalar(0.0), Core.FILLED);

        if (result.size() / (maxX - minX + 1.0) / (maxY - minY + 1.0) < threshold) {
            result.clear();
        }

        return result;
    }

    private static void printSeedData(Map<String,String> data, List<Map<String,String>> seedData, PrintWriter writer) {
        for (Map<String,String> map : seedData) {
            for (String key : map.keySet()) {
                data.put(key, map.get(key));
            }
            printMap(writer, data);
        }
    }

    private static void printSeeds(Mat image, PrintWriter writer,
            Map<String, String> data, Double scale) {
        List<MatOfPoint> contours = Helper.getContours(image);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        int seedNumber = 0;
        for (MatOfPoint contour : contours) {
            Double area = scale * Imgproc.contourArea(contour);
            if (area < 30.0 && area > 5.0) {
                List<Map<String,String>> seedData = getSeedData(contour, image, seedBuffer, 0.5);
                if (!seedData.isEmpty()) {
                    data.put("seed_number", String.valueOf(seedNumber++));
                    data.put("area", area.toString());
                    printSeedData(data, seedData, writer);
                }
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

    private static Mat getMask(Mat image, Double scale) {
        Mat mask = Helper.binarizeSeed(image, SEED_TYPES);
        Mat whiteMask = Helper.whiteThreshold(image);
        Core.bitwise_and(mask, whiteMask, mask);
        whiteMask.release();

        int kernelSize = (int)(1.5 / Math.sqrt(scale));
        if (kernelSize < 10) {
            kernelSize = 10;
        }
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        kernel.release();

        return mask;
    }

    public static void main(String[] args) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

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

        PrintWriter seedLog = new PrintWriter(RESULT_DIR + "/seed_log.txt");
        Map<String, String> seedData = new HashMap<>();
        seedData.put("header", "1");

        for (String fileName : inputFiles) {
            System.out.println(fileName);
            Mat image = Imgcodecs.imread(fileName,
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
            seedData.put("file", fileName);

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

            seedData.put("type", "source");
            Mat mask = getMask(image, scale);
            Mat filtered = filterByMask(image, mask);
            mask.release();
            printSeeds(filtered, seedLog, seedData, scale);
            filtered.release();

            image.release();
            extractedColorChecker.release();
            findColorChecker.fillColorChecker(calibrated, quad);

            seedData.put("type", "calibrated");
            mask = getMask(calibrated, scale);
            filtered = filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();
            printSeeds(filtered, seedLog, seedData, scale);
            filtered.release();
        }
        seedLog.close();
    }
}
