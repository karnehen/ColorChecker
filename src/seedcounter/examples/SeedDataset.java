package seedcounter.examples;

import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

class SeedDataset {
    private static final String INPUT_FILES = "src/seedcounter/examples/input_files.txt";
    private static final String RESULT_DIR = "src/seedcounter/examples/seed_dataset_results";
    private static final String REFERENCE_FILE = "reference.png";

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
                List<Map<String,String>> seedData = SeedUtils.getSeedData(contour, image, seedBuffer, 0.5);
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

        RegressionModel model = RegressionFactory.createModel(Order.FIRST);

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

            seedData.put("type", "source");
            Mat mask = SeedUtils.getMask(calibrated, scale);
            Mat filtered = SeedUtils.filterByMask(image, mask);
            printSeeds(filtered, seedLog, seedData, scale);
            filtered.release();

            image.release();
            extractedColorChecker.release();

            seedData.put("type", "calibrated");
            filtered = SeedUtils.filterByMask(calibrated, mask);
            calibrated.release();
            mask.release();
            printSeeds(filtered, seedLog, seedData, scale);
            filtered.release();
        }
        seedLog.close();
    }
}
