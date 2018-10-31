package seedcounter.examples;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Range;
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
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

class SeedDatasetClustering {
    private static final String INPUT_DIR = "src/seedcounter/examples/seed_dataset_clustering";
    private static final String RESULT_DIR = "src/seedcounter/examples/seed_dataset_clustering_results";
    private static final String REFERENCE_FILE = "reference.png";

    private static final Order ORDER = Order.THIRD;
    private static final ColorSpace FEATURE_SPACE = ColorSpace.RGB;
    private static final ColorSpace TARGET_SPACE = ColorSpace.RGB;
    private static final boolean MASK_BY_CALIBRATED = true;
    private static final boolean LOG_IMAGES = false;

    private enum FileType {
        TWO_CLASSES,
        FOUR_CLASSES,
        OTHER
    }

    private static class Experiment {
        public Experiment(Path path) {
            this.path = path;
        }

        private Path path;

        public String getAbsolutePath() {
            return path.toFile().getAbsolutePath();
        }

        public boolean isCC() {
            return fileName().contains("_cc_");
        }

        public Mat getCC(Mat image) {
            return new Mat(image,
                    new Range(0, image.rows()),
                    new Range(0, 4 * image.cols() / 10));
        }

        public Map<Integer,Mat> getClasses(Mat image) {
            Range firstCols = new Range(4 * image.cols() / 10, 2 * image.cols() / 3);
            Range secondCols = new Range(2 * image.cols() / 3, image.cols());
            Range firstRows = new Range(0, 4 * image.rows() / 10);
            Range secondRows = new Range(4 * image.rows() / 10, image.rows());
            Range totalRows = new Range(0, image.rows());

            Map<Integer,Mat> result = new HashMap<>();

            switch (fileType()) {
                case FOUR_CLASSES:
                    result.put(1, new Mat(image, firstRows, firstCols));
                    result.put(2, new Mat(image, firstRows, secondCols));
                    result.put(3, new Mat(image, secondRows, firstCols));
                    result.put(4, new Mat(image, secondRows, secondCols));
                    break;
                case TWO_CLASSES:
                    result.put(getDigit(5), new Mat(image, totalRows, firstCols));
                    result.put(getDigit(12), new Mat(image, totalRows, secondCols));
                    break;
            }

            return result;
        }

        private Integer getDigit(Integer index) {
            return Integer.parseInt(String.valueOf(fileName().charAt(index)));
        }

        public String fileName() {
            return path.getFileName().toString();
        }

        public String camera() {
            return path.getParent().getFileName().toString();
        }

        public FileType fileType() {
            String fileName = path.getFileName().toString();

            if (!fileName.endsWith(".jpg") || fileName.contains("label") ||
                    !fileName.contains("exp")) {
                return FileType.OTHER;
            }

            if (fileName.startsWith("class1_class2_class3_class4")) {
                return FileType.FOUR_CLASSES;
            }

            return FileType.TWO_CLASSES;
        }

        public File getDirectory() {
            File f =  new File(RESULT_DIR + "/" + path.getParent().getFileName().toString());

            if (LOG_IMAGES) {
                f.mkdir();
            }

            return new File(f.getPath() + "/" + fileType());
        }
    }

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MatchingModel matchingModel = new MatchingModel(
                BRISK.create(), BRISK.create(),
                DescriptorMatcher.BRUTEFORCE_HAMMING, 0.75f
        );
        FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, matchingModel);
        RegressionModel model = RegressionFactory.createModel(ORDER);

        PrintWriter seedLog = new PrintWriter(RESULT_DIR + "/seed_log.txt");
        Map<String, String> seedData = new HashMap<>();
        seedData.put("header", "1");

        List<Experiment> inputFiles = Files.walk(Paths.get(INPUT_DIR))
                .map(x -> new Experiment(x))
                .filter(x -> x.fileType() != FileType.OTHER)
                .collect(Collectors.toList());

        if (LOG_IMAGES) {
            File resultDirectory = new File(RESULT_DIR);
            resultDirectory.mkdir();
        }

        for (Experiment file : inputFiles) {
            System.out.println(file.getAbsolutePath());
            seedData.put("camera", file.camera());
            seedData.put("file", file.fileName());
            seedData.put("colorchecker", file.isCC() ? "1" : "0");
            seedData.put("experiment", file.fileType().name());

            File resultDirectory = file.getDirectory();
            if (LOG_IMAGES) {
                resultDirectory.mkdir();
            }

            Mat image = Imgcodecs.imread(file.getAbsolutePath(),
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
            Map<Integer,Mat> classes = file.getClasses(image);

            Mat cc = null;
            Mat extractedColorChecker = null;
            ColorChecker checker = null;
            // default scale is calculated by A4 size
            double scale = (297.0 / 0.8 / image.cols()) * (210.0 / 0.85 / image.rows());

            if (file.isCC()) {
                cc = file.getCC(image);
                Quad quad = findColorChecker.findBestFitColorChecker(cc);

                if (quad.getArea() / cc.cols() / cc.rows() < 0.25) {
                    extractedColorChecker = quad.getTransformedField(cc);
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

                if (LOG_IMAGES) {
                    File ccDirectory = new File(resultDirectory.getAbsolutePath() + "/" + "ColorChecker");
                    ccDirectory.mkdir();
                    Imgcodecs.imwrite(ccDirectory.getAbsolutePath() + "/" + file.fileName(), cc);
                }
            }

            int seedNumber = 0;

            for (Integer class_ : classes.keySet()){
                File classDirectory = new File(resultDirectory.getAbsolutePath() + "/" + class_);
                if (LOG_IMAGES) {
                    classDirectory.mkdir();
                }

                Mat source = classes.get(class_);
                Mat mask;
                Mat colorData;
                Mat forFilter;

                if (checker != null) {
                    Mat calibrated;
                    try {
                        calibrated = checker.calibrate(source, model, FEATURE_SPACE, TARGET_SPACE);
                        if (MASK_BY_CALIBRATED) {
                            mask = SeedUtils.getMask(calibrated, scale);
                        } else {
                            mask = SeedUtils.getMask(source, scale);
                        }

                        colorData = SeedUtils.filterByMask(calibrated, mask);
                        if (MASK_BY_CALIBRATED) {
                            forFilter = colorData;
                        } else {
                            forFilter = SeedUtils.filterByMask(source, mask);
                        }
                        seedData.put("calibrated", "1");
                        source.release();
                    } catch (IllegalStateException e) {
                        seedData.put("calibrated", "0");
                        System.out.println("Couldn't calibrate the image " + file.fileName() + " using the source...");
                        mask = SeedUtils.getMask(source, scale);
                        colorData = SeedUtils.filterByMask(source, mask);
                        forFilter = colorData;
                        calibrated = source;
                    }

                    if (LOG_IMAGES) {
                        Imgcodecs.imwrite(classDirectory.getAbsolutePath() + "/" + file.fileName(), calibrated);
                    }
                    calibrated.release();
                } else {
                    seedData.put("calibrated", "0");
                    mask = SeedUtils.getMask(source, scale);
                    colorData = SeedUtils.filterByMask(source, mask);
                    forFilter = colorData;

                    if (LOG_IMAGES) {
                        Imgcodecs.imwrite(classDirectory.getAbsolutePath() + "/" + file.fileName(), source);
                    }
                    source.release();
                }

                seedData.put("class", class_.toString());
                seedNumber = SeedUtils.printSeeds(colorData, forFilter, seedLog, seedData, scale, seedNumber);

                mask.release();
                colorData.release();
                if (colorData != forFilter) {
                    forFilter.release();
                }

            }

            if (cc != null) {
                cc.release();
            }
            if (extractedColorChecker != null) {
                extractedColorChecker.release();
            }

            image.release();
        }

        seedLog.close();
    }
}
