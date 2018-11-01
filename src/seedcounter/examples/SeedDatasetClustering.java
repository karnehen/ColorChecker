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

    // порядок регрессии
    private static final Order ORDER = Order.THIRD;
    // цветовое пространство для признаков регрессионной модели
    private static final ColorSpace FEATURE_SPACE = ColorSpace.RGB;
    // цветовое пространство для ответов регрессионной модели
    private static final ColorSpace TARGET_SPACE = ColorSpace.RGB;
    // распознавать зерна по калиброваному изображению (иначе по исходному)
    private static final boolean MASK_BY_CALIBRATED = true;
    // записать нарезаные сектора изображений в файлы - для дебага
    private static final boolean LOG_IMAGES = false;

    private enum FileType {
        TWO_CLASSES,
        FOUR_CLASSES,
        OTHER
    }

    private static class Experiment {
        // граница между областью с ColorChecker и первой колонкой зерен
        private final double FIRST_COLUMN_BORDER = 0.38;
        // граница между первой и второй колонками зерен
        private final double SECOND_COLUMN_BORDER = 0.65;
        // граница между строчками зерен
        private final double ROW_BORDER = 0.45;

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
            Range firstCols = new Range(firstColumnBorder(image), secondColumnBorder(image));
            Range secondCols = new Range(secondColumnBorder(image), image.cols());
            Range firstRows = new Range(0, rowBorder(image));
            Range secondRows = new Range(rowBorder(image), image.rows());
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
                    result.put(leftClass(), new Mat(image, totalRows, firstCols));
                    result.put(rightClass(), new Mat(image, totalRows, secondCols));
                    break;
            }

            return result;
        }

        public int xOffset(Mat image, int class_) {
            switch (fileType()) {
                case FOUR_CLASSES:
                    if (class_ % 2 == 1) {
                        return firstColumnBorder(image);
                    } else {
                        return secondColumnBorder(image);
                    }
                case TWO_CLASSES:
                    if (class_ == leftClass()) {
                        return firstColumnBorder(image);
                    } else if (class_ == rightClass()) {
                        return secondColumnBorder(image);
                    }
            }

            throw new IllegalStateException("Wrong filename or wrong class!");
        }

        public int yOffset(Mat image, int class_) {
            switch (fileType()) {
                case FOUR_CLASSES:
                    if (class_ < 3) {
                        return 0;
                    } else {
                        return rowBorder(image);
                    }
                case TWO_CLASSES:
                    return 0;
            }

            throw new IllegalStateException("Wrong filename!");
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

        private int leftClass() {
            return getDigit(5);
        }

        private int rightClass() {
            return getDigit(12);
        }

        private Integer getDigit(Integer index) {
            return Integer.parseInt(String.valueOf(fileName().charAt(index)));
        }

        private int firstColumnBorder(Mat image) {
            return (int) (image.cols() * FIRST_COLUMN_BORDER);
        }

        private int secondColumnBorder(Mat image) {
            return (int) (image.cols() * SECOND_COLUMN_BORDER);
        }

        private int rowBorder(Mat image) {
            return (int) (image.rows() * ROW_BORDER);
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
        /* объект, логирующий информацию по зернам в файл. Параметры
         *   threshold - отфильтровать зерна, площдь которых меньше доли X от площади bounding box вокруг зерна
         *   whiteThreshold - отфильтровать зерна, у которых процентиль 10% от яркости больше X
         *   filterByArea - отфильтровать зерна с площадью <5мм2 или >30мм2 (границы заданы константами в SeedUtils)
         * рекомендуемые параметры: 0.5, 150.0, true
         * без дополнительной фильтрации: 0.0, 255.0, false
         */
        SeedUtils seedUtils = new SeedUtils(0.5, 150.0, true);

        // Куда писать результат
        PrintWriter seedLog = new PrintWriter(RESULT_DIR + "/seed_log.txt");
        /* Вспомогательная структура, для заполнения общих между отдельным записями полей - например названия
         * исходного файла или модели камеры. Формат - "название колонки": "значения поля". Ланные пишутся в файл,
         * отсортированные по названию колонки. В нее можно записывать какую-нибудь дополнительную информация -
         * но важно, чтобы "колонка":"значение" присутствовало на всех вызовах записи в файл (seedUtils.printSeeds),
         * иначе файл "поедет".
         */
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
            // модель камеры берется из названия директории с изображениями
            seedData.put("camera", file.camera());
            seedData.put("file", file.fileName());
            // есть ли ColorChecker на исходном изображении
            seedData.put("colorchecker", file.isCC() ? "1" : "0");
            // 2 или 4 класса
            seedData.put("experiment", file.fileType().name());

            File resultDirectory = file.getDirectory();
            if (LOG_IMAGES) {
                resultDirectory.mkdir();
            }

            /* Считываем изображение - здесь же можно провести дополнительную предобработку
               (например исправление перспективы) */
            Mat image = Imgcodecs.imread(file.getAbsolutePath(),
                    Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
            /* Разбивка изображения на сектора, соответствующие классам */
            Map<Integer,Mat> classes = file.getClasses(image);

            Mat cc = null;
            Mat extractedColorChecker = null;
            ColorChecker checker = null;
            // Если не нашли ColorChecker - вычисляем масштаб по размеру A4
            double scale = (297.0 / 0.8 / image.cols()) * (210.0 / 0.85 / image.rows());

            if (file.isCC()) {
                // Выделяем на изображении сектор с ColorChecker (левая часть изображения до FIRST_COLUMN_BORDER
                cc = file.getCC(image);
                Quad quad = findColorChecker.findBestFitColorChecker(cc);

                if (quad.getArea() / cc.cols() / cc.rows() < 0.25) {
                    extractedColorChecker = quad.getTransformedField(cc);
                    checker = new ColorChecker(extractedColorChecker);
                    // ColorChecker слишком сильно отличается от эталонного изображения
                    if (checker.labDeviationFromReference() > 25) {
                        checker = null;
                        System.out.println("Couldn't detect colorchecker: colors deviate too much from the reference");
                        extractedColorChecker.release();
                        extractedColorChecker = null;
                    } else {
                        scale = checker.pixelArea(quad);
                    }
                } else { // Область, занимаемая найденный ColorChecker больше 0.25 от области соответствующего сектора
                    checker = null;
                    System.out.println("Couldn't detect colorchecker: detected area is too large");
                }

                if (LOG_IMAGES) {
                    File ccDirectory = new File(resultDirectory.getAbsolutePath() + "/" + "ColorChecker");
                    ccDirectory.mkdir();
                    Imgcodecs.imwrite(ccDirectory.getAbsolutePath() + "/" + file.fileName(), cc);
                }
            }

            // Обнуляем счетчик номера зерна
            seedUtils.setSeedNumber(0);

            for (Integer class_ : classes.keySet()){
                File classDirectory = new File(resultDirectory.getAbsolutePath() + "/" + class_);
                if (LOG_IMAGES) {
                    classDirectory.mkdir();
                }

                // Сектор с зернами класса class_
                Mat source = classes.get(class_);
                // Маска с зернами. Вычисляется SeedUtils.getMask - вместо него можно вставить другой метод
                // распознавания зерна (сейчас там применяется HSVBinarization+отсечение белого цвета+морфология)
                Mat mask;
                // Отфильтрованый по маске сектор (исходный или калиброваный), с которого берутся значения цветов
                Mat colorData;
                /* Отфильтрованый по маске сектор (исходный или калиброваный), по которому вычисляются контуры и дополнительные
                 * фильтрация по квантилю яркости (whiteThreshold). Может отличаться от colorData.
                 * Выделено отдельно от colorData, для возможности проведения согласованых экспериментов. Например,
                 * если хотим сравнить случай без калибровки вообще со случаем, когда цвет берется по калиброванному
                 * изображению, а зерна выделяются по исходному, в forFilter должно быть пропущенное через маску
                 * исходное изображение - иначе количество распознанных зерен может различаться.
                 */
                Mat forFilter;

                // Калибровка сектора
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
                        // удалось откалибровать
                        seedData.put("calibrated", "1");
                        source.release();
                    } catch (IllegalStateException e) {
                        // не удалось откалибровать
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
                    // калибровка не проводилась
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
                // Смещение, чтобы в лог писались не координаты пикселя в секторе, а координаты пикселя на всем изображении
                seedUtils.setXOffset(file.xOffset(image, class_));
                seedUtils.setYOffset(file.yOffset(image, class_));
                // Пишем информацию о зернах в файл. Здесь же происходит вычисление контуров зерен, по отфильтрованному изображению
                seedUtils.printSeeds(colorData, forFilter, seedLog, seedData, scale);

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
