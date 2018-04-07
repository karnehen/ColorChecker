package seedcounter.common;

import java.util.ArrayList;
import java.util.List;

import javafx.util.Pair;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class Helper {
    public static Mat whiteThreshold(Mat image) {
        Mat result = image.clone();
        Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(result, result, 200.0, 255.0, Imgproc.THRESH_BINARY_INV);

        return result;
    }

    public static Mat binarizeSeed(Mat image,
            List<Pair<Scalar, Scalar>> targetsAndRanges) {
        HSVBinarization hsv = new HSVBinarization(targetsAndRanges);
        return hsv.apply(image);
    }

    public static Mat filterByMask(Mat image, Mat mask) {
        List<Mat> channels = new ArrayList<>();
        Core.split(image, channels);
        for (int i = 0; i < 3; ++i) {
            Mat c = channels.get(i);
            Core.bitwise_and(c, mask, c);
            channels.set(i, c);
        }
        Mat filtered = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
        Core.merge(channels, filtered);
        for (Mat c : channels) {
            c.release();
        }

        return filtered;
    }

    public static List<MatOfPoint> getContours(Mat image) {
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        gray.release();
        hierarchy.release();
        return contours;
    }
}
