package seedcounter.colormetric;

import java.nio.DoubleBuffer;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Color {
    private final DoubleBuffer bgr;
    private DoubleBuffer lab;

    public Color(DoubleBuffer bgr) {
        this.bgr = bgr;
        this.lab = null;
    }

    public static double channel(DoubleBuffer color, int channel) {
        return color.get(color.position() + channel);
    }

    public double red() {
        return channel(bgr, 2);
    }

    public double green() {
        return channel(bgr, 1);
    }

    public double blue() {
        return channel(bgr, 0);
    }

    public double lightness() {
        calculateLab();
        return channel(lab, 0) / 2.55;
    }

    public double a() {
        calculateLab();
        return channel(lab, 1) - 128.0;
    }

    public double b() {
        calculateLab();
        return channel(lab, 2) - 128.0;
    }

    private void calculateLab() {
        if (this.lab == null) {
            Scalar lab = bgrToLabScalar(new Scalar(channel(bgr, 0), channel(bgr, 1), channel(bgr, 2)));
            this.lab = DoubleBuffer.wrap(lab.val);
        }
    }

    private Scalar bgrToLabScalar(Scalar color) {
        Mat bgr = new Mat(1, 1, CvType.CV_8UC4, color);
        Mat lab = new Mat(1, 1, CvType.CV_32FC3);
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);

        return new Scalar(lab.get(0,  0));
    }

}
