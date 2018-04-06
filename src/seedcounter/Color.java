package seedcounter;

import java.nio.DoubleBuffer;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Color {
    private final DoubleBuffer bgr;
    private final double lightness;
    private final double a;
    private final double b;

    private Color(double red, double green, double blue) {
        this.bgr = DoubleBuffer.wrap(new double[] {blue, green, red});

        Scalar lab = bgrToLabScalar(new Scalar(blue, green, red));
        this.lightness = lab.val[0];
        this.a = lab.val[1];
        this.b = lab.val[2];
    }

    public Color(DoubleBuffer bgr) {
        this.bgr = bgr;

        Scalar lab = bgrToLabScalar(new Scalar(channel(bgr, 0), channel(bgr, 1), channel(bgr, 2)));
        this.lightness = lab.val[0];
        this.a = lab.val[1];
        this.b = lab.val[2];
    }

    public Color(Color color) {
        this.bgr = color.bgr;
        this.lightness = color.lightness;
        this.a = color.a;
        this.b = color.b;
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
        return this.lightness;
    }

    public double a() {
        return this.a;
    }

    public double b() {
        return this.b;
    }

    private Scalar bgrToLabScalar(Scalar color) {
        Mat bgr = new Mat(1, 1, CvType.CV_32FC3, color);
        Mat lab = new Mat(1, 1, CvType.CV_32FC3);
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);

        return new Scalar(lab.get(0,  0));
    }

    public static double linearizeRGB(double channelColor) {
        channelColor /= 255.0;
        if (channelColor > 0.04045) {
            channelColor = Math.pow((channelColor + 0.055) / 1.055, 2.4);
        } else {
            channelColor /= 12.92;
        }

        return channelColor * 100.0;
    }

    public static double inverseLinearizeRGB(double channelColor) {
        channelColor /= 100.0;
        if (channelColor > 0.0031308) {
            channelColor = 1.055 * Math.pow(channelColor, 1.0 / 2.4) - 0.055;
        } else {
            channelColor *= 12.92;
        }

        return channelColor * 255.0;
    }
}
