package seedcounter;

import java.nio.DoubleBuffer;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Color {
    private final double red;
    private final double green;
    private final double blue;
    private final double lightness;
    private final double a;
    private final double b;

    private Color(double red, double green, double blue) {
        this.red = red;
        this.green = green;
        this.blue = blue;

        Scalar lab = bgrToLabScalar(new Scalar(blue, green, red));
        this.lightness = lab.val[0];
        this.a = lab.val[1];
        this.b = lab.val[2];
    }

    public Color(Color color) {
        this(color.red, color.green, color.blue);
    }

    public double red() {
        return this.red;
    }

    public double green() {
        return this.green;
    }

    public double blue() {
        return this.blue;
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

    public double X() {
        return linearizeRGB(this.red) * 0.4124
                + linearizeRGB(this.green) * 0.3576
                + linearizeRGB(this.blue) * 0.1805;
    }

    public double Y() {
        return linearizeRGB(this.red) * 0.2126
                + linearizeRGB(this.green) * 0.7152
                + linearizeRGB(this.blue) * 0.0722;
    }

    public double Z() {
        return linearizeRGB(this.red) * 0.0193
                + linearizeRGB(this.green) * 0.1192
                + linearizeRGB(this.blue) * 0.9505;
    }

    public DoubleBuffer toBGR() {
        return DoubleBuffer.wrap(new double [] {this.blue, this.green, this.red});
    }

    public DoubleBuffer toXYZ() {
        return DoubleBuffer.wrap(new double [] {this.X(), this.Y(), this.Z()});
    }

    public static Color ofBGR(Scalar color) {
        return ofBGR(color.val);
    }

    // TODO: convert all to DoubleBuffer
    public static Color ofBGR(double[] color) {
        return new Color(color[2], color[1], color[0]);
    }

    public static Color ofBGR(DoubleBuffer color) {
        return new Color(color.get(color.position() + 2),
                color.get(color.position() + 1), color.get(color.position()));
    }

    public static Color ofXYZ(double[] color) {
        return new Color(
            inverseLinearizeRGB(
                3.2404542 * color[0] - 1.5371385 * color[1] - 0.4985314 * color[2]),
            inverseLinearizeRGB(
                -0.9692660 * color[0] + 1.8760108 * color[1] + 0.0415560 * color[2]),
            inverseLinearizeRGB(
                0.0556434 * color[0] - 0.2040259 * color[1] + 1.0572252 * color[2])
        );
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
