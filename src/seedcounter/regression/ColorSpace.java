package seedcounter.regression;

import seedcounter.colormetric.Color;

import java.nio.DoubleBuffer;

public enum ColorSpace {
    RGB(false, false),
    RGB_LINEAR(false, true),
    XYZ(true, false),
    XYZ_LINEAR(true, true);

    private final boolean isXYZ;
    private final boolean isLinear;

    ColorSpace(boolean isXYZ, boolean isLinear) {
        this.isXYZ = isXYZ;
        this.isLinear = isLinear;
    }

    private static double linearizeRGB(double channelColor) {
        channelColor /= 255.0;
        if (channelColor > 0.04045) {
            channelColor = Math.pow((channelColor + 0.055) / 1.055, 2.4);
        } else {
            channelColor /= 12.92;
        }

        return channelColor * 100.0;
    }

    private static double inverseLinearizeRGB(double channelColor) {
        channelColor /= 100.0;
        if (channelColor > 0.0031308) {
            channelColor = 1.055 * Math.pow(channelColor, 1.0 / 2.4) - 0.055;
        } else {
            channelColor *= 12.92;
        }

        return channelColor * 255.0;
    }

    public DoubleBuffer convertFromBGR(DoubleBuffer color, boolean inplace) {
        if (!inplace) {
            color = DoubleBuffer.wrap(new double[] {Color.channel(color, 0),
                    Color.channel(color, 1), Color.channel(color, 2)});
        }

        double b = Color.channel(color, 0);
        double g = Color.channel(color, 1);
        double r = Color.channel(color, 2);

        if (isLinear) {
            b = linearizeRGB(b);
            g = linearizeRGB(g);
            r = linearizeRGB(r);
        }

        if (isXYZ) {
            color.put(color.position(), r * 0.4124 + g * 0.3576 + b * 0.1805);
            color.put(color.position() + 1, r * 0.2126  + g * 0.7152 + b * 0.0722);
            color.put(color.position() + 2, r * 0.0193  + g * 0.1192 + b * 0.9505);
        } else {
            color.put(color.position(), b);
            color.put(color.position() + 1, g);
            color.put(color.position() + 2, r);
        }

        return color;
    }

    public void convertToBGR(DoubleBuffer color) {
        if (isXYZ) {
            double x = Color.channel(color, 0);
            double y = Color.channel(color, 1);
            double z = Color.channel(color, 2);
            color.put(color.position(), 0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
            color.put(color.position() + 1, -0.9692660 * x + 1.8760108 * y + 0.0415560 * z);
            color.put(color.position() + 2, 3.2404542 * x - 1.5371385 * y - 0.4985314 * z);
        }
        if (isLinear) {
            color.put(color.position(), inverseLinearizeRGB(Color.channel(color, 0)));
            color.put(color.position() + 1, inverseLinearizeRGB(Color.channel(color, 1)));
            color.put(color.position() + 2, inverseLinearizeRGB(Color.channel(color, 2)));
        }
    }
}
