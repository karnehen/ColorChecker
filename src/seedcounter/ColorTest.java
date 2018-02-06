package seedcounter;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;

import seedcounter.Color;

public class ColorTest {
	private final double RGB_MAX = 255.0;
	private final double RED_X = 41.24;
	private final double RED_Y = 21.26;
	private final double RED_Z = 1.93;
	private final double GREEN_X = 35.76;
	private final double GREEN_Y = 71.52;
	private final double GREEN_Z = 11.92;
	private final double BLUE_X = 18.05;
	private final double BLUE_Y = 7.22;
	private final double BLUE_Z = 95.05;

	@BeforeClass
	public static void setUpBeforeClass() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	@Test
	public void BGRtoXYZRed() {
		Color red = Color.ofBGR(new double[] {0.0, 0.0, RGB_MAX});
		Assert.assertEquals(RED_X, red.X(), 0.5);
		Assert.assertEquals(RED_Y, red.Y(), 0.5);
		Assert.assertEquals(RED_Z, red.Z(), 0.5);
	}

	@Test
	public void BGRtoXYZGreen() {
		Color green = Color.ofBGR(new double[] {0.0, RGB_MAX, 0.0});
		Assert.assertEquals(GREEN_X, green.X(), 0.5);
		Assert.assertEquals(GREEN_Y, green.Y(), 0.5);
		Assert.assertEquals(GREEN_Z, green.Z(), 0.5);
	}

	@Test
	public void BGRtoXYZBlue() {
		Color blue = Color.ofBGR(new double[] {RGB_MAX, 0.0, 0.0});
		Assert.assertEquals(BLUE_X, blue.X(), 0.5);
		Assert.assertEquals(BLUE_Y, blue.Y(), 0.5);
		Assert.assertEquals(BLUE_Z, blue.Z(), 0.5);
	}

	@Test
	public void XYZtoBGRRed() {
		Color red = Color.ofXYZ(new double[] {RED_X, RED_Y, RED_Z});
		Assert.assertEquals(RGB_MAX, red.red(), 0.5);
		Assert.assertEquals(0.0, red.green(), 0.5);
		Assert.assertEquals(0.0, red.blue(), 0.5);
	}

	@Test
	public void XYZtoBGRGreen() {
		Color green = Color.ofXYZ(new double[] {GREEN_X, GREEN_Y, GREEN_Z});
		Assert.assertEquals(0.0, green.red(), 0.5);
		Assert.assertEquals(RGB_MAX, green.green(), 0.5);
		Assert.assertEquals(0.0, green.blue(), 0.5);
	}

	@Test
	public void XYZtoBGRBlue() {
		Color blue = Color.ofXYZ(new double[] {BLUE_X, BLUE_Y, BLUE_Z});
		Assert.assertEquals(0.0, blue.red(), 0.5);
		Assert.assertEquals(0.0, blue.green(), 0.5);
		Assert.assertEquals(RGB_MAX, blue.blue(), 0.5);
	}
}
