package seedcounter.regression;

public class RegressionFactory {
	public static RegressionModel createModel(ColorSpace colorSpace,
			Order order, boolean intercept) {
		if (order == Order.IDENTITY) {
			return new IdentityModel();
		}
		if (colorSpace == ColorSpace.RGB) {
			switch(order) {
				case FIRST: return new SimpleRGB(intercept);
				case SECOND: return new SecondOrderRGB(intercept);
				default: return new ThirdOrderRGB(intercept);
			}
		} else {
			switch(order) {
				case FIRST: return new SimpleXYZ(intercept);
				case SECOND: return new SecondOrderXYZ(intercept);
				default: return new ThirdOrderXYZ(intercept);
			}
		}
	}
	
	public enum ColorSpace {
		RGB,
		XYZ
	}

	public enum Order {
		IDENTITY,
		FIRST,
		SECOND,
		THIRD
	}
}
