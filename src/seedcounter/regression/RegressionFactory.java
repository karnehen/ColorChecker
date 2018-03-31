package seedcounter.regression;

import seedcounter.ColorSpace;

public class RegressionFactory {
	public static RegressionModel createModel(Order order, boolean intercept) {
		if (order == Order.IDENTITY) {
			return new IdentityModel();
		}
	 	switch(order) {
			case FIRST: return new SimpleOLS(intercept);
			case SECOND: return new SecondOrderOLS(intercept);
			default: return new ThirdOrderOLS(intercept);
		}
	}

	public enum Order {
		IDENTITY,
		FIRST,
		SECOND,
		THIRD
	}
}
