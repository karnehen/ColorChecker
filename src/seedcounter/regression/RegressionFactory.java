package seedcounter.regression;

public class RegressionFactory {
    public static RegressionModel createModel(Order order) {
        if (order == Order.IDENTITY) {
            return new IdentityModel();
        }
        switch(order.order) {
            case 1: return new SimpleOLS(order.intercept);
            case 2: return new SecondOrderOLS(order.intercept);
            default: return new ThirdOrderOLS(order.intercept);
        }
    }

    public enum Order {
        IDENTITY(0, false),
        FIRST(1, false),
        FIRST_INTERCEPT(1, true),
        SECOND(2, false),
        SECOND_INTERCEPT(2, true),
        THIRD(3, false),
        THIRD_INTERCEPT(3, true);

        public final int order;
        public final boolean intercept;

        Order(int order, boolean intercept) {
            this.order = order;
            this.intercept = intercept;
        }
    }
}
