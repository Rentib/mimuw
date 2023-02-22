package pl.edu.mimuw.matrix;

public class ConstDoubleMatrix extends DoubleMatrix {
    double xd;
    public ConstDoubleMatrix(Shape shape, double lol) {
        this.shape = shape;
        this.xd    = lol;
    }
    public IDoubleMatrix times(IDoubleMatrix other) {
        assert(this.shape().columns == other.shape().rows);
        Shape ns = Shape.matrix(shape.rows, other.shape().columns);
        if (other.getClass() == IdentityDoubleMatrix.class) return this;
        return xd == 0 ? new ConstDoubleMatrix(ns, 0)
                       : FullDoubleMatrix.mult(this, other);
    }
    public IDoubleMatrix times(double scalar) {
        return new ConstDoubleMatrix(shape, xd * scalar);
    }
    public IDoubleMatrix plus(IDoubleMatrix other) {
        assert(shape.equals(other.shape()));
        return this.getClass() == other.getClass()
                ? new ConstDoubleMatrix(shape, this.get(0, 0) * other.get(0, 0))
                : xd == 0 ? other : other.plus(xd);
    }
    public IDoubleMatrix plus(double scalar) {
        return new ConstDoubleMatrix(shape, xd + scalar);
    }
    public double get(int row, int column) {
        shape.assertInShape(row, column);
        return xd;
    }
    public double normOne() { return shape.rows * Math.abs(xd); }
    public double normInfinity() { return shape.columns * Math.abs(xd); }
    public double frobeniusNorm() {
        return Math.sqrt(shape.rows * shape.columns * Math.abs(xd));
    }
    @Override public String toString() {
        return (xd == 0 ? "Type: zero\nShape: " : "Type: const\nShape: ") + shape.rows + "x" + shape.columns
                + (xd == 0 ? "" : "Value: " + xd) + "\n";
    }
}